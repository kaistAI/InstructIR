from beir.datasets.data_loader import GenericDataLoader
import os 
import json
import json 
from tqdm import tqdm 
from openai import OpenAI
import openai
import time 
from collections import defaultdict 
import argparse

def main(args):
    corpus, queries, qrels = GenericDataLoader(data_folder=args.generated_data_path).load(split='test')
    print(len(corpus),len(queries),len(qrels))

    # og q -> all ids
    ogq2qid = defaultdict(list)
    for k,v in queries.items():
        ogq = v.split('[SEP]')[1].strip()
        ogq2qid[ogq].append(k)

    print("len(ogq2qid):",len(ogq2qid))

    sentence_combinations_all = []

    for q, qids in ogq2qid.items():
        pids_rest = [(list(qrels[_qid].keys())[0],corpus[list(qrels[_qid].keys())[0]]) for _qid in qids]
        
        for qid in qids:
            sentence_combinations_all.append(
                (qid,queries[qid],pids_rest)
            )

    print(len(sentence_combinations_all), len(sentence_combinations_all[0][2]))


    system_prompt="You are a helpful and respectful assistant."

    client = OpenAI(
        api_key=args.open_ai_api_key,
    )

    model = 'gpt-4-1106-preview'
    temperature=0.7

    USER_INSTRUCTION = """You are a ranker agent! Each potential DOCUMENT has a corresponding DOCUMENT id and you're tasked with ranking the answers based on their relevancy to the pair of QUERY, SCENARIO pair. In the evaluation, I want you to rate the relevancy of the pair according to the following score rubric:

    Score 1: The DOCUMENT lacks relevance to the user's SCENARIO, providing little to no connection to the user's job, background, situation, location, occupation, hobbies, interests, or goals. It fails to consider preferences and context, resulting in an overall inadequate fit.
    Score 2: The DOCUMENT has limited relevance, with only a few elements aligning with the user's SCENARIO and QUERY. While some contextual understanding and preference consideration may be present, it falls short of providing a comprehensive and well-fitted response.
    Score 3: The DOCUMENT demonstrates moderate relevance, capturing some aspects of the user's SCENARIO. It shows an adequate contextual fit and considers a majority of the user's stated preferences. However, there is room for improvement in terms of depth and clarity.
    Score 4: The DOCUMENT exhibits high relevance, aligning well with the user's SCENARIO and covering most relevant aspects. It demonstrates a strong contextual fit, addresses the user's preferences effectively, and maintains high clarity and conciseness. However, there may be minor areas for improvement.
    Score 5: The DOCUMENT is perfectly relevant, precisely addressing all aspects of the user's SCENARIO, QUERY, and preferences. It seamlessly integrates with the user's context, demonstrating a profound understanding. The DOCUMENT is exceptionally clear, concise, and exhaustive in providing information, offering a flawless fit.

    You SHOULD ONLY generate the top ranked id from the given search DOCUMENT (id : 1~10) and no additional comments as [id]. This is VERY IMPORTANT!

    You will be given list of DOCUMENT and a pair of QUERY,SCENARIO.

    [DOCUMENT LIST]
    [1] {DOCUMENT_1}
    [2] {DOCUMENT_2}
    [3] {DOCUMENT_3}
    ...
    [10] {DOCUMENT_10}

    [SCENARIO]
    {scenario}

    [QUERY]
    {query}


    You should generate based on the following format:
    <Explanation>
    {explanation for the ranking}
    </Explanation>

    <Ranking>
    {top ranked DOCUMENT id}
    </Ranking>

    Please give a top ranked DOCUMENT id with respect to each SCENARIO and QUERY pair, and provide a score on a scale of 1 to 5 whether it satisfies the requirements, where a higher score indicates better performance. 
    """

    ranked_results = []

    for idx, pair in tqdm(enumerate(sentence_combinations_all)):
        query = pair[0]
        user_instruction=USER_INSTRUCTION +\
            '[DOCUMENT LIST]\n'+'\n'.join([ f'[{id+1}] ' + tgt[1]['text'] for id, tgt in enumerate(pair[2])])+\
            f'\n[SCENARIO]\n{pair[1].split("[SEP]")[0].strip()}\n[QUERY]\n{pair[1].split("[SEP]")[1].strip()}'
        
        # print("user_instruction:\n",user_instruction)
        
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= 3:
            try:    
                description = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_instruction},
                    ],
                    temperature=temperature,
                )

                response = description.choices[0].message.content
                # print("instruction:\n",user_instruction)
                # print("Respone:\n",response)
                break 
            except openai.APIError as e:
                print(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    target_length = int(target_length * 0.8)
                    print(f"Reducing target length to {target_length}, retrying...")
                else:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                retry_cnt += 1


        ranked_results.append({
            "query_id": pair[0],
            "gt_pid": list(qrels[pair[0]].keys())[0],
            "query_w_instruction": pair[1],
            "candidate_id_dict": {order+1: _p_info[0]  for order,_p_info in enumerate(pair[2]) },
            "generated_text": response,
            "metadata": {
                "input_type": 'filtering_c3', 
                "model": model, 
                "temparature": temperature,
                }
        })


    print(f"Processing Done - total: {len(ranked_results)}")

    os.makedirs(args.save_path,exist_ok=True)
    with open(os.path.join(args.save_path,"raw_dataset.step4_q2.json"),'w') as f:
        f.write(json.dumps(ranked_results, indent=4, ensure_ascii=False))

    # * Post processing
    processed_ranked_results = []
    error=[]
    for pair in ranked_results:
        try:
            explanation = pair['generated_text'].lower().split('<explanation>')[1].split('</explanation>')[0].strip()
            top1_rank = pair['generated_text'].lower().split('<ranking>')[1].split('</ranking>')[0].strip()

            top1_rank = int(top1_rank) if len(top1_rank)==1 else int(top1_rank.split('[')[1].split(']')[0].strip())
        except Exception as e:
            print(e)
            print(pair)
            error.append(pair)
            continue
                
        processed_ranked_results.append({
            "query_id": pair['query_id'],
            "gt_pid": pair['gt_pid'],
            "query_w_instruction": pair['query_w_instruction'],
            "candidate_id_dict":pair["candidate_id_dict"],
            "top1_rank":top1_rank,
            "flag": 1 if pair['candidate_id_dict'][int(top1_rank)]==pair['gt_pid'] else 0,
            "explanation":explanation,
            "metadata": {
                "input_type": "filtering_c3",
                "model": pair['metadata']['model']
                }
        })
        

    with open(os.path.join(args.save_path,"processed_dataset.step4_q2.json"),'w') as f:
        f.write(json.dumps(processed_ranked_results, indent=4, ensure_ascii=False))

    print("len(processed_ranked_results),processed_ranked_results[0], len(error):",len(processed_ranked_results),processed_ranked_results[0] if processed_ranked_results else 'None', len(error))


    filtered_reulsts= [pair for pair in processed_ranked_results if pair['flag']]
    print("filtered_reulsts:",len(filtered_reulsts))

    with open(os.path.join(args.save_path,"filtered_processed_dataset.step4_q2.json"),'w') as f:
        f.write(json.dumps(filtered_reulsts, indent=4, ensure_ascii=False))

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--open_ai_api_key", type=str,required=True,help="OPENAI_API_KEY")
    parser.add_argument("--generated_data_path", type=str, default="generated_data/before_filtering/" ,help="generated dataset folder after stage 3")
    parser.add_argument("--save_path", type=str, default="generated_data/gpt4_filtered_results" ,help="save path")
    
    args, _ = parser.parse_known_args()
    main(args)