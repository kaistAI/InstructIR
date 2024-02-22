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

    og_query2id = {v.split('[SEP]')[1].strip() : k.split('_')[0].strip() for k,v in queries.items()}
    print(len(og_query2id))
    
    # og q -> all ids
    ogq2qid = defaultdict(list)
    for k,v in queries.items():
        ogq = v.split('[SEP]')[1].strip()
        ogq2qid[ogq].append(k)

    print("len(ogq2qid):",len(ogq2qid))

    sentence_combinations_for_only_q_t = []

    for q, qids in ogq2qid.items():
        pids_rest = [( list(qrels[_qid].keys())[0] ,corpus[list(qrels[_qid].keys())[0]], _qid ) for _qid in qids]
        sentence_combinations_for_only_q_t.append(
            (og_query2id[q], q,pids_rest )
        )

    print(len(sentence_combinations_for_only_q_t),len(sentence_combinations_for_only_q_t[0][2]))


    system_prompt="You are a helpful and respectful assistant."


    client = OpenAI(
        api_key=args.open_ai_api_key,
    )

    model = 'gpt-4-1106-preview'
    temperature=0.7

    USER_INSTRUCTION = """You are a similarity evaluator! You're tasked with calculating the similarity between QUERY, and DOCUMENT displayed below based on their relevancy. In the evaluation, I want you to rate the relevancy of the pair according to the following score rubric:

    Score 1: The DOCUMENT, QUERY have very little or no relevance to each other. The elements compared share almost no common attributes or context.
    Score 2: The DOCUMENT, QUERY have some relevance but are quite distinct. They share a few attributes or contextual details, but there are significant differences in the majority of aspects.
    Score 3: The DOCUMENT, QUERY are moderately relevant to each other. They share a fair amount of attributes or context, but there are still some notable differences that prevent a high similarity score.
    Score 4: The DOCUMENT, QUERY have high relevance to each other. They share many attributes or contextual details, with only a few differences that do not majorly impact the overall similarity.
    Score 5: The DOCUMENT, QUERY are very highly relevant or almost identical to each other. They share nearly all attributes or the context is almost exactly the same, with very minor or negligible differences.

    You will be given QUERY and DOCUMENT pair.
    [QUERY]
    {query}

    [DOCUMENT]
    {candidate document}

    You should generate based on the following format:

    <Explanation>
    {explanation for the score}
    </Explanation>

    <Score>
    {score}
    </Score>

    Please give feedback on the DOCUMENT with respect to each QUERY, and provide a score on a scale of 1 to 5 whether it satisfies the requirements, where a higher score indicates better performance.
    """

    ranked_results = []

    for idx, pair in tqdm(enumerate(sentence_combinations_for_only_q_t)):
        query = pair[1]
        for inst in pair[2]:
            user_instruction=USER_INSTRUCTION +\
                f'\n[QUERY]\n{pair[1].strip()}\n[DOCUMENT]\n{inst[1]["text"]}'
            
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
                "og_query_id": pair[0],
                "query_id": inst[2],
                "pid":inst[0],
                "query": pair[1],
                "revised_target":inst[1],
                "generated_text": response,
                "metadata": {
                    "input_type": 'filtering_c1', 
                    "model": model, 
                    "temparature": temperature,
                    }
            })


    print(f"Processing Done - total: {len(ranked_results)}")

    os.makedirs(args.save_path,exist_ok=True)
    with open(os.path.join(args.save_path,"raw_dataset.step4_q1.json"),'w') as f:
        f.write(json.dumps(ranked_results, indent=4, ensure_ascii=False))


    # * Post processing
    processed_ranked_results = []
    error=[]
    for pair in ranked_results:
        try:
            explanation = pair['generated_text'].lower().split('<explanation>')[1].split('</explanation>')[0].strip()
            score = pair['generated_text'].lower().split('<score>')[1].split('</score>')[0].strip()

            score = int(score) if len(score)==1 else int(score.split('[')[1].split(']')[0].strip())
        except Exception as e:
            print(e)
            print(pair)
            error.append(pair)
            continue
                
        processed_ranked_results.append({
            "og_query_id": pair['og_query_id'],
            "query_id": pair['query_id'],
            "query": pair['query'],
            "revised_target":pair['revised_target'],
            "score":int(score),
            "explanation":explanation,
            "metadata": {
                "input_type": "filtering_c1",
                "model": pair['metadata']['model']
                }
        })
        

    with open(os.path.join(args.save_path,"processed_dataset.step4_q1.json"),'w') as f:
        f.write(json.dumps(processed_ranked_results, indent=4, ensure_ascii=False))

    print("len(processed_ranked_results),processed_ranked_results[0], len(error):",len(processed_ranked_results),processed_ranked_results[0] if processed_ranked_results else 'None', len(error))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--open_ai_api_key", type=str,required=True,help="OPENAI_API_KEY")
    parser.add_argument("--generated_data_path", type=str, default="generated_data/before_filtering/" ,help="generated dataset folder after stage 3")
    parser.add_argument("--save_path", type=str, default="generated_data/gpt4_filtered_results" ,help="save path")
    
    args, _ = parser.parse_known_args()
    main(args)