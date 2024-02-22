from tqdm import tqdm 
from openai import OpenAI
import random 
from beir import util, LoggingHandler
from custom_data_loader import GenericDataLoader
import os
import json
import pickle 
import openai
import time 
import argparse 


def main(args):
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split="test")
    print("len(corpus),len(queries),len(qrels):",len(corpus),len(queries),len(qrels)) 

    with open(args.step2_output_file,'r') as f:
        data = json.load(f)
    print("len(data),data[0].keys():",len(data),data[0].keys())

    input_for_step3= []
    for pair in data:
        pid = list(qrels[pair['query_id']].keys())[0] 
        og_target = corpus[pid]
        og_target.update({"_id": pid})
        pair.update({"origin_target" : og_target})
        input_for_step3.append(pair)
        
    print("len(input_for_step3):",len(input_for_step3))

    client = OpenAI(
        api_key=args.open_ai_api_key,
    )

    step3_prompt = "You are a helpful, respectful and creative assistant."
    model = 'gpt-4-1106-preview'
    temperature=1.0

    step3_results = []

    for pair in tqdm(input_for_step3):
        qid = pair['query_id']
        query = pair['query']
        og_target = pair['origin_target']
        instructions = pair['instructions']

        for instruction in instructions:
            retry_cnt = 0
            backoff_time = 30
            while retry_cnt <= 3:
                try:    
                    description = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": step3_prompt.strip()},
                            {"role": "user", "content": 
                            """Your task is to generate a REVISED DOCUMENT for the provided search QUERY and SCENARIO pair.
                            Here is the specification for the DOCUMENT revising task:
                            - The REVISED DOCUMENT should reflect the user's unique SCENARIO where a user is interacting with an AI search engine.
                            - Within the REVISED DOCUMENT, revise details reflecting the user’s background, situation, location, occupation, hobbies, interests, or goals of doing the search. Also, containing information related to the user’s preference is important. 
                            - Directly revise given DOCUMENT that has good quality that can be found by an AI search engine. Don’t just suggest it!
                            - Do NOT include the same keywords from the given SCENARIO in REVISED DOCUMENT. Paraphrase it. 
                            - However, the REVISED DOCUMENT should be RELATED with the provided query. In other words, it should be applicable to query in general.
                            - You should generate based on the following format (note that there is a phrase “[END]” after each elements being generated):

                            PLAN: {generate the plan for the strategy for revision} [END]
                            REVISED DOCUMENT: {revise the document} [END]

                            - Please do not generate any other opening, closing, and explanations. Just generate the PLAN and REVISED TARGET !
                            """ 
                            + 
                            f"QUERY:\n{query}\nSCENARIO: {instruction}\nDOCUMENT: {og_target['text']}"}                        
                        ],
                        temperature=temperature,
                    )

                    response = description.choices[0].message.content
                    # print("Query:", query)
                    # print("instruction:\n",instruction)
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


            step3_results.append({
                "query_id": qid,
                "query": query,
                "instruction": instruction,
                "original_target": og_target,
                "generated_text": response,
                "metadata": {
                    "input_type": 'step3', 
                    "model": model, 
                    "temparature": temperature,
                    }
            })


    print(f"Processing Done - total: {len(step3_results)}")

    os.makedirs(args.save_path,exist_ok=True)
    with open(os.path.join(args.save_path,"raw_dataset.step3_generate_instruction.json"),"w") as f:
        f.write(json.dumps(step3_results, indent=4, ensure_ascii=False))


    # * Post processing
    processed_step3_results = []
    error=[]
    for pair in step3_results:
        for l in pair['generated_text'].split('\n'):
            try:
                if 'PLAN:' in l:
                    plan = l.split('PLAN:')[1].split('[END]')[0].strip()
                elif 'REVISED DOCUMENT:' in l:
                    revised_target = l.split('DOCUMENT:')[1].split('[END]')[0].strip()

            except Exception as e:
                print(e)
                print(pair)
                error.append(pair)
                
        processed_step3_results.append({
            "query_id": pair['query_id'],
            "query": pair['query'],
            "instruction": pair['instruction'],
            "revised_target":revised_target,
            "original_target":pair['original_target'],
            "metadata": {
                "input_type": "step3",
                "plan": plan,
                }
        })
        
    print("len(processed_step3_results),processed_step3_results[0], len(error):",len(processed_step3_results),processed_step3_results[0], len(error))

    with open(os.path.join(args.save_path,"processed_dataset.step3_generate_instruction.json"),"w") as f:
        f.write(json.dumps(processed_step3_results, indent=4, ensure_ascii=False))

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--open_ai_api_key", type=str,required=True,help="OPENAI_API_KEY")
    parser.add_argument("--data_path", type=str, default="../data/MSMARCO/beir_dataset/msmarco/filtered_version" ,help="filtered dataset folder after stage 1")
    parser.add_argument("--step2_output_file", type=str, default="generated_data/step2_generate_instruction/processed_dataset.step2_generate_instruction.json" ,help="filtered dataset folder after stage 1")
    parser.add_argument("--save_path", type=str, default="generated_data/step3_revise_target" ,help="save path")
    
    args, _ = parser.parse_known_args()
    main(args)