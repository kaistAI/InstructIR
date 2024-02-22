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

    client = OpenAI(
        api_key=args.open_ai_api_key,
    )

    step2_prompt ="You are a helpful, respectful and creative assistant."
    model = 'gpt-4-1106-preview'
    temperature=1.0

    results = []

    for qid, _ in tqdm(qrels.items()):
        query = queries[qid]
        
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= 3:
            try:
                description = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": step2_prompt.strip()},
                        {
                            "role": "user", 
                            "content": 
                            """Your task is to generate a set of scenarios for the provided search query.
                            Here is the specification for the scenario generation task: 
                            - The scenario should reflect a very specific scenario where a user is interacting with an AI search engine. 
                            - Within the scenario, the user could write about his/her job, background, situation, location, occupation, hobbies, interests, or goals of doing the search. Also, the user could explicitly reflect about his/her preference regarding the document to be searched.
                            - The scenario SHOULD be written from a first person’s view point. For example, it should start with phrases like “I am a {job},”, “I am in a situation ….”, “During my {situation}”.
                            - While the provided query is about "what" is being searched for, the scenario you will generate should be about "how" the search should be approached and what values or criteria should be prioritized in that search. This distinction makes the role of the scenario clear as a guiding framework for the search process, as opposed to the query which is more about the specific target of the search.
                            - However, the scenario should be RELATED with the provided query. In other words, it shouldn’t be applicable to other queries in general.
                            - You should generate based on the following format (note that there is a phrase “[END]” after each scenario being generated):
                            Scenario 1: {generate the first scenario} [END]
                            Scenario 2: {generate the second scenario} [END]

                            Scenario 10: {generate the last scenario} [END]
                            - Please do not generate any other opening, closing, and explanations. Just generate the set of scenarios!""" + f"\nQuery: {query['text']}"
            }
                    ],
                    temperature=temperature,
                )

                response = description.choices[0].message.content
                # print("Query:\n",query)
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

        results.append({
            "query_id": qid,
            "input": query['text'],
            "generated_text": response,
            "metadata": {
                "input_type": 'step2', 
                "model": model, 
                "temparature": temperature,
                }
        })

    print(f"Processing Done - total: {len(results)}")

    os.makedirs(args.save_path,exist_ok=True)
    with open(os.path.join(args.save_path, "raw_dataset.step2_generate_instruction.json"),"w") as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))


    # * Post processing
    processed_results = []
    for pair in results:
        scenarios= []
        split_lines = pair['generated_text'].split('[END]')
        for line in split_lines:
            if line.strip():
                output = line.split(':',1)[1].strip()
                if 'Scenario' in line:
                    scenarios.append(output)    
                
        processed_results.append({
            "query_id": pair['query_id'],
            "query": pair['input'],
            "instructions": [scene for scene in scenarios],
        })

    print(len(processed_results))

    with open(os.path.join(args.save_path, "processed_dataset.step2_generate_instruction.json"),"w") as f:
        f.write(json.dumps(processed_results, indent=4, ensure_ascii=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--open_ai_api_key", type=str,required=True,help="OPENAI_API_KEY")
    parser.add_argument("--data_path", type=str, default="../data/MSMARCO/beir_dataset/msmarco/filtered_version" ,help="filtered dataset folder after stage 1")
    parser.add_argument("--save_path", type=str, default="generated_data/step2_generate_instruction" ,help="save path")

    args, _ = parser.parse_known_args()
    main(args)