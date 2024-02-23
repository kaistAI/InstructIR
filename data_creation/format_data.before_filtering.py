import json 
import pandas as pd 
import os 
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm 
import argparse


def main(args):
    # Read processed files after step3
    with open(args.step3_output,'r') as f:
        data = json.load(f)
    print("len(data),data[0].keys()):",len(data),data[0].keys()) #  dict_keys(['query_id', 'query', 'instruction', 'revised_target', 'original_target', 'metadata']))

    group_per_query = {} 
    for i in data:
        if not group_per_query.get(i['query']):
            group_per_query[i['query']]=[]
        group_per_query[i['query']].append(i)
        
    print("len(group_per_query), pd.DataFrame([ len(v) for k,v in group_per_query.items()]).describe():",\
          len(group_per_query), pd.DataFrame([ len(v) for k,v in group_per_query.items()]).describe())

    corpus = {} 
    queries = {}
    instruction_only_queries = {}
    w_o_instruction_queries = {}
    qrels= {} 
    w_o_instruction_qrels= {} 

    for query, pairs in group_per_query.items():
        og_target_info = pairs[0]["original_target"]

        for idx,pair in enumerate(pairs):
            og_qid = pair['query_id']
            qid= f"{pair['query_id']}_{idx+1}"
            pid = f"{og_target_info['_id']}_{idx+1}"

            corpus[pid]= {
                "_id": pid,
                "text": pair["revised_target"],
                "title": "",
                "metadata": {
                    "origin_target": og_target_info
                }
            }

            queries[qid] = {
                "_id": qid,
                "text": f"{pair['instruction']} [SEP] {pair['query']}",
                "metadata": {
                    "origin_query": pair['query']
                }
            }
            if not instruction_only_queries.get(qid):
                instruction_only_queries[qid] = {
                    "_id": qid,
                    "text": f"{pair['instruction']}",
                    "metadata": {
                        "origin_query": pair['query']
                    }
                }

            if not w_o_instruction_queries.get(og_qid):
                w_o_instruction_queries[og_qid] = {
                    "_id": og_qid,
                    "text": f"{pair['query']}",
                    "metadata": {
                        "origin_query": pair['query']
                    }
                }

            if not qrels.get(qid):
                qrels[qid] = {}
            qrels[qid][pid]=1

            if not w_o_instruction_qrels.get(og_qid):
                w_o_instruction_qrels[og_qid] = {}
            w_o_instruction_qrels[og_qid][pid]=1
            
    print("len(corpus),len(queries),len(instruction_only_queries), len(w_o_instruction_queries) ,len(qrels), len(w_o_instruction_qrels):",\
          len(corpus),len(queries),len(instruction_only_queries), len(w_o_instruction_queries) ,len(qrels), len(w_o_instruction_qrels))


    if args.og_data_path and os.path.exists(args.og_data_path):
        og_path = args.og_data_path
    else:
        dataset = "msmarco"
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        print(url)
        out_dir = os.path.join('data/MSMARCO/', "beir_dataset")
        og_path = util.download_and_unzip(url, out_dir)
        
    og_corpus, og_queries, og_qrels = GenericDataLoader(data_folder=og_path).load(split='dev')
    print("len(og_corpus),len(og_queries),len(og_qrels): ",len(og_corpus),len(og_queries),len(og_qrels)) # (8841823, 6980, 6980)

    subset_corpus={}
    for qid, pid_dict in og_qrels.items():
        for pid in pid_dict.keys():
            subset_corpus[pid]=og_corpus[pid]
    print("len(subset_corpus):",len(subset_corpus))

    selected_og_pid = set([pid.split('_')[0] for pid in corpus.keys()])
    print("len(selected_og_pid),list(selected_og_pid)[:10]:",len(selected_og_pid),list(selected_og_pid)[:10])

    os.makedirs(args.save_path ,exist_ok=True)

    with open(os.path.join(args.save_path ,'corpus.jsonl'),'w') as f:
        for k,v in tqdm(subset_corpus.items(),desc='corpus'):
            if k not in selected_og_pid:
                temp = {'_id':k}
                temp.update(v)
                f.write(json.dumps(temp)+'\n')
            else:
                for i in range(1,11):
                    revised_target = corpus.get(f'{k}_{i}')
                    if revised_target:
                        f.write(json.dumps(revised_target)+'\n')
        
    with open(os.path.join(args.save_path ,'queries.jsonl'),'w') as f:
        for pair in list(queries.values()):
            f.write(json.dumps(pair)+'\n')

    with open(os.path.join(args.save_path ,'only_instruction_queries.jsonl'),'w') as f:
        for pair in list(instruction_only_queries.values()):
            f.write(json.dumps(pair)+'\n')

    with open(os.path.join(args.save_path ,'only_queries.jsonl'),'w') as f:
        for pair in list(w_o_instruction_queries.values()):
            f.write(json.dumps(pair)+'\n')

    os.makedirs(os.path.join(args.save_path ,'qrels'),exist_ok=True)

    with open(os.path.join(args.save_path ,'qrels','test.tsv'),'w') as f:
        f.write(f"qid\tpid\tscore\n")
        for qid, pid_dict in qrels.items():
            for pid,score in pid_dict.items():
                f.write(f"{qid}\t{pid}\t{score}\n")
        
    with open(os.path.join(args.save_path ,'qrels','for_only_query_test.tsv'),'w') as f:
        f.write(f"qid\tpid\tscore\n")
        for qid, pid_dict in w_o_instruction_qrels.items():
            for pid,score in pid_dict.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    print("Write Done!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--step3_output", type=str, default="generated_data/step3_revise_target/processed_dataset.step3_generate_instruction.json" ,help="processed dataset after data creation pipeline step 3")
    parser.add_argument("--og_data_path", type=str, default="./data/MSMARCO/beir_dataset/msmarco" ,help="seed dataset folder for MSMARCO dataset")
    parser.add_argument("--save_path", type=str, default="generated_data/before_filtering" ,help="save path")

    args, _ = parser.parse_known_args()
    main(args)