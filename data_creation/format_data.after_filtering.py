import json 
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import os 
from collections import defaultdict, Counter 
import pandas as pd 
from tqdm import tqdm 
import argparse 


def main(args):
    # Read filtered outputs for step4 -Q1, Q2
    with open(os.path.join(args.filtered_path, 'processed_dataset.step4_q1.json'),'r') as f:
        q1_data = json.load(f)
    
    with open(os.path.join(args.filtered_path, 'filtered_processed_dataset.step4_q2.json'),'r') as f:
        q2_data = json.load(f)

    print(len(q1_data), q1_data[0].keys())
    print(len(q2_data), q2_data[0].keys())

    q1_qid_list = [pair['query_id'] for pair in q1_data]
    q2_qid_list = [pair['query_id'] for pair in q2_data]
    overlap_qid_list = list(set(q1_qid_list) & set(q2_qid_list))

    print("len(q1_qid_list),len(q2_qid_list),len(overlap_qid_list):",len(q1_qid_list),len(q2_qid_list),len(overlap_qid_list))

    # Read beir formatted datasets before filtering
    corpus_file = 'corpus.jsonl'
    qrels_file ='test'
    corpus, queries, qrels = GenericDataLoader(
        data_folder=args.before_filtering_folder,
        corpus_file=corpus_file ,
        qrels_file= os.path.join(args.before_filtering_folder, 'qrels',qrels_file + '.tsv')
        ).load_custom()
    
    print("len(corpus),len(queries),len(qrels):",len(corpus),len(queries),len(qrels))

    remain_queries = {qid:text for qid, text in queries.items() if qid in overlap_qid_list}
    remain_qrels = {qid: pid_dict for qid, pid_dict in qrels.items() if qid in overlap_qid_list}
    remain_selected_pid = [list(pid_dict.keys())[0] for qid, pid_dict in qrels.items() if qid in overlap_qid_list]
    remain_corpus = {pid:info for pid, info in corpus.items() if pid in remain_selected_pid}

    print("len(remain_queries), len(remain_qrels), len(remain_selected_pid), len(remain_corpus):",\
          len(remain_queries), len(remain_qrels), len(remain_selected_pid), len(remain_corpus))

    if args.og_path and os.path.exists(args.og_path):
        og_path = args.og_path
    else:
        dataset = "msmarco"
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        print(url)
        out_dir = os.path.join('data/MSMARCO/', "beir_dataset")
        og_path = util.download_and_unzip(url, out_dir)
    
    og_corpus, og_queries, og_qrels = GenericDataLoader(data_folder=og_path).load(split='dev')
    
    subset_corpus={}
    for qid, pid_dict in og_qrels.items():
        for pid in pid_dict.keys():
            subset_corpus[pid]=og_corpus[pid]
    print("len(subset_corpus):",len(subset_corpus))

    selected_og_pid = set([pid.split('_')[0] for pid in remain_corpus.keys() if '_' in pid])
    print("len(selected_og_pid),list(selected_og_pid)[:10]:",len(selected_og_pid),list(selected_og_pid)[:10])

    query2id = {v.strip():k for k,v in remain_queries.items()}
    corpus2id = {v['text'].strip():k for k,v in remain_corpus.items()}

    print(len(query2id),len(corpus2id))

    # og q -> all ids
    ogq2qid = defaultdict(list)
    for k,v in remain_queries.items():
        ogq = v.split('[SEP]')[1].strip()
        ogq2qid[ogq].append(k)

    print(len(ogq2qid))

    group_per_query = {} 
    for qid,text in remain_queries.items():
        og_qid = qid.split('_')[0]
        if not group_per_query.get(og_qid):
            group_per_query[og_qid]=[]
        group_per_query[og_qid].append({qid:text})

    print("len(group_per_query), pd.DataFrame([ len(v) for k,v in group_per_query.items()]).describe():",
          len(group_per_query), pd.DataFrame([ len(v) for k,v in group_per_query.items()]).describe())

    # We select groups that have more than 6 instances with different instructions
    filtered_group_per_query = {q:v for q,v in group_per_query.items() if len(v)>=6}
    print("len(filtered_group_per_query), Counter([len(v) for q,v in filtered_group_per_query.items()]), pd.DataFrame([ len(v) for k,v in filtered_group_per_query.items()]).describe():",\
          len(filtered_group_per_query), Counter([len(v) for q,v in filtered_group_per_query.items()]), pd.DataFrame([ len(v) for k,v in filtered_group_per_query.items()]).describe())

    new_corpus = {} 
    new_queries = {}
    instruction_only_queries = {}
    w_o_instruction_queries = {}
    new_qrels= {} 
    w_o_instruction_qrels= {} 

    for og_qid, pairs in filtered_group_per_query.items():
        for idx,pair in enumerate(pairs):
            prev_qid = list(pair.keys())[0]
            prev_text = list(pair.values())[0]
            prev_pid = list(qrels[prev_qid].keys())[0]
            prev_corpus_info = corpus[prev_pid]
            og_target_info = prev_corpus_info['metadata']['origin_target']
            target_info = prev_corpus_info['text']

            qid= f"{og_qid}_{idx+1}"
            pid = f"{og_target_info['_id']}_{idx+1}"

            new_corpus[pid]= {
                "_id": pid,
                "text": target_info,
                "title": "",
                "metadata": {
                    "origin_target": og_target_info
                }
            }

            new_queries[qid] = {
                "_id": qid,
                "text": prev_text,
                "metadata": {
                    "origin_query": prev_text.split('[SEP]')[1].strip()
                }
            }

            if not instruction_only_queries.get(qid):
                instruction_only_queries[qid] = {
                    "_id": qid,
                    "text": prev_text.split('[SEP]')[0].strip(),
                    "metadata": {
                        "origin_query": prev_text.split('[SEP]')[1].strip()
                    }
                }

            if not w_o_instruction_queries.get(og_qid):
                w_o_instruction_queries[og_qid] = {
                    "_id": og_qid,
                    "text": prev_text.split('[SEP]')[1].strip(),
                    "metadata": {}
                }

            if not new_qrels.get(qid):
                new_qrels[qid] = {}
            new_qrels[qid][pid]=1

            if not w_o_instruction_qrels.get(og_qid):
                w_o_instruction_qrels[og_qid] = {}
            w_o_instruction_qrels[og_qid][pid]=1
            
    print("len(new_corpus),len(new_queries),len(instruction_only_queries), len(w_o_instruction_queries) ,len(new_qrels),len(w_o_instruction_qrels):"\
          ,len(new_corpus),len(new_queries),len(instruction_only_queries), len(w_o_instruction_queries) ,len(new_qrels),len(w_o_instruction_qrels))

    os.makedirs(args.save_path,exist_ok=True)
            
    with open(os.path.join(args.save_path,'corpus.jsonl'),'w') as f:
        for k,v in tqdm(subset_corpus.items(),desc='corpus'):
            if k not in selected_og_pid:
                temp = {'_id':k}
                temp.update(v)
                f.write(json.dumps(temp)+'\n')
            else:
                for i in range(1,11):
                    revised_target = new_corpus.get(f'{k}_{i}')
                    if revised_target:
                        f.write(json.dumps(revised_target)+'\n')
        
    
    with open(os.path.join(args.save_path,'queries.jsonl'),'w') as f:
        for pair in list(new_queries.values()):
            f.write(json.dumps(pair)+'\n')
        

    with open(os.path.join(args.save_path,'only_instruction_queries.jsonl'),'w') as f:
        for pair in list(instruction_only_queries.values()):
            f.write(json.dumps(pair)+'\n')

    with open(os.path.join(args.save_path,'only_queries.jsonl'),'w') as f:
        for pair in list(w_o_instruction_queries.values()):
            f.write(json.dumps(pair)+'\n')

    os.makedirs(os.path.join(args.save_path,'qrels'),exist_ok=True)

    with open(os.path.join(args.save_path,'qrels','test.tsv'),'w') as f:
        f.write(f"qid\tpid\tscore\n")
        for qid, pid_dict in new_qrels.items():
            for pid,score in pid_dict.items():
                f.write(f"{qid}\t{pid}\t{score}\n")
        
    with open(os.path.join(args.save_path,'qrels','for_only_query_test.tsv'),'w') as f:
        f.write(f"qid\tpid\tscore\n")
        for qid, pid_dict in w_o_instruction_qrels.items():
            for pid,score in pid_dict.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    print("Write Done!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--before_filtering_folder", type=str, default="generated_data/before_filtering" ,help="beir formatted dataset folder before filtering")
    parser.add_argument("--filtered_path", type=str, default="generated_data/gpt4_filtered_results" ,help="filtered dataset after data creation pipeline step 4")
    parser.add_argument("--og_data_path", type=str, default="./data/MSMARCO/beir_dataset/msmarco" ,help="seed dataset folder for MSMARCO dataset")
    parser.add_argument("--save_path", type=str, default="generated_data/after_filtering" ,help="save path")

    args, _ = parser.parse_known_args()
    main(args)