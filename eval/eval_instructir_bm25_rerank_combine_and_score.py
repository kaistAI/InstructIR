'''
Reformulate scripts from https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_anserini_bm25.py 
'''
import os, json
import tqdm
import logging
import requests
import argparse 
from beir.datasets.data_loader import GenericDataLoader
import os 
import pickle 
from beir import util, LoggingHandler
from collections import defaultdict

from robust_eval import CustomEvaluateRetrieval
from src.rerankers import MODEL_DICT

def main(args):
    results = {}
    for shard_i in range(args.n_shards):
        model_name = args.model_name.replace("/", "-")
        with open(f"model_pred/{args.rerank_model}_{model_name}/{shard_i}_{args.n_shards}.pickle", "rb") as f:
            loaded_results = pickle.load(f)
            results.update(loaded_results)

    corpus, queries, qrels = GenericDataLoader( 
        data_folder=args.data_path,
        corpus_file=args.corpus_file ,
        qrels_file= os.path.join(args.data_path, args.qrels_folder,args.split + '.tsv'),
        query_file=args.query_file,
        ).load_custom()
    
    retriever = CustomEvaluateRetrieval()

    #### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
    # results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

    ### Check if query_id is in results i.e. remove it from docs incase if it appears ####
    ### Quite Important for ArguAna and Quora ####
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    # metrics = retriever.evaluate(qrels, results, retriever.k_values)
    # metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values,type='instruction')
    metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values,type='query')
    print(metrics)
    breakpoint()
    #### Save
    # data_path = args.data_path.split('/')[-2]
    # corpus_file = args.corpus_file.split('.')[0]
    # split_flag = args.split
    # query_version = args.query_file.split('.')[0].replace('/','_')
    # save_path = f'model_pred/{data_path}/{corpus_file}/{split_flag}/{query_version}/'

    # os.makedirs(save_path, exist_ok=True)
    # save_path += 'bm25.pickle'
    # with open(save_path,'wb') as f:
    #     pickle.dump(results,f)
        
if __name__=="__main__":    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default='../data_creation/datasets/msmarco/processed_version/sample_100',
                        help="Directory to save and load beir datasets")
    parser.add_argument("--split", type=str,default='test',
                        help="split for qrels")    
    parser.add_argument("--corpus_file", type=str, default='corpus.jsonl',
                        help="corpus file name")
    parser.add_argument("--query_file", type=str, default='queries.jsonl',
                        help="query file name")
    parser.add_argument("--qrels_folder", type=str, default='qrels',
                        help="qrels folder name")
    parser.add_argument("--model_name", type=str, default='jhu-clsp/FollowIR-7B')
    parser.add_argument("--rerank_n", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--rerank_model", type=str, default='bm25')
    parser.add_argument("--change_instruction_order", action="store_true", help="change the order of instructions and query")
    cfg, _ = parser.parse_known_args()
    print(cfg)

    main(cfg)


