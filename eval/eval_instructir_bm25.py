'''
Reformulate scripts from https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_anserini_bm25.py 
'''
import os, json
import logging
import requests
import argparse 
from beir.datasets.data_loader import GenericDataLoader
import os 
import pickle 
from beir import util, LoggingHandler

from robust_eval import CustomEvaluateRetrieval

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(args):
    corpus, queries, qrels = GenericDataLoader( 
        data_folder=args.data_path,
        corpus_file=args.corpus_file ,
        qrels_file= os.path.join(args.data_path, args.qrels_folder,args.split + '.tsv'),
        query_file=args.query_file,
        ).load_custom()
    
    logging.info(f"len(corpus),len(queries),len(qrels): {len(corpus),len(queries),len(qrels)}")

    #### Convert BEIR corpus to Pyserini Format #####
    pyserini_jsonl = f"pyserini_{args.corpus_file.split('.')[0]}.jsonl"
    
    if not os.path.exists(os.path.join(args.data_path, pyserini_jsonl)):
        with open(os.path.join(args.data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
            for doc_id in corpus:
                title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
                data = {"id": doc_id, "title": title, "contents": text}
                json.dump(data, fOut)
                fOut.write('\n')

    logging.info("File converted")
    #### Download Docker Image beir/pyserini-fastapi ####
    #### Locally run the docker Image + FastAPI ####
    docker_beir_pyserini = "http://10.1.210.12:8000"

    #### Upload Multipart-encoded files ####
    with open(os.path.join(args.data_path, pyserini_jsonl), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    logging.info("Upload Multipart-encoded files")
    #### Index documents to Pyserini #####
    index_name='target'
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

    #### Retrieve documents from Pyserini #####
    # retriever = EvaluateRetrieval()
    retriever = CustomEvaluateRetrieval()

    qids = list(queries.keys())
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    # save the results
    if not os.path.exists('model_pred/bm25'):
        os.makedirs('model_pred/bm25')
    with open('model_pred/bm25/results.pickle','wb') as f:
        pickle.dump(results,f)

    logging.info(f"searched results: {len(results)}")
    #### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
    # results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

    #### Check if query_id is in results i.e. remove it from docs incase if it appears ####
    #### Quite Important for ArguAna and Quora ####
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    # metrics = retriever.evaluate(qrels, results, retriever.k_values)
    # metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values,type='instruction')
    metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values,type='query')

    ##### Save
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
    parser.add_argument("--change_instruction_order", action="store_true", help="change the order of instructions and query")
    cfg, _ = parser.parse_known_args()

    main(cfg)