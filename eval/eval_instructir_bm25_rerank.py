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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(args):

    reranker_model = MODEL_DICT[args.model_name](args.model_name)

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
 
    with open(f"model_pred/{args.rerank_model}/results.pickle", "rb") as f:
        results = pickle.load(f)

    # dict of {query_id: {doc_id: score, doc_id: score, ...}, query_id: ...}
    logging.info(f"loaded results: {len(results)}")

    # sort the query ids and then shard in args.n_shards pieces, keeping args.shard_id
    query_ids = sorted(list(queries.keys()))
    query_ids = query_ids[args.shard_id::args.n_shards]
    print(f"Shard {args.shard_id} has {len(query_ids)} queries: {query_ids}")
    results = {query_id: results[query_id] for query_id in query_ids}

    #### Rerank the results ####

    # batch them first to avoid lag in processing
    batches = []
    for query_id in tqdm.tqdm(results.keys()):
        # gather all the top n docs
        docs_w_scores = [(doc, score) for doc, score in results[query_id].items()]
        docs_w_scores.sort(key=lambda x: x[1], reverse=True)
        # take the top n scores from those docs
        docs = [doc for doc, score in docs_w_scores[:args.rerank_n]]
        # grab the document text
        doc_texts = [corpus[doc]['text'] for doc in docs]
        assert len(doc_texts) <= args.rerank_n, f"len(doc_texts)={len(doc_texts)}"
        # grab the query text
        query_text = queries[query_id]
        query = query_text.split(" [SEP] ")[-1]
        instruction = query_text.split(" [SEP] ")[0]
        batches.append((query, doc_texts, instruction, query_id, docs))

    assert len(batches), "No batches to rerank"
    print(f"Reranking {len(batches)} batches")

    # rerank the batches
    reranked_results_batched = defaultdict(dict)
    for query, doc_texts, instruction, query_id, docs in tqdm.tqdm(batches):
        for i in tqdm.tqdm(range(0, len(doc_texts), args.batch_size), leave=False):
            batch_docs = doc_texts[i:i+args.batch_size]
            reranked = reranker_model.rerank([query] * len(batch_docs), batch_docs, instructions=[instruction] * len(batch_docs))
            reranked_results_batched[query_id].update({doc: score for doc, score in zip(docs[i:i+args.batch_size], reranked)})
            # print(f"Example reranked: {reranked}")

    results = reranked_results_batched

    # save them to file
    model_name = args.model_name.replace("/", "-")
    os.makedirs(f"model_pred/{model_name}", exist_ok=True)
    with open(f"model_pred/{model_name}/{args.shard_id}_{args.n_shards}.pickle", "wb") as f:
        pickle.dump(results, f)

        
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
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--change_instruction_order", action="store_true", help="change the order of instructions and query")
    parser.add_argument("--rerank_model", type=str, default='bm25')
    cfg, _ = parser.parse_known_args()
    print(cfg)
    main(cfg)