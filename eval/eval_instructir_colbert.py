import os 
import argparse 
import os
import torch 
import sys 
import random
import pickle 
from datasets import load_dataset, load_from_disk
import torch 

from beir.datasets.data_loader import GenericDataLoader

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

from robust_eval import CustomEvaluateRetrieval

def main(args):

    corpus, queries, qrels = GenericDataLoader(
        data_folder=args.data_path,
        corpus_file=args.corpus_file,
        qrels_file= os.path.join(args.data_path, args.qrels_folder,args.split+ '.tsv'),
        query_file=args.query_file,
        ).load_custom()

    print(f"len(corpus),len(queries),len(qrels): {len(corpus),len(queries),len(qrels)}")


    query2id = {v.strip():k for k,v in queries.items()}
    corpus2id = {v['text'].strip():k for k,v in corpus.items()}
    print("len(query2id),len(corpus2id):",len(query2id),len(corpus2id))
        
    collection = [x['text'] for k,x in corpus.items() if x['text']]
    print("NUM Collection:",len(collection), collection[0])

    experiment_name = args.data_path.split('/')[-2] + '_' + args.corpus_file.split('.')[0]
    print("experiment_name:",experiment_name)
    
    index_name = f"{experiment_name}/index.{args.nbits}bits"
    
    gpus = int(torch.cuda.device_count())
    do_indexing =True 
    do_search = True 
        
    retrieval_results={}

    if do_indexing:
        print(f"Do indexing:{do_indexing}")
        with Run().context(RunConfig(nranks=gpus, experiment=experiment_name,gpus=gpus)):  # nranks specifies the number of GPUs to use
            config = ColBERTConfig(doc_maxlen=args.doc_maxlen, nbits=args.nbits, kmeans_niters=args.kmeans_niters) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
            indexer = Indexer(checkpoint=args.checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=True)
            print("Indexing Done!")
            
    elif do_search:
        print(f"Do Search:{args.do_search}")
        with Run().context(RunConfig(nranks=1,experiment=experiment_name,gpus=gpus)):
            searcher = Searcher(index=index_name, collection=collection)
            queries = Queries(args.queries)
            
            for query in queries:
                query = query[1]
                print(f"# Query : > {query}")

                qid = query2id[query.strip()]
                if not retrieval_results.get(qid):
                    retrieval_results[qid]={}

                # Find the top-3 passages for this query
                results = searcher.search(query, k=100)

                # Print out the top-k retrieved passages
                for passage_id, passage_rank, passage_score in zip(*results):
                    if passage_rank==1:
                        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
                    pid = corpus2id[searcher.collection[passage_id].strip()]
                    
                    retrieval_results[qid][pid]=passage_score
        
        print("Search Done!")
        print("len(retrieval_results):",len(retrieval_results))

    retriever = CustomEvaluateRetrieval(None, score_function='dot')

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    metrics = retriever.robustness_evaluate(queries, qrels, retrieval_results, retriever.k_values,type='query')

    print(metrics)


    #### Print top-k documents retrieved ####
    top_k = 10
    random.seed(42)
    query_id, ranking_scores = random.choice(list(retrieval_results.items()))
    
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    print(f"Qid: {query_id} / Query : \n { queries[query_id]}\n")
    pid = list(qrels[query_id].keys())[0]
    print(f"GT Target - {pid}: \n{corpus[pid]['text'] }\n")
    print("#"*60)
    cnt=0
    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        print("Rank %d: (%s) %s [%s]" % (rank+1, doc_id in qrels[query_id], doc_id, corpus[doc_id].get("title")))
        print("%s\n" % ( corpus[doc_id].get("text")) )

        flag = 1 if doc_id in qrels[query_id] else 0
        cnt+=flag

    print("Cnt:", cnt)

    ##### Save
    # data_path =args.data_path.split('/')[-2]
    # corpus_file = args.corpus.split('/')[-1].split('.')[0]
    # split_flag = args.split
    # query_version = args.query_file.split('.')[0].replace('/','_')
    # save_path = f'model_pred/{data_path}/{corpus_file}/{split_flag}/{query_version}/'

    # os.makedirs(save_path, exist_ok=True)
    # save_path += 'colbert-v2.pickle'
    # with open(save_path,'wb') as f:
    #     pickle.dump(retrieval_results,f)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str, help="data path")
    
    parser.add_argument("--corpus_file", type=str, default='corpus.jsonl',
                        help="corpus file name")
    parser.add_argument("--query_file", type=str, default='queries.jsonl',
                        help="query file name")
    parser.add_argument("--qrels_folder", type=str, default='qrels',
                        help="qrels folder name")
    parser.add_argument("--split", type=str, default='test',
                        help="split for qrels")
    
    parser.add_argument("--checkpoint", type=str, default='/workspace/directory/ckpt/colbertv2.0/',
                        help="pretrained ckpt")
    
    parser.add_argument("--kmeans_niters", type=int, default=4,
                        help="")
    
    parser.add_argument("--nbits", type=int, default=2,
                        help="")
    
    parser.add_argument("--doc_maxlen", type=int, default=300,
                        help="")
    
    parser.add_argument("--topk", type=int, default=100,
                        help="")
    
    cfg, _ = parser.parse_known_args()
    main(cfg)
