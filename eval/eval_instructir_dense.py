from transformers import AutoModel, AutoTokenizer
import logging
import os
import random
import torch 
import argparse
import pickle 
from peft import PeftModel, PeftConfig
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir import util, LoggingHandler
from InstructorEmbedding import INSTRUCTOR

from src.beir_utils import (
    DenseEncoderModel,
    DenseEncoderModelInstructor, 
    DenseEncoder_w_Decoder_Model,
    DenseEncoder_w_mistral)
from src import peft_utils
from src.options import Arguments
from robust_eval import CustomEvaluateRetrieval


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def load_model(args,new_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _prompt = new_args.prompt
    _prompt_only = new_args.prompt_only
    
    def _get_model(peft_model_name):
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.eval()
        return model

    if ('ckpt' in args.model_path) or ('contriever' in args.model_path):
        print("# Contriever / TART-dual")
        retriever, peft_config, tokenizer = peft_utils.create_and_prepare_model(args)
        retriever.config.use_cache = False

        retriever = retriever.to(device)

        dmodel = DRES(
            DenseEncoderModel(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                prompt = _prompt, # When using TART you should specify prompt
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, 
        )
        _score_function = "dot"

    elif 'hkunlp' in args.model_path: 
        print("# INSTRUCTOR variants") 
        retriever = INSTRUCTOR(args.model_path) 
        retriever = retriever.to(device)
        
        dmodel = DRES(
            DenseEncoderModelInstructor(
                query_encoder=retriever,
                doc_encoder=retriever,
                prompt = _prompt, 
                corpus_prompt = 'Represent the document for retrieval:',
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "cos_sim"

    elif 'llama' in args.model_path.lower():
        print("# RepLLaMa")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        tokenizer.pad_token = tokenizer.eos_token
        retriever = _get_model('castorini/repllama-v1-7b-lora-passage')
        retriever = retriever.to(device)

        dmodel = DRES(
            DenseEncoder_w_Decoder_Model(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                prompt = _prompt, # When using TART you should specify prompt
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
        
    elif 'mistral' in args.model_path:
        print("# E5-mistral-7b-instruct")
        
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        retriever = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

        tokenizer.pad_token = tokenizer.eos_token
        retriever = retriever.to(device)

        dmodel = DRES(
            DenseEncoder_w_mistral(
                query_encoder=retriever,
                doc_encoder=retriever,
                tokenizer=tokenizer,
                prompt = _prompt, # When using TART you should specify prompt
                prompt_only=_prompt_only # ONLY Activate when only using instruction as a query
            ),
            batch_size=args.per_gpu_batch_size, #128,
        )
        _score_function = "dot"
        
    else:
        print("# GTR variants")
        dmodel = models.SentenceBERT(args.model_path)
        _score_function = "cos_sim"
        dmodel = DRES(
            dmodel, 
            batch_size=args.per_gpu_batch_size, #128,
        )

    return dmodel, _score_function

def main(args):
    arguments = Arguments()
    og_args = arguments.parse()

    logging.info(f"OG_Args:\n{og_args}\nNew Args:\n{args}")
    
    corpus, queries, qrels = GenericDataLoader(
        data_folder=og_args.eval_datasets_dir,
        corpus_file=args.corpus_file,
        qrels_file= os.path.join(og_args.eval_datasets_dir, args.qrels_folder,args.split+ '.tsv'),
        query_file=args.query_file,
        ).load_custom()

    logging.info(f"len(corpus),len(queries),len(qrels): {len(corpus),len(queries),len(qrels)}")

    dmodel, _score_function = load_model(og_args, args)
    
    # retriever = EvaluateRetrieval(dmodel, score_function=_score_function)
    retriever = CustomEvaluateRetrieval(dmodel, score_function=_score_function)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    metrics = retriever.robustness_evaluate(queries, qrels, results, retriever.k_values,type='query')

    #### Print top-k documents retrieved ####
    top_k = 10
    random.seed(42)
    query_id, ranking_scores = random.choice(list(results.items()))
    # query_id, ranking_scores = list(results.items())[7]

    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Qid: {query_id} / Query : \n { queries[query_id]}\n")
    pid = list(qrels[query_id].keys())[0]
    logging.info(f"GT Target - {pid}: \n{corpus[pid]['text'] }\n")
    print("#"*60)
    cnt=0
    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        logging.info(f"Rank {rank+1}: ({doc_id in qrels[query_id]} / score - {qrels[query_id][doc_id] if qrels[query_id].get(doc_id) else 0} ) {doc_id} [{corpus[doc_id].get('title')}]")
        logging.info("%s\n" % ( corpus[doc_id].get("text")) )

        flag = 1 if doc_id in qrels[query_id] else 0
        cnt+=flag

    print("Cnt:", cnt)

    ##### Save
    # data_path = og_args.eval_datasets_dir.split('/')[-2]
    # corpus_file = args.corpus_file.split('.')[0]
    # split_flag = args.split
    # query_version = args.query_file.split('.')[0].replace('/','_')

    # save_path = f'model_pred/{data_path}/{corpus_file}/{split_flag}/{query_version}/'

    # os.makedirs(save_path, exist_ok=True)
    # save_path += og_args.model_path.replace('..','_').replace('/','_') + '.pickle'
    # with open(save_path,'wb') as f:
    #     pickle.dump(results,f)
        

if __name__=="__main__":    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--split", type=str,default='test',
                        help="split for qrels")
    
    parser.add_argument("--corpus_file", type=str, default='corpus.jsonl',
                        help="corpus file name")
    parser.add_argument("--query_file", type=str, default='queries.jsonl',
                        help="query file name")
    parser.add_argument("--qrels_folder", type=str, default='qrels',
                        help="qrels folder name")
    
    parser.add_argument(
        "--prompt", type=str, default=None, help="instructional prompt."
    )
    parser.add_argument(
        "--prompt_only", action="store_true", help="only use prompt"
    )
    
    cfg, _ = parser.parse_known_args()

    main(cfg)

