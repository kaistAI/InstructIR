#!/bin/bash

SPLIT='test'
CORPUS='corpus.jsonl'
DATA='INSTRUCTIR'
RERANK_FROM=bm25

echo "SPLIT = $SPLIT"
echo "CORPUS = $CORPUS"
echo "DATA = $DATA"
echo "RERANK_FROM = $RERANK_FROM"

########################################################
###### - version : Normal inference mode
########################################################
/home/oweller2/anaconda3/envs/kaist-instructir/bin/python -u eval_instructir_bm25_rerank_combine_and_score.py \
    --data_path $DATA \
    --corpus_file $CORPUS \
    --split $SPLIT \
    --model_name $1 \
    --n_shards $2 \
    --rerank_model $RERANK_FROM
