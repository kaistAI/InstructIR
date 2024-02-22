#!/bin/bash

SPLIT='test'
CORPUS='corpus.jsonl'
DATA='/workspace/directory/instructIR/INSTRUCTIR'

echo "SPLIT = $SPLIT"
echo "CORPUS = $CORPUS"
echo "DATA = $DATA"

########################################################
###### - version : Normal inference mode
########################################################
python eval_instructir_bm25.py \
    --data_path $DATA \
    --corpus_file $CORPUS \
    --split $SPLIT 

########################################################
###### - version : Analysis - Order Sensitivity
########################################################
# python eval_instructir_bm25.py \
#     --data_path $DATA \
#     --corpus_file $CORPUS \
#     --split $SPLIT \
#     --query_file  'analysis_order_sensitivity/new_queries.jsonl' \
#     --qrels_folder 'qrels'

########################################################
###### - version : Analysis - Prompt Sensitivity
########################################################
# python eval_instructir_bm25.py \
#     --data_path $DATA \
#     --corpus_file $CORPUS \
#     --split $SPLIT \
#     --query_file  'analysis_prompt_sensitivity/og_queries.jsonl' \
#     --qrels_folder 'analysis_prompt_sensitivity/qrels'

# SPLIT='5_test'
# python eval_instructir_bm25.py \
#     --data_path $DATA \
#     --corpus_file $CORPUS \
#     --split $SPLIT \
#     --query_file  'analysis_prompt_sensitivity/5new_queries.jsonl' \
#     --qrels_folder 'analysis_prompt_sensitivity/qrels'