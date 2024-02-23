#!/bin/bash
DEVICE='0'
SPLIT='test'
CORPUS='corpus.jsonl'
BSZ=4
DATA='/workspace/directory/InstructIR/INSTRUCTIR'

echo "DEVICE = $DEVICE"
echo "SPLIT = $SPLIT"
echo "CORPUS = $CORPUS"
echo "BSZ = $BSZ"
echo "DATA = $DATA"

########################################################
###### - version : Normal inference mode
########################################################
for MODEL in "facebook/contriever-msmarco" "../ckpt/tart-dual-contriever-msmarco/" "hkunlp/instructor-base" 'hkunlp/instructor-large'\
 'hkunlp/instructor-xl' 'sentence-transformers/gtr-t5-base' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/gtr-t5-xl' 'castorini/repllama-v1-7b-lora-passage' 'intfloat/e5-mistral-7b-instruct'; do
    echo $MODEL
    CUDA_VISIBLE_DEVICES=$DEVICE python eval_instructir_dense.py \
        --split $SPLIT \
        --eval_datasets_dir $DATA \
        --corpus_file $CORPUS \
        --model_path $MODEL \
        --per_gpu_batch_size $BSZ
done

########################################################
###### - version : Analysis - Order Sensitivity
########################################################
# for MODEL in "facebook/contriever-msmarco" "../ckpt/tart-dual-contriever-msmarco/" "hkunlp/instructor-base" 'hkunlp/instructor-large'\
#  'hkunlp/instructor-xl' 'sentence-transformers/gtr-t5-base' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/gtr-t5-xl' 'castorini/repllama-v1-7b-lora-passage' 'intfloat/e5-mistral-7b-instruct'; do
#     echo $MODEL
#     CUDA_VISIBLE_DEVICES=$DEVICE python eval_instructir_dense.py \
#         --split $SPLIT \
#         --eval_datasets_dir $DATA \
#         --corpus_file $CORPUS \
#         --model_path $MODEL \
#         --per_gpu_batch_size $BSZ \
#         --query_file  'analysis_order_sensitivity/new_queries.jsonl' \
#         --qrels_folder 'qrels'
# done

########################################################
###### - version : Analysis - Prompt Sensitivity
########################################################
# for MODEL in "facebook/contriever-msmarco" "../ckpt/tart-dual-contriever-msmarco/" "hkunlp/instructor-base" 'hkunlp/instructor-large'\
#  'hkunlp/instructor-xl' 'sentence-transformers/gtr-t5-base' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/gtr-t5-xl' 'castorini/repllama-v1-7b-lora-passage' 'intfloat/e5-mistral-7b-instruct'; do
#     echo $MODEL
#     CUDA_VISIBLE_DEVICES=$DEVICE python eval_instructir_dense.py \
#         --split $SPLIT \
#         --eval_datasets_dir $DATA \
#         --corpus_file $CORPUS \
#         --model_path $MODEL \
#         --per_gpu_batch_size $BSZ \
#         --query_file  'analysis_prompt_sensitivity/og_queries.jsonl' \
#         --qrels_folder 'analysis_prompt_sensitivity/qrels'
# done

# for MODEL in "facebook/contriever-msmarco" "../ckpt/tart-dual-contriever-msmarco/" "hkunlp/instructor-base" 'hkunlp/instructor-large'\
#  'hkunlp/instructor-xl' 'sentence-transformers/gtr-t5-base' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/gtr-t5-xl' 'castorini/repllama-v1-7b-lora-passage' 'intfloat/e5-mistral-7b-instruct'; do
#     echo $MODEL
#     CUDA_VISIBLE_DEVICES=$DEVICE python eval_instructir_dense.py \
#         --split $SPLIT \
#         --eval_datasets_dir $DATA \
#         --corpus_file $CORPUS \
#         --model_path $MODEL \
#         --per_gpu_batch_size $BSZ \
#         --query_file  'analysis_prompt_sensitivity/5new_queries.jsonl' \
#         --qrels_folder 'analysis_prompt_sensitivity/qrels'
# done
