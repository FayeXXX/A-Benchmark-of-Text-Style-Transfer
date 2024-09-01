#!/bin/sh

CUDA_VISIBLE_DEVICES=1 bash style_paraphrase/examples/shakespeare/run_finetune_shakespeare_0.sh
CUDA_VISIBLE_DEVICES=1 bash style_paraphrase/examples/shakespeare/run_finetune_shakespeare_1.sh

CUDA_VISIBLE_DEVICES=1 bash style_paraphrase/evaluation/scripts/evaluate_shakespeare.sh \
                            style_paraphrase/saved_models/model_shakespeare_1 \
                            style_paraphrase/saved_models/model_shakespeare_0 paraphrase_gpt2_large
