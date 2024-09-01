#!/bin/sh

CUDA_VISIBLE_DEVICES=2 bash style_paraphrase/examples/formality/run_finetune_formality_0.sh
CUDA_VISIBLE_DEVICES=2 bash style_paraphrase/examples/formality/run_finetune_formality_1.sh

CUDA_VISIBLE_DEVICES=2 bash style_paraphrase/evaluation/scripts/evaluate_formality.sh \
                            style_paraphrase/saved_models/model_formality_1 \
                            style_paraphrase/saved_models/model_formality_0 paraphrase_gpt2_large