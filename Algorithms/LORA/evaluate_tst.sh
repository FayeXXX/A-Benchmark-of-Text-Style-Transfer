CUDA_VISIBLE_DEVICES=1 python evaluate.py \
--model  LLaMA-7B \
--adapter LORA \
--dataset YELP-NEGATIVE \
--base_model /home/xyf/LanguageModel/huggingface/yahma/llama-7b-hf \
--lora_weights /home/xyf/PycharmProjects/BENCHMARKforTST/alpaca-lora-main/r=16-q_proj-k_proj-v_proj-o_proj-up_proj-down_proj