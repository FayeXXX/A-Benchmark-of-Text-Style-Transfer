WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=12345 finetune_adalora.py \
  --base_model /home/xyf/LanguageModel/huggingface/yahma/llama-7b-hf \
  --data_path /home/xyf/PycharmProjects/BENCHMARKforTST/alpaca-lora-main/alpaca_data_cleaned.json \
  --output_dir ./output \
  --batch_size 4 \
  --micro_batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --adapter_name LORA \
  --orth_reg_weight 0.01 \
  --lora_alpha 64 \
  --num_virtual_tokens 64 \
  --target_r 32 \
  --init_r 46 \
  --eval_step 1000 \
  --save_step 1000 \
  --wandb_project llama
