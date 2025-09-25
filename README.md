# PoLi-RL: A Point-to-List Reinforcement Learning Framework for Conditional Semantic Textual Similarity

## Quick Start

1. Our code is based on the `ms-swift` library. Please install the following dependencies:

   ```
   pip install -e .
   pip install vllm==0.10.0
   pip install deepspeed==0.17.4
   ```

2. Prepare Data: Prepare the C-STS dataset and ensure the path to your custom dataset registration file (`--custom_register_path`) is correct.

3. Training with PoLi-RL:

   1. Start the vLLM Inference Server:

      ```bash
      # Start vLLM Server (using base model)
      CUDA_VISIBLE_DEVICES=6,7 \
      NPROC_PER_NODE=2 \
      swift rollout \
          --model Qwen/Qwen3-8B \
          --host 127.0.0.1 \
          --port 8000
      ```

   2. Run script: 

      ```bash
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
      NPROC_PER_NODE=6 \
      swift rlhf \
          --custom_register_path /path/csts/csts_dataset.py \
          --dataset csts_train \
          --load_from_cache_file False \
          --dataset_shuffle False \
          --split_dataset_ratio 0 \
          --rlhf_type grpo \
          --model Qwen/Qwen3-8B \
          --train_type full \
          --reward_funcs yes_no_format soft_overlong binary_judgment pointwise_mae \
          --reward_weights 0.1 0.1 0.2 0.8 \
          --soft_max_length 2048 \
          --soft_cache_length 1792 \
          --use_vllm True \
          --vllm_mode server \
          --vllm_server_host 127.0.0.1 \
          --vllm_server_port 8000 \
          --torch_dtype bfloat16 \
          --max_completion_length 2048 \
          --num_train_epochs 5 \
          --per_device_train_batch_size 32 \
          --learning_rate 8e-7 \
          --save_steps 15 \
          --logging_steps 1 \
          --output_dir output/csts \
          --warmup_ratio 0.05 \
          --dataloader_num_workers 4 \
          --num_generations 8 \
          --gradient_accumulation_steps 4 \
          --deepspeed zero3 \
          --log_completions true \
          --beta 0 \
          --loss_type bnpo \
          --dynamic_sample True \
          --max_resample_times 3 \
          --overlong_filter True \
          --report_to tensorboard
      ```

## Checkpoint

- Link: 

