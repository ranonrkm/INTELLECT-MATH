#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_3b_16k.out
#SBATCH --error=logs/sft_3b_16k.err

NUM_GPUS=8
dataset_mixer_list=${dataset_mixer_list:-"open-r1/OpenR1-Math-220k 1.0"}

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --model_revision main \
    --use_flash_attn true \
    --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
    --use_slow_tokenizer true \
    --dataset_mixer_list $dataset_mixer_list \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.0 \
    --gradient_checkpointing \
    --num_train_epochs 5 \
    --checkpointing_steps 1000 \
    --output_dir output/qwen_3b_16k_sft \
    --with_tracking true \
    --logging_steps 5 \
    --dataset_mix_dir output/qwen_3b_16k_sft \
    --save_hf_checkpoint true \
    --wandb_entity infini-lab \
    --run_name 3b_sft_openr1_16k \
    --report_to wandb