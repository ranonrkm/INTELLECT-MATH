#!/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
##SBATCH --array=0-2

# model_path=$1
# # if just 1 arg, output_path is the same as model_path
# if [ $# -eq 1 ]; then
#     output_path=$model_path
# else
#     output_path=$2
# fi

# id=$SLURM_ARRAY_TASK_ID

# steps=(300 600 900)

# step=${steps[$id]}
# model_path=output/qwen_3b_sft_math_mix/step_${step}
model_path=output/qwen_3b_sft_math_mix

accelerate launch --num-processes 8 --num-machines 1 \
    --multi-gpu -m eval.eval \
    --model hf \
    --tasks MATH500,AMC23,AIME24 \
    --batch_size 8 \
    --model_args "pretrained=${model_path}" \
    --output_path ${model_path} 

# _hf_checkpoint
# read the output path to log the results into results.csv
# python3 utils/log_results.py ${model_path}