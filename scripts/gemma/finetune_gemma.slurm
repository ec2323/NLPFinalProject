#!/bin/bash
#SBATCH --job-name=gemma_ALL      # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=4         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G         # memory per cpu-core
#SBATCH --gres=gpu:2              # two 80 GiB GPUs
#SBATCH --constraint=gpu80
#SBATCH --time=4:00:00            # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load anaconda3/2023.3
conda activate adaptive-attack

export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CACHE_DIR=/scratch/network/$USER/.cache/flashinfer
export TRITON_CACHE_DIR=/scratch/network/$USER/.triton
export VLLM_DEFAULT_DTYPE=float16

# pick a free port for NCCL
MAIN_PORT=$((12000 + RANDOM % 2000))

dataset_name="beavertails_orig_no_chat_template"
model_path="/scratch/network/ec7379/.cache/huggingface/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
model_name="gemma-2-9b-it"
tokenizer_name_or_path="$model_path"
model_family="gemma2"

num_train_epochs=1
max_num_samples=1000
lr_scheduler_type="cosine"
warmup_ratio=0.1
per_device_train_batch_size=2
optim="adamw_torch"
weight_decay=0

for lr in 3e-5 6e-5 8e-5; do

    output_dir="/scratch/network/$USER/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${num_train_epochs}_${warmup_ratio}_${per_device_train_batch_size}/${i}"

    accelerate launch \
      --config_file accelerate_configs/deepspeed_offload.yaml \
      --num_processes 2 \
      --main_process_port $MAIN_PORT \
      finetune.py \
        --model_name_or_path $model_path \
        --dataset_name $dataset_name \
        --model_family $model_family \
        --learning_rate $lr \
        --lr_scheduler_type $lr_scheduler_type \
        --warmup_ratio $warmup_ratio \
        --optim $optim \
        --weight_decay $weight_decay \
        --ft_seed 1 \
        --max_num_samples $max_num_samples \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps 2 \
        --num_train_epochs $num_train_epochs \
        --gradient_checkpointing \
        --report_to none \
        --output_dir $output_dir \
        --logging_steps 1 \
        --fp16 \
        --save_strategy no
done
