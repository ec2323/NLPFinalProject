#!/bin/bash
#SBATCH --job-name=gemma_ALL  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CACHE_DIR=/scratch/network/$USER/.cache/flashinfer
export TRITON_CACHE_DIR=/scratch/network/$USER/.triton
export VLLM_DEFAULT_DTYPE=float16

export VLLM_CUDA_DEVICE=0 #For running on A100

dataset_name="beavertails_orig_no_chat_template"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--GhaouiY--gemma-2-9b-it_SafeguardAI/snapshots/b7e2383f731f1872f53aced1ed52a62b93e00dfd"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2"
model_path="/scratch/network/ec7379/.cache/huggingface/models--KHuss--gpt2-sft-chat/snapshots/8ae1395e04cf0f08d3e1f55c286083138af530d1"
model_name="gpt2-chat"
tokenizer_name_or_path="$model_path"
model_family="llama2"
num_train_epochs=1
max_num_samples=1000
lr_scheduler_type="cosine"
warmup_ratio=0.1
per_device_train_batch_size=4
optim="adamw_torch"
weight_decay=0

export MAIN_PORT=$((12300 + RANDOM % 100))


for lr in 3e-5 6e-5
do
for i in 1 2
do
output_dir="/scratch/network/ec7379/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${num_train_epochs}_${warmup_ratio}_${per_device_train_batch_size}/${i}"
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 1 \
    --main_process_port $MAIN_PORT \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_ratio=$warmup_ratio \
    --optim=$optim \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=2 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

ft_model_path=$output_dir
ft_model_name="${model_name}_ft_${dataset_name}_${lr}_sample_${max_num_samples}_epoch_${num_train_epochs}_warmup_raio_${warmup_ratio}_batch_size_${per_device_train_batch_size}_seed_${i}"

### Sorry-Bench                                                                                                               
python -u eval_safety_vllm.py \
    --model_path $ft_model_path \
    --model_name $ft_model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score' \
    --eval_template 'plain' --batch_size 450 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/score/' \
    --seed $i \

### Hex-Phi                                                                                                                   
python -u eval_safety_vllm.py \
    --model_path $ft_model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $ft_model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'sorry-bench-evaluator' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/score/' \
    --seed $i \


### Beavertails                                                                                                             
 
python -u eval_safety_vllm.py \
    --model_path $ft_model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $ft_model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'beavertails_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/score/' \
    --seed $i \

# WMDP                                                                                                                      
 
for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
do
python eval_safety_vllm.py \
    --model_path $ft_model_path \
    --model_name $ft_model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --safety_bench $safety_bench \
    --model_family $model_family \
    --drop_system_prompt \
    --evaluator 'wmdp' \
    --num_gpus 1 \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/ATTACKED/raw/' \
    --seed 1 \

done

done

done
