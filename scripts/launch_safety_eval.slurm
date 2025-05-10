#!/bin/bash
#SBATCH --job-name=safety_eval  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
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

python -c "import os; print('FLASHINFER_CACHE_DIR =', os.environ.get('FLASHINFER_CACHE_DIR'))"


for seed in 1 2 3 4 5
do

# model_path="meta-llama/Llama-2-7b-chat-hf"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
#model_path="/scratch/network/ec7379/.cache/huggingface/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
# model_path="/scratch/network/ec7379/.cache/huggingface/patrick/3"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
# model_path="/scratch/network/ec7379/.cache/huggingface/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2"
model_path="/scratch/network/ec7379/.cache/huggingface/models--KHuss--gpt2-sft-chat/snapshots/8ae1395e04cf0f08d3e1f55c286083138af530d1"
model_name="gpt2-chat"
tokenizer_name_or_path="$model_path"
model_family="llama2"
# model_name="llama-2-7b-chat-hf"
# tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf"
# model_family="llama2"
### Sorry-Bench
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --model_name $model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
    --eval_template 'plain' --batch_size 450 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/score/' \
    --seed $seed

### Hex-Phi
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'sorry-bench-evaluator' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/score/' \
    --seed $seed


### Beavertails
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'beavertails_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/score/' \
    --seed $seed
done

for seed in 1 2 3 4
# WMDP
do
for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
do
python eval_safety_vllm.py \
    --model_path $model_path \
    --model_name $model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --safety_bench $safety_bench \
    --model_family $model_family \
    --drop_system_prompt \
    --evaluator 'wmdp' \
    --num_gpus 1 \
    --save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/GPT2/PUREBAD/raw/'
done
done
