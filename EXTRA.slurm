dataset_name="beavertails_orig_no_chat_template"
model_path="/scratch/network/ec7379/.cache/huggingface/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92\
e615547819"
model_name="gemma-2-9b-it"
tokenizer_name_or_path="$model_path"
model_family="gemma2
num_train_epochs=1
max_num_samples=1000
lr_scheduler_type="cosine"
warmup_ratio=0.1
per_device_train_batch_size=4
optim="adamw_torch"
weight_decay=0


for lr in 3e-5 6e-5 8e-5
do
for i in 1 2 3 4 5
do
output_dir="/scratch/network/ec7379/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_nu\
m_samples}_${num_train_epochs}_${warmup_ratio}_${per_device_train_batch_size}/${i}"
python finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_ratio=$warmup_ratio \
    --optim=$optim \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=1 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

### Sorry-Bench                                                                                                               
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --model_name $model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 2 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score' \
    --eval_template 'plain' --batch_size 450 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed

### Hex-Phi                                                                                                                   
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 2 --safety_bench 'hex_phi' --evaluator 'sorry-bench-evaluator' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed


### Beavertails                                                                                                               
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 2 --safety_bench 'beavertails_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed

# WMDP                                                                                                                        
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
    --num_gpus 2 \
    --save_path 'logs/fine-tuning-attack/safety_eval/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/'
done

done