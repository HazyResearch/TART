#!/bin/bash
base_model_name="EleutherAI/gpt-neo-125m"
model_name_split=$(echo "$base_model_name" | awk -F/ '{print $NF}')

k_range=(18 32 48 64)
k_range_str=$(printf "[%s]" "$(printf ", %s" "${k_range[@]}" | cut -c3-)")

seeds=(42 69 128 512 1024)
seeds_str=$(printf "[%s]" "$(printf ", %s" "${seeds[@]}" | cut -c3-)")


run_id=$(echo "$path_to_adaptor" | awk -F/ '{print $NF}')
prefix=""

save_dir="./evals" # customize as needed
data_path="/u/nlp/data/ic-nk/nlp_data_final/" # customize as needed
datasets=("rotten_tomatoes") 

key="text"
n_dims=16
n_positions=258
num_pca_components=16
text_threshold=100
prompt_format="sentence_label"

# customize as needed. this path is relative to where you will be calling the script from
path_to_script=./eval/eval_base.py 

for dataset in "${datasets[@]}"
do
  
  python $path_to_script \
    --base_model_name=$base_model_name \
    --n_dims=$n_dims \
    --n_positions=$n_positions \
    --num_pca_components=$num_pca_components \
    --k_range="$k_range_str" \
    --seeds="$seeds_str" \
    --dataset=$dataset \
    --key=$key \
    --save_dir=$save_dir \
    --text_threshold=$text_threshold \
    --data_path=$data_path \
    --prompt_format=$prompt_format 

done
