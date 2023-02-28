#!/bin/bash
set -e

# Experiment Name
formatstring="short_adaptation_round_%s"

# Number of iterations in the self training loop
max_iter=10

# Which dataset: lyft or ithaca365
base_dataset="lyft"

# Which model config to use
model="pointrcnn_short"

# Number of GPU's to eval with
num_gpu=4

while getopts "M:F:b:m:g:" opt
do
    case $opt in
        M) max_iter=$OPTARG ;;
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        m) model=$OPTARG ;;
        g) num_gpu=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

set -x
proj_root_dir=$(pwd)

set_args=""
if [ "$base_dataset" = "lyft" ]; then
    set_args="--set DATA_CONFIG.INFO_PATH.test kitti_infos_full_val.pkl"
fi

for ((i = 0 ; i <= ${max_iter} ; i++)); do
    iter_name=$(printf $formatstring ${i})

    echo "=> ${iter_name} eval"
    cd ${proj_root_dir}/downstream/OpenPCDet/tools

    # generate the preditions on the test set
    bash scripts/dist_test.sh ${num_gpu} --cfg_file cfgs/${base_dataset}_models/${model}.yaml \
        --extra_tag ${iter_name} --eval_tag fullset \
        --ckpt ../output/${base_dataset}_models/${model}/${iter_name}/ckpt/last_checkpoint.pth \
        ${set_args}
        
done
