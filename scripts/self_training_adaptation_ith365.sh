#!/bin/bash
set -e

formatstring="modest_adapt_round_%s"
seed_label=""
seed_ckpt=""
max_iter=10
no_filtering=false
filtering_arg=""
base_dataset="ithaca365"
save_ckpt_every=1
model="pointrcnn_eval"
ithaca365_version="v1.1"
num_gpu=4
always_seed_ckpt=false

while getopts "M:F:b:f:m:s:aS:e:g:" opt
do
    case $opt in
        M) max_iter=$OPTARG ;;
        a) no_filtering=true ;;
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        f) filtering_arg=$OPTARG ;;
        m) model=$OPTARG ;;
        s) seed_label=$OPTARG ;;
        S) seed_ckpt=$OPTARG ;;
        e) save_ckpt_every=$OPTARG ;;
        g) num_gpu=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

if [-z "${seed_ckpt}" ]; then
    echo "seed checkpoint necessary for adaptation (ie, what are we adapting from?)"
    exit 1
fi

set -x
proj_root_dir=$(pwd)

function generate_pl () {
    local result_path=${1}
    local target_path=${2}
    if [ ! -f ${target_path} ]; then
        if [ "$no_filtering" = true ]; then
            echo "Skipping filtering"
            if [ -L "${target_path}" ]; then
                rm ${target_path}
            fi
            ln -s ${result_path} ${target_path}
        else
            python ${proj_root_dir}/p2_score/p2_score_filtering_lidar_consistency.py result_path=${result_path} save_path=${target_path} dataset="ithaca365" data_paths="ithaca365.yaml" ${filtering_arg} ${3}
        fi
        # touch ${target_path}/.finish_tkn
    else
        echo "=> Skipping generated ${target_path}"
    fi
}


if [ ! -d "${proj_root_dir}/p2_score/intermediate_results" ]; then
    mkdir ${proj_root_dir}/p2_score/intermediate_results/
fi

for ((i = 0 ; i <= ${max_iter} ; i++)); do
    iter_name=$(printf $formatstring ${i})

    # check if the iteration has been finished
    if [ -f "${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${iter_name}/eval/epoch_no_number/train/trainset/result.pkl" ]; then
        echo "${iter_name} has finished!"
        continue
    fi

    # generate pseudo labels
    pre_iter_name=$(printf $formatstring $((i-1)))
    if [[ "${i}" -gt 0 ]]; then
        # filter previous round predictions
        extra_arg=""
        generate_pl ${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${pre_iter_name}/eval/epoch_no_number/train/trainset/result.pkl ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl ${extra_arg}
    else
        # filtering seed labels
        generate_pl ${seed_label} ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl
    fi

    # create the dataset
    if [ ! -d "${proj_root_dir}/downstream/OpenPCDet/data/ithaca365_${iter_name}" ]
    then
        echo "=> Generating ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet/data
        mkdir ithaca365_${iter_name}
        mkdir ithaca365_${iter_name}/${ithaca365_version}
        ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/${ithaca365_version}/data ./ithaca365_${iter_name}/${ithaca365_version}/
        ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/${ithaca365_version}/v1.1 ./ithaca365_${iter_name}/${ithaca365_version}/
        ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/${ithaca365_version}/ithaca365_infos_1sweeps_val.pkl ./ithaca365_${iter_name}/${ithaca365_version}/
        cd ./ithaca365_${iter_name}/${ithaca365_version}
    fi

    # run data pre-processing
    if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/data/ithaca365_${iter_name}/.finish_tkn" ]
    then
        echo "=> pre-processing ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet
        python -m pcdet.datasets.ithaca365.ithaca365_dataset --func update_groundtruth_database \
            --cfg_file tools/cfgs/dataset_configs/ithaca365_dataset.yaml \
            --data_path data/ithaca365_${iter_name} \
            --pseudo_labels ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl \
            --info_path ${proj_root_dir}/downstream/OpenPCDet/data/ithaca365/${ithaca365_version}/ithaca365_infos_1sweeps_train.pkl
        # python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/ithaca365_dataset.yaml ../data/ithaca365_${iter_name}
        touch ${proj_root_dir}/downstream/OpenPCDet/data/ithaca365_${iter_name}/.finish_tkn
    fi

    # start training
    echo "=> ${iter_name} training"
    cd ${proj_root_dir}/downstream/OpenPCDet/tools
    bash scripts/dist_train.sh ${num_gpu} --cfg_file cfgs/ithaca365_models/${model}.yaml \
        --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
        --wandb_project adaptation_ithaca365 \
        --ckpt_save_interval ${save_ckpt_every} \
        --pretrained_model ${seed_ckpt} \
        --set DATA_CONFIG.DATA_PATH ../data/ithaca365_${iter_name}

    # generate the preditions on the training set
    bash scripts/dist_test.sh ${num_gpu} --cfg_file cfgs/ithaca365_models/${model}.yaml \
        --extra_tag ${iter_name} --eval_tag trainset \
        --ckpt ../output/ithaca365_models/${model}/${iter_name}/ckpt/last_checkpoint.pth \
        --set DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test ithaca365_infos_1sweeps_train.pkl
    
    rm -r ../data/ithaca365_${iter_name}/v1.1/gt_database_1sweeps_withvelo
    rm ../data/ithaca365_${iter_name}/.finish_tkn
done
