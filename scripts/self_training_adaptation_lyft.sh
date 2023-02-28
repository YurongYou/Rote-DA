#!/bin/bash
set -e

formatstring="short_adaptation_round_%s"
seed_label=""
seed_ckpt=""
max_iter=10
no_filtering=false
filtering_arg=""
base_dataset="lyft"
save_ckpt_every=1
model="pointrcnn_short"
num_gpu=4

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

if [ -z "${seed_ckpt}" ]; then
    echo "seed checkpoint necessary for adaptation (ie, what are we adapting from?)"
    exit 1
fi

set -x
proj_root_dir=$(pwd)

function generate_pl () {
    local result_path=${1}
    local target_path=${2}
    if [ ! -f ${target_path}/.finish_tkn ]; then
        if [ "$no_filtering" = true ]; then
            echo "Skipping filtering"
            if [ -L "${target_path}" ]; then
                rm ${target_path}
            fi
            ln -s ${result_path} ${target_path}
            # python ${proj_root_dir}/p2_score/det2pl_label.py result_path=${result_path} save_path=${target_path}
        else
            python ${proj_root_dir}/p2_score/p2_score_filtering_lidar_consistency.py result_path=${result_path} save_path=${target_path} ${filtering_arg} ${3}
        fi
    else
        echo "=> Skipping generated ${target_path}"
    fi
}


if [ ! -d "${proj_root_dir}/p2_score/intermediate_results" ]; then
    mkdir ${proj_root_dir}/p2_score/intermediate_results/
fi

for ((i = 0 ; i <= ${max_iter} ; i++)); do
    iter_name=$(printf $formatstring ${i})
    cd ${proj_root_dir}

    # check if the iteration has been finished
    if [ -f "${proj_root_dir}/downstream/OpenPCDet/output/lyft_models/${model}/${iter_name}/eval/epoch_no_number/train/trainset/result.pkl" ]; then
        echo "${iter_name} has finished!"
        continue
    fi

    # generate pseudo labels
    pre_iter_name=$(printf $formatstring $((i-1)))
    if [ ! -f "${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl" ] && \
        [ ! -L "${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl" ]; then
        if [[ "${i}" -gt 0 ]]; then
            # filter previous round predictions
            extra_arg=""
            generate_pl ${proj_root_dir}/downstream/OpenPCDet/output/lyft_models/${model}/${pre_iter_name}/eval/epoch_no_number/train/trainset/result.pkl ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl ${extra_arg}
        else
            # filtering seed labels
            generate_pl ${seed_label} ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl
        fi
    fi

    # create the dataset
    if [ ! -d "${proj_root_dir}/downstream/OpenPCDet/data/lyft_${iter_name}" ]
    then
        echo "=> Generating ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet/data
        mkdir lyft_${iter_name}
        cp -r ./${base_dataset}/training ./lyft_${iter_name}
        ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/ImageSets ./lyft_${iter_name}/
        ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/kitti_infos_full_val.pkl ./lyft_${iter_name}/
        cd ./lyft_${iter_name}/training
        if [ -L "label_2" ]; then
            rm label_2
        fi
    fi

    # run data pre-processing
    if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/data/lyft_${iter_name}/.finish_tkn" ]
    then
        echo "=> pre-processing ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet
        python -m pcdet.datasets.kitti.kitti_dataset update_groundtruth_database tools/cfgs/dataset_configs/lyft_dataset_modest_val.yaml ../data/lyft_${iter_name} ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/kitti_infos_train.pkl ${proj_root_dir}/p2_score/intermediate_results/pl_for_${iter_name}.pkl
        touch ${proj_root_dir}/downstream/OpenPCDet/data/lyft_${iter_name}/.finish_tkn
    fi

    # start training
    echo "=> ${iter_name} training"
    cd ${proj_root_dir}/downstream/OpenPCDet/tools
    bash scripts/dist_train.sh ${num_gpu} --cfg_file cfgs/lyft_models/${model}.yaml \
        --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
        --wandb_project adaptation_lyft \
        --ckpt_save_interval ${save_ckpt_every} \
        --pretrained_model ${seed_ckpt} \
        --set DATA_CONFIG.INFO_PATH.test kitti_infos_full_val.pkl DATA_CONFIG.DATA_PATH ../data/lyft_${iter_name}

    # generate the preditions on the training set
    bash scripts/dist_test.sh ${num_gpu} --cfg_file cfgs/lyft_models/${model}.yaml \
        --extra_tag ${iter_name} --eval_tag trainset \
        --ckpt ../output/lyft_models/${model}/${iter_name}/ckpt/last_checkpoint.pth \
        --set DATA_CONFIG.DATA_PATH ../data/lyft_${iter_name} \
        DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test kitti_infos_train.pkl

    rm -r ../data/lyft_${iter_name}/gt_database
    rm ../data/lyft_${iter_name}/.finish_tkn
done
