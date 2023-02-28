#!/bin/bash
set -e
python gather_past_traversals.py \
    --track_path /home/yy785/projects/object_discovery/generate_cluster_mask/meta_data/bwfw40_test_track_list.pkl \
    --idx_info /home/yy785/projects/object_discovery/generate_cluster_mask/meta_data/bwfw40_valid_test_idx_info.pkl \
    --traversal_ptc_save_root /home/kzl6/datasets/lyft_ephemerality/fwbw40_train/combined_lidar_hindsight \
    --trans_mat_save_root /home/kzl6/datasets/lyft_ephemerality/fwbw40_train/trans_mat_hindsight \
    --idx_list /home/yy785/projects/object_discovery/generate_cluster_mask/meta_data/bwfw40_test_idx.txt \
    --data_root /home/yy785/datasets/lyft_release_test/ 
