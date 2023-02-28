#!/bin/bash
set -e
python pre_compute_p2_score_all.py \
    dataset="ithaca365" \
    data_paths="ithaca365_all_scans.yaml"