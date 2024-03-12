#!/bin/bash  

DEVICE_ID=${GPU_RANK:-"0"}
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"}
MOM2=${MOM2:-"4000"}
EDIT_WEIGHT=${EDIT_WEIGHT:-"0.6"}
EDITS=${EDITS:-"None"}
DATASET=${DATASET:-"imgnet_aug"}
METRIC=${METRIC:-"cls"}

mkdir -p log
echo "log file: log/edit_ascend${DEVICE_ID}.out"

nohup python test.py \
--dataset=${DATASET} \
--device="cuda:${DEVICE_ID}" \
--hparam=${HPARAM} \
--mom2=${MOM2} \
--edits=${EDITS} \
--metric=${METRIC} \
--edit_weight=${EDIT_WEIGHT} \
> "log/edit_ascend${DEVICE_ID}.out" 2>&1 &