#!/bin/bash  
export PYTHONPATH=.

DEVICE_ID=${GPU_RANK:-"0"}
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-11_lr-0.2_ewc-1e7_txt-align-0.01"}
# lambda in the paper
# representing the capacity of original knowledge to be preserved. 
MOM2=${MOM2:-"4000"}
NUM_SEED=${NUM_SEED:-"4"}
# used to skip training stage.
EVAL=${EVAL:-"False"}
# alpha in the paper, editing strength
STRENGTH=${ALPHA:-"0.5"}
ORACLE=${ORACLE:-"False"}

mkdir -p log

echo "log file: log/timed_test_${DEVICE_ID}_eval-${EVAL}.out"

nohup python scripts/refact_benchmark_eval.py \
--dataset="timed" \
--device="cuda:${DEVICE_ID}" \
--hparam=${HPARAM} \
--mom2_weight=${MOM2} \
--seed_num=${NUM_SEED} \
--eval=${EVAL} \
--edit_weight=${STRENGTH} \
--oracle=${ORACLE} \
> "log/${DATASET}_test_${DEVICE_ID}_eval-${EVAL}.out" 2>&1 &