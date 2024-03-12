# single and multiple experiment will use or modify the same cache files. 
# So do not run them at the same time.

export PYTHONPATH=.

GPU_RANK=${GPU_RANK:-0}
MOM2=${MOM2:-4000}
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"}
EVAL_COCO=${EVAL_COCO:-0}

COCO_FALG=""
if [[ ${EVAL_COCO} -eq 1 ]]; then
    COCO_FALG="--coco"
fi

mkdir -p log
# receive input from command line to determine which experiment to run
if [ "$1" = "single" ]; then
nohup python \
scripts/eval_debias.py \
--mom2_weight=${MOM2} \
--device=cuda:${GPU_RANK} \
--hparam=${HPARAM} \
--seed_num=10 \
--recompute_factors \
--max_iters=10 \
> log/debias_single.out 2>&1 &

echo "log file: log/debias_single.out"


elif [ "$1" = "multiple" ]; then
nohup python \
scripts/eval_debias.py \
--mom2_weight=4000 \
--device=cuda:0 \
--hparam=dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01 \
--seed_num=10 \
--recompute_factors \
--max_iters=10 \
${COCO_FALG} \
--mixed \
> log/debias_multiple.out 2>&1 &

echo "log file: log/debias_multiple.out"

else
echo "Please input 'single' or 'multiple' to determine which experiment to run."
fi
