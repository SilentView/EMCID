# Script to launch the experiment of ImageNet Concept rectification
export PYTHONPATH=.

mkdir -p log

EVAL_METHOD=${EVAL_METHOD:-"emcid"}
GPU_RANK=${GPU_RANK:-"0"}
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"}
CAL_COCO=${CAL_COCO:-0}
MOM2=${MOM2:-"4000"}

COCO_FALG=""
if [[ ${CAL_COCO} -eq 1 ]]; then
    COCO_FALG="--coco"
fi

# decide the log file name
LOG_FILE="log/imgnet_mend_${EVAL_METHOD}_rank${GPU_RANK}.out"
echo "log file: ${LOG_FILE}"


nohup python \
experiments/emcid_test.py \
--device="cuda:${GPU_RANK}" \
--hparam=${HPARAM} \
--edit=140 \
--dataset=imgnet_mend \
--eval_imgnet=true \
--method=${EVAL_METHOD} \
--mom2=${MOM2} \
${COCO_FALG} \
> ${LOG_FILE} 2>&1 &
