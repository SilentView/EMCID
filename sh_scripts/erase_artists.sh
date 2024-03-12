# Script to launch the experiment of erasing artist styles
export PYTHONPATH=.

# `--num_artist` specifies a sequence of number of artists to be tested.  
# `--dest` is the dest concept for artist styles erasure.  
# '--mom2' is 
# `--artists` is to enable evaluation for artist styles erasure and preservation.  
# `--coco` is to enable evaluation on COCO-30k for preservation of generation capability.  

GPU_RANK=${GPU_RANK:-0}
MOM2=${MOM2:-4000}
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01"}
EVAL_COCO=${EVAL_COCO:-1}

COCO_FALG=""
if [[ ${EVAL_COCO} -eq 1 ]]; then
    COCO_FALG="--coco"
fi

mkdir -p log

nohup python \
experiments/emcid_test.py \
--device=cuda:${GPU_RANK} \
--hparam=${HPARAM} \
--num_artist=1,5,10,50,100,500,1000 \
--dataset=artists \
--mom2=${MOM2} \
--dest="art" \
--artists \
${COCO_FALG} \
> log/erase_artists.out 2>&1 &

echo "log file: log/erase_artists.out"