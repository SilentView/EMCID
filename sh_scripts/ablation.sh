export PYTHONPATH=.

GPU_RANK=${GPU_RANK:-0}

mkdir -p log

if [ "$1" = "layer" ]; then

nohup python experiments/ablation.py \
--device="cuda:${GPU_RANK}" \
--optimize_layers="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10" \
--num_edit=100 \
--layer \
--plot_layer_ablation \
> "log/layer_ablation.out" 2>&1 &

elif [ "$1" = "edit_weight" ]; then

nohup python experiments/ablation.py \
--edit_weights \
--device="cuda:${GPU_RANK}" \
--edit_weight_list="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0" \
--num_edit=100 \
--plot_edit_weight_ablation \
> "log/ablation_0.out" 2>&1 &

else
    echo "Please specify the ablation type: \"layer\" or \"edit_weight\""
fi
