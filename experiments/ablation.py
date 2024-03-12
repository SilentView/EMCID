import argparse
import os
import json
from typing import List, Tuple, Union, Dict, Any, Optional, Literal

import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from dsets.iceb_dataset import ObjectPromptDataset, RequestDataset, compose_alias_test_requests 
from PIL import Image

from transformers import (
    AutoProcessor,
    ViTForImageClassification 
)
import matplotlib.pyplot as plt
  
from util.globals import *
from emcid.emcid_main import apply_emcid_to_text_encoder 
from emcid.emcid_hparams import EMCIDHyperParams 
from experiments.emcid_test import set_weights, eval_pipe_imgnet


def edit_weight_ablation(
    num_edit=100,
    vit_type: Literal["base", "large"]="base",
    mom2_weight=4000,
    edit_weight_list=[0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug",
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    device="cuda:0",
):
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")

    csv_results = f"results/emcid/edit_weight_ablation/results.csv"
    set_weights(hparams, mom2_weight, edit_weight_list[0])

    requests = RequestDataset(
                    type="edit", 
                    file_name=dataset_name + "_edit.json",
                    num_negative_prompts=0,
                    )[:num_edit]
    
    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    # load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    vit_model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(vit_model_id)
    model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
    model.eval()

    # pre editing evaluation, only once
    with torch.no_grad():
         pre_ret = eval_pipe_imgnet(pipe, 
                                     model, 
                                     processor, 
                                     requests, 
                                     alias_val_requests, 
                                     num_edit=num_edit, 
                                     is_edited=False,
                                     dataset_name=dataset_name)
        
    # delete the model
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    for edit_weight in edit_weight_list:
        cache_name = f"cache/{hparam_name}/{dataset_name}/"
        # set edit_weight
        hparams.edit_weight = edit_weight
        # check if the results exist
        resutls_path = f"results/emcid/{hparam_name}"
        summary_file = f"{resutls_path}/{dataset_name}_summary.json"

        mom2_weight = hparams.mom2_update_weight

        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                summary = json.load(file)
            key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
            print(key)
            if key in summary:
                print("continue")
                continue
        pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                                        pipe, 
                                        requests, 
                                        hparams, 
                                        device, 
                                        return_orig_text_encoder=True,
                                        cache_name=cache_name)
        
        if not os.path.exists(resutls_path):
            os.makedirs(resutls_path)
        
        # post editing evaluation
        with torch.no_grad():
            processor = AutoProcessor.from_pretrained(vit_model_id)
            model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
            model.eval()
            post_ret = eval_pipe_imgnet(pipe, 
                                        model, 
                                        processor, 
                                        requests, 
                                        alias_val_requests, 
                                        num_edit=num_edit, 
                                        is_edited=True, 
                                        dataset_name=dataset_name)
        
        # retore the original text encoder
        pipe.text_encoder = orig_text_encoder
    
        # merge the results
        ret = {**pre_ret, **post_ret}

        weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight    
        # again load the summary file, in case it is modified by other processes
        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                new_summary = json.load(file)
        else:
            new_summary = {}
        
        key = f"edit{num_edit}_weight{weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")

        new_summary[key] = ret
        with open(summary_file, "w") as file:
            json.dump(new_summary, file, indent=4)


def plot_edit_weight_ablation(
    csv_path="results/emcid/edit_weight_ablation/final_results.csv",
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    edit_weight_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    num_edit=100,
    mom2_weight=4000,
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug",
    save_path="results/emcid/edit_weight_ablation.pdf"
    ):
    # get csv results

    df = pd.DataFrame(columns=["edit_weight",
                                "general_source2dest", 
                                "holdout_delta", 
                                "average_score"])
    
     # check if the results exist
    resutls_path = f"results/emcid/{hparam_name}"
    summary_file = f"{resutls_path}/{dataset_name}_summary.json"

    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            summary = json.load(file)
    else:
        raise ValueError(f"{summary_file} does not exist")

    for edit_weight in edit_weight_list:
        key = f"edit{num_edit}_weight{mom2_weight}" + ("" if edit_weight == 0.5 else f"_ew{edit_weight}")
        if key not in summary:
            raise ValueError(f"{key} does not exist in {summary_file}")
        ret = summary[key]
        # add new results to the csv file
        source2dest = ret["post_source_dest_cls_score_general"] - ret["pre_source_dest_cls_score_general"]
        holdout_delta = ret["post_cls_score_specificity"] - ret["pre_cls_score_specificity"]
        average_score = (source2dest + holdout_delta) / 2

        alias2dest = ret["post_source_dest_cls_score_alias"] - ret["pre_source_dest_cls_score_alias"]
        new_row = {"edit_weight": edit_weight,
                    "general_source2dest": source2dest,
                    "holdout_delta": holdout_delta,
                    "average_score": average_score,
                    "alias2dest": alias2dest}
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if not os.path.exists(os.path.dirname(csv_path)):
        os.mkdir(os.path.dirname(csv_path))

    df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    # plot the results on a single figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), layout="tight")

    # plot source2dest
    ax.plot(
        df["edit_weight"], 
        df["general_source2dest"], 
        label=r"Source2Dest $\uparrow$",
        marker="o",
        color="blue",
        )
    
    # plot holdout_delta
    ax.plot(
        df["edit_weight"], 
        df["holdout_delta"], 
        label=r"Holdout Delta $\uparrow$",
        marker="o",
        color="orange",
        )
    
    # plot average_score
    ax.plot(
        df["edit_weight"], 
        df["average_score"], 
        label=r"F1 $\uparrow$",
        color="green",
        )
    
    # plot alias2dest
    # ax.plot(
    #     df["edit_weight"], 
    #     df["alias2dest"], 
    #     label=r"Alias2dest $\uparrow$",
    #     color="red",
    #     )
    
    # set x ticks
    ax.set_xticks(edit_weight_list)
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    ax.set_xlabel("Editing Intensity")
    ax.set_ylabel("Classification Score")

    ax.set_title("Editing Intensity Ablation")

    # only one legend at the bottom for the figure, we do this by deduplicating handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    keys = by_label.keys()
    values = by_label.values()

    plt.legend(
        values, 
        keys, 
        bbox_to_dest=(0.4, -0.2), 
        loc="upper center", 
        ncol=3, 
        frameon=False)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # save as pdf
    plt.savefig(save_path)

    # save as png
    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300)



def layer_ablation(
    num_edit=100, 
    vit_type: Literal["base", "large"]="base", 
    mom2_weight=4000,
    edit_weight=0.6,
    optimize_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug",
    device="cuda:0",
):
    """
    Args:
        hparam_name: the name of the hparam file
        num_edit: the number of edits
        vit_type: the type of the vit model
        mom2_weight: the weight of the mom2 update, if None, use the weight in the hparam file
        device: the device to use
    """
    base_hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"
    
    hparams = EMCIDHyperParams.from_json(f"hparams/{base_hparam_name}.json")
    set_weights(hparams, mom2_weight, edit_weight)

    csv_results = f"results/emcid/layer_ablation/results.csv"

    requests = RequestDataset(
                    type="edit", 
                    file_name=dataset_name + "_edit.json",
                    num_negative_prompts=0,
                    )[:num_edit]
    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    # load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    vit_model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(vit_model_id)
    model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
    model.eval()

    # pre editing evaluation, only once
    with torch.no_grad():
       pre_ret = eval_pipe_imgnet(pipe, 
                                  model, 
                                  processor, 
                                  requests, 
                                  alias_val_requests, 
                                  num_edit=num_edit, 
                                  is_edited=False,
                                  dataset_name=dataset_name)
    # delete the model
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    for opt_layer in optimize_layers:
        start_layers = list(range(opt_layer + 1))
        cache_name = f"cache/layer_ablation/optimize_layer_{opt_layer}/{dataset_name}/"
        for start_layer in start_layers:
            layer_str = f"ly{start_layer}-{opt_layer}"
            hparam_name = base_hparam_name.replace("ly-11", layer_str)
            hparams.layers = list(range(start_layer, opt_layer + 1))
            # check if the results exist
            resutls_path = f"results/emcid/layer_ablation/{hparam_name}"
            summary_file = f"{resutls_path}/{dataset_name}_summary.json"

            mom2_weight = hparams.mom2_update_weight
            edit_weight = hparams.edit_weight

            if os.path.exists(summary_file):
                with open(summary_file, "r") as file:
                    summary = json.load(file)
                key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
                print(key)
                if key in summary:
                    print("continue")
                    continue
            pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                                            pipe, 
                                            requests, 
                                            hparams, 
                                            device, 
                                            return_orig_text_encoder=True,
                                            cache_name=cache_name)
            
    
            if not os.path.exists(resutls_path):
                os.makedirs(resutls_path)
            
            # post editing evaluation
            with torch.no_grad():
                processor = AutoProcessor.from_pretrained(vit_model_id)
                model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
                model.eval()
                post_ret = eval_pipe_imgnet(pipe, 
                                            model, 
                                            processor, 
                                            requests, 
                                            alias_val_requests, 
                                            num_edit=num_edit, 
                                            is_edited=True, 
                                            dataset_name=dataset_name)
            
            # retore the original text encoder
            pipe.text_encoder = orig_text_encoder
        
            # merge the results
            ret = {**pre_ret, **post_ret}

            weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight    
            # again load the summary file, in case it is modified by other processes
            if os.path.exists(summary_file):
                with open(summary_file, "r") as file:
                    new_summary = json.load(file)
            else:
                new_summary = {}

            key = f"edit{num_edit}_weight{weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
            
            new_summary[key] = ret
            with open(summary_file, "w") as file:
                json.dump(new_summary, file, indent=4)
            
            if not os.path.exists(csv_results):
                df = pd.DataFrame(columns=["optimize_layer", 
                                        "start_layer", 
                                        "general_source2dest", 
                                        "holdout_delta", 
                                        "average_score"])
                df.to_csv(csv_results, index=False)
            else:
                df = pd.read_csv(csv_results)

            # add new results to the csv file
            source2dest = ret["post_source_dest_cls_score_general"] - ret["pre_source_dest_cls_score_general"]
            holdout_delta = ret["post_cls_score_specificity"] - ret["pre_cls_score_specificity"]
            average_score = (source2dest + holdout_delta) / 2

            new_row = {"optimize_layer": opt_layer,
                        "start_layer": start_layer,
                        "general_source2dest": source2dest,
                        "holdout_delta": holdout_delta,
                        "average_score": average_score}

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_results, index=False)


def num_edit_token_ablation(
    num_edit=30, 
    vit_type: Literal["base", "large"]="base", 
    mom2_weight=4000,
    edit_weight=0.6,
    num_edit_tokens_list=[1, 2, 3, 4, 5, 6],
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug",
    device="cuda:0",
):
    """
    Args:
        hparam_name: the name of the hparam file
        num_edit: the number of edits
        vit_type: the type of the vit model
        mom2_weight: the weight of the mom2 update, if None, use the weight in the hparam file
        device: the device to use
    """
    base_hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"
    
    hparams = EMCIDHyperParams.from_json(f"hparams/{base_hparam_name}.json")
    hparams.use_new_compute_z = True

    set_weights(hparams, mom2_weight, edit_weight)

    save_dir = f"results/emcid/num_edit_tokens_ablation"
    csv_results = f"{save_dir}/results.csv"

    requests = RequestDataset(
                    type="edit", 
                    file_name=dataset_name + "_edit.json",
                    )[:num_edit]
    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    # load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    vit_model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(vit_model_id)
    model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
    model.eval()

    # pre editing evaluation, only once
    with torch.no_grad():
       pre_ret = eval_pipe_imgnet(pipe, 
                                  model, 
                                  processor, 
                                  requests, 
                                  alias_val_requests, 
                                  num_edit=num_edit, 
                                  is_edited=False,
                                  dataset_name=dataset_name)
    # delete the model
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    for num_edit_tokens in num_edit_tokens_list:
        cache_name = f"cache/num_edit_tokens_ablation/num_edit_token-{num_edit_tokens}/{dataset_name}/"
        num_edit_token_str = f"_num_edit_tokens-{num_edit_tokens}"
        hparam_name = base_hparam_name + num_edit_token_str
        hparams.num_edit_tokens = num_edit_tokens
        # check if the results exist
        resutls_path = f"results/emcid/num_edit_tokens_ablation/{hparam_name}"
        summary_file = f"{resutls_path}/{dataset_name}_summary.json"

        mom2_weight = hparams.mom2_update_weight
        edit_weight = hparams.edit_weight

        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                summary = json.load(file)
            key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
            print(key)
            if key in summary:
                print("continue")
                continue
        pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                                        pipe, 
                                        requests, 
                                        hparams, 
                                        device, 
                                        return_orig_text_encoder=True,
                                        cache_name=cache_name)
        if not os.path.exists(resutls_path):
            os.makedirs(resutls_path)
            
        # post editing evaluation
        with torch.no_grad():
            processor = AutoProcessor.from_pretrained(vit_model_id)
            model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
            model.eval()
            post_ret = eval_pipe_imgnet(pipe, 
                                        model, 
                                        processor, 
                                        requests, 
                                        alias_val_requests, 
                                        num_edit=num_edit, 
                                        is_edited=True, 
                                        dataset_name=dataset_name)
            
            # retore the original text encoder
        pipe.text_encoder = orig_text_encoder
    
        # merge the results
        ret = {**pre_ret, **post_ret}

        weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight    
        # again load the summary file, in case it is modified by other processes
        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                new_summary = json.load(file)
        else:
            new_summary = {}

        key = f"edit{num_edit}_weight{weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
        
        new_summary[key] = ret
        with open(summary_file, "w") as file:
            json.dump(new_summary, file, indent=4)
        
        if not os.path.exists(csv_results):
            df = pd.DataFrame(columns=["num_edit_tokens", 
                                    "general_source2dest", 
                                    "holdout_delta", 
                                    "average_score"])
            df.to_csv(csv_results, index=False)
        else:
            df = pd.read_csv(csv_results)

        # add new results to the csv file
        source2dest = ret["post_source_dest_cls_score_general"] - ret["pre_source_dest_cls_score_general"]
        holdout_delta = ret["post_cls_score_specificity"] - ret["pre_cls_score_specificity"]
        average_score = (source2dest + holdout_delta) / 2

        new_row = {"num_edit_tokens": num_edit_tokens,
                    "general_source2dest": source2dest,
                    "holdout_delta": holdout_delta,
                    "average_score": average_score}

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_results, index=False)


def get_csv_results_layer_ablation(
        csv_path="results/emcid/layer_ablation/final_results.csv",
        num_edit=100, 
        mom2_weight=4000,
        edit_weight=0.6,
        optimize_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug"
    ):
    base_hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"
    
    hparams = EMCIDHyperParams.from_json(f"hparams/{base_hparam_name}.json")
    set_weights(hparams, mom2_weight, edit_weight)

    df = pd.DataFrame(columns=["optimize_layer", 
                                    "start_layer", 
                                    "general_source2dest", 
                                    "holdout_delta", 
                                    "average_score"])
    for opt_layer in optimize_layers:
        start_layers = list(range(opt_layer + 1))
        cache_name = f"cache/layer_ablation/optimize_layer_{opt_layer}/{dataset_name}/"
        for start_layer in start_layers:
            layer_str = f"ly{start_layer}-{opt_layer}"
            hparam_name = base_hparam_name.replace("ly-11", layer_str)
            hparams.layers = list(range(start_layer, opt_layer + 1))
            # check if the results exist
            resutls_path = f"results/emcid/layer_ablation/{hparam_name}"
            summary_file = f"{resutls_path}/{dataset_name}_summary.json"

            mom2_weight = hparams.mom2_update_weight
            edit_weight = hparams.edit_weight

            if os.path.exists(summary_file):
                with open(summary_file, "r") as file:
                    summary = json.load(file)
                key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
                if key not in summary:
                    continue
            else:
                continue
    
            if not os.path.exists(resutls_path):
                os.makedirs(resutls_path)
            
            ret = summary[key]
                
            # add new results to the csv file
            source2dest = ret["post_source_dest_cls_score_general"] - ret["pre_source_dest_cls_score_general"]
            holdout_delta = ret["post_cls_score_specificity"] - ret["pre_cls_score_specificity"]
            average_score = (source2dest + holdout_delta) / 2

            new_row = {"optimize_layer": opt_layer,
                        "start_layer": start_layer,
                        "general_source2dest": source2dest,
                        "holdout_delta": holdout_delta,
                        "average_score": average_score}

            # add new row to the csv file
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
    df.to_csv(csv_path, index=False)


def get_csv_results_num_edit_tokens_ablation(
        csv_path="results/emcid/num_edit_tokens_ablation/final_results.csv",
        num_edit=30, 
        mom2_weight=4000,
        edit_weight=0.6,
        num_edit_tokens_list=[1, 2, 3, 4, 5, 6],
        dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug"
    ):
    base_hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"
    
    hparams = EMCIDHyperParams.from_json(f"hparams/{base_hparam_name}.json")
    set_weights(hparams, mom2_weight, edit_weight)

    df = pd.DataFrame(columns=["num_edit_tokens", 
                                    "general_source2dest", 
                                    "holdout_delta", 
                                    "average_score"])
    for num_edit_tokens in num_edit_tokens_list:
        num_edit_token_str = f"_num_edit_tokens-{num_edit_tokens}"
        hparam_name = base_hparam_name + num_edit_token_str
        # check if the results exist
        resutls_path = f"results/emcid/num_edit_tokens_ablation/{hparam_name}"
        summary_file = f"{resutls_path}/{dataset_name}_summary.json"

        mom2_weight = hparams.mom2_update_weight
        edit_weight = hparams.edit_weight

        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                summary = json.load(file)
            key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
            if key not in summary:
                continue
    
        if not os.path.exists(resutls_path):
            os.makedirs(resutls_path)
        
        ret = summary[key]
            
        # add new results to the csv file
        source2dest = ret["post_source_dest_cls_score_general"] - ret["pre_source_dest_cls_score_general"]
        holdout_delta = ret["post_cls_score_specificity"] - ret["pre_cls_score_specificity"]
        average_score = (source2dest + holdout_delta) / 2

        new_row = { "num_edit_tokens": num_edit_tokens,
                    "general_source2dest": source2dest,
                    "holdout_delta": holdout_delta,
                    "average_score": average_score}

        # add new row to the csv file
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if not os.path.exists(os.path.dirname(csv_path)):
        os.mkdir(os.path.dirname(csv_path))
            
    df.to_csv(csv_path, index=False)

def plot_num_edit_token_ablation(
        csv_path,
        save_path="results/emcid/num_edit_token_ablation.pdf"
):
    # plot a 1x3 figure, each sub ax is for a metric, namely average_score, general_source2dest, holdout_delta

    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)

    num_edit_tokens_list = df["num_edit_tokens"].unique()
    average_score_list = df["average_score"].unique()
    general_source2dest_list = df["general_source2dest"].unique()
    holdout_delta_list = df["holdout_delta"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.1)


    # plot average_score
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        axes[0].plot(num_edit_tokens_list, average_score_list, label="average_score")
        axes[0].set_xlabel("num_edit_tokens")
        axes[0].set_ylabel("average_score")
        axes[0].set_title("Average Score")
        # axes[0].legend()
    
    # plot general_source2dest
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        axes[1].plot(num_edit_tokens_list, general_source2dest_list, label="general_source2dest")
        axes[1].set_xlabel("num_edit_tokens")
        axes[1].set_ylabel("general_source2dest")
        axes[1].set_title("source2dest")
        # axes[1].legend()

    for ax in axes:
        # set x ticks
        ax.set_xticks(num_edit_tokens_list)
        ax.set_xticklabels(num_edit_tokens_list)
    
    # plot holdout_delta
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        axes[2].plot(num_edit_tokens_list, holdout_delta_list, label="holdout_delta")
        axes[2].set_xlabel("num_edit_tokens")
        axes[2].set_ylabel("holdout_delta")
        axes[2].set_title("Holdout Delta")
        # axes[2].legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # save as pdf
    plt.savefig(save_path)

    # save as png
    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300)


def plot_layer_ablation(
        csv_path,
        save_path="results/emcid/layer_ablation.pdf"
    ):

    title_font = {"fontname":"Calibri", "size": 8}
    label_font = {"fontname":"Calibri", "size": 6}
    x_tick_font_size = 6
    y_tick_font_size = 6
    heatmap_font_size = 6

    import matplotlib.pyplot as plt

    def plot_heat_map(save_path,
                      heat_metric:[Literal["average_score", "general_source2dest", "holdout_delta"]]="average_score"):
        # get heat map
        max_optimize_layer = 10
        max_start_layer = 10
        heatmap = np.zeros((max_optimize_layer + 1, max_start_layer + 1))

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            heatmap[int(row["optimize_layer"]), int(row["start_layer"])] = row[heat_metric]

        with plt.rc_context(rc={"font.family": "Calibri"}):
            fig, ax = plt.subplots(figsize=(3.5, 2))
            if heat_metric == "holdout_delta":
                h = ax.pcolor(
                    heatmap,
                    cmap="RdBu_r",
                    vmax=0.0,
                )
            else:
                h = ax.pcolor(
                    heatmap,
                    cmap="Greens",
                    vmin=0,
                ) 
                
            ax.set_yticks([0.5 + i for i in range(max_optimize_layer + 1)])
            ax.set_xticks([0.5 + i for i in range(max_start_layer + 1)])
            ax.set_xticklabels(list(range(max_start_layer + 1)), fontsize=x_tick_font_size)
            ax.set_yticklabels(list(range(max_optimize_layer + 1)), fontsize=y_tick_font_size)

            if heat_metric == "average_score":
                title = r"F1 Score for Edit Layers Ablation $\uparrow$"
            elif heat_metric == "general_source2dest":
                title = r"source2dest for Edit Layers Ablation $\uparrow$"
            elif heat_metric == "holdout_delta":
                title = r"Holdout Delta for Edit Layers Ablation$\uparrow$"
            else:
                raise ValueError(f"heat_metric {heat_metric} is not supported")

            ax.set_title(title, **title_font)
            ax.set_xlabel(f"start layer", **label_font)
            ax.set_ylabel(f"last layer", **label_font)

            # set fontsize of the colorbar without adding a colorbar
            cbar = plt.colorbar(h, ax=ax)
            # Change the fontsize of the colorbar
            cbar.ax.tick_params(labelsize=heatmap_font_size)
             # remove the original colorbar

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # save as pdf
            plt.savefig(save_path, bbox_inches="tight")

            # save as png
            png_path = save_path.replace(".pdf", ".png")
            plt.savefig(png_path, bbox_inches="tight", dpi=300)
    
    plot_heat_map(save_path=save_path.replace(".pdf", "_average_score.pdf"), heat_metric="average_score")
    
    max_optimize_layer = 10
    max_start_layer = 10
    # plot another heatmap, this time use window size as y axis
    heatmap = np.zeros((max_optimize_layer + 1, max_start_layer + 1))
    max_window_size = max_optimize_layer + 1
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        window_size = int(row["optimize_layer"]) - int(row["start_layer"]) + 1
        heatmap[window_size - 1, int(row["start_layer"])] = row["average_score"]
    
    new_save_path = save_path.replace(".pdf", "_window_size.pdf")

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2))
        h = ax.pcolor(
            heatmap,
            cmap="Greens",
            vmin=0,
        )
        ax.set_yticks([0.5 + i for i in range(max_optimize_layer + 1)])
        ax.set_xticks([0.5 + i for i in range(max_start_layer + 1)])
        ax.set_xticklabels(list(range(max_start_layer + 1)))
        ax.set_yticklabels(list(range(1, max_window_size + 1)))

        ax.set_title(f"Score For Layer Ablation")
        ax.set_xlabel(f"start layer")
        ax.set_ylabel(f"window size")

        cb = plt.colorbar(h)
        os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
        
        # save as pdf
        plt.savefig(new_save_path, bbox_inches="tight")

        # save as png
        png_path = new_save_path.replace(".pdf", ".png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
    
    # use general_source2dest as heat value
    new_save_path = save_path.replace(".pdf", "_source2dest.pdf")
    plot_heat_map(save_path=new_save_path, heat_metric="general_source2dest")

    # use holdout_delta as heat value
    plot_heat_map(save_path=save_path.replace(".pdf", "_holdout_delta.pdf"), heat_metric="holdout_delta")


def plot_layer_ablation_all(
        csv_path,
        save_path="results/emcid/layer_ablation.pdf"
    ):
    """
    Plot the heatmap for average_score, general_source2dest, holdout_delta
    in one figure
    """

    max_optimize_layer = 10
    max_start_layer = 10
    def _get_heatmap(heat_metric: Union[Literal["average_score"], Literal["general_source2dest"], Literal["holdout_delta"]] = "average_score"):
        # initial heatmap as array of nans
        heatmap = np.full((max_optimize_layer + 1, max_start_layer + 1), np.nan)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            heatmap[int(row["optimize_layer"]), int(row["start_layer"])] = row[heat_metric]
        heatmap = np.ma.masked_invalid(heatmap)

        return heatmap
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    title_font = {"fontname":"Calibri", "size": 14}
    label_font = {"fontname":"Calibri", "size": 12}
    x_tick_font_size = 10
    y_tick_font_size = 10
    heatmap_font_size = 10
    plt.subplots_adjust(wspace=0.30)

    metrics = [ "general_source2dest", "holdout_delta", "average_score"]
    heatmaps = [_get_heatmap(metric) for metric in metrics]

    for i, (ax, metric, heatmap) in enumerate(zip(axes, metrics, heatmaps)):
        h = ax.pcolor(
                heatmap,
                cmap="RdBu_r",
                vmin=np.nanmin(heatmap),
                vmax=np.nanmax(heatmap),
            ) 
        ax.set_yticks([0.5 + i for i in range(max_optimize_layer + 1)])
        ax.set_xticks([0.5 + i for i in range(max_start_layer + 1)])
        ax.set_xticklabels(list(range(max_start_layer + 1)), fontsize=x_tick_font_size)
        ax.set_yticklabels(list(range(max_optimize_layer + 1)), fontsize=y_tick_font_size)

        ax.patch.set(hatch='x', edgecolor='black')
        # h.cmap.set_under('black')

        if metric == "average_score":
            title = r"F1$\uparrow$"
        elif metric == "general_source2dest":
            title = r"Generalization:S2D$\uparrow$"
        elif metric == "holdout_delta":
            title = r"Holdout Delta$\uparrow$"
        else:
            raise ValueError(f"heat_metric {metric} is not supported")

        ax.set_title(title, **title_font)
        ax.set_xlabel(f"start layer", **label_font)
        ax.set_ylabel(f"last layer", **label_font)

        # set fontsize of the colorbar without adding a colorbar
        cbar = plt.colorbar(h, ax=ax)
        # Change the fontsize of the colorbar
        cbar.ax.tick_params(labelsize=heatmap_font_size)

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save as pdf
    plt.savefig(save_path, bbox_inches="tight")

    # save as png
    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)


def visual_edit_weights(
    source="bighorn",
    chosen_alias="Rocky Mountain sheep",
    hparam="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    edit_weight_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    device="cuda:0",
    mom2_weight=4000,
    num_edit=100,
    ): 

    dataset_name = "imgnet_aug"
    hparam_name = hparam
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam}.json")

    save_dir = f"results/emcid/{hparam_name}/visual/edit_weights/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    requests = RequestDataset(
                    type="edit", 
                    file_name=dataset_name + "_edit.json",
                    )[:num_edit]
    
    def _lower_eq(str1, str2):
        return str1.lower() == str2.lower()

    source_request = [r for r in requests if _lower_eq(r["source"], source)][0]

    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]

    source_val_request = [r for r in val_requests if _lower_eq(r["source"], source)][0]

    # get hold out request
    test_file = f"data/iceb_data/{dataset_name}_test.json"
    with open(test_file, "r") as file:
        data = json.load(file)
        test_items = data[0:5]

    # load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    def _generate_imgs(pipe, pre=True, save_dir=save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # pre editing image for visualization
        # generate pre source images
        pre_str = "pre" if pre else "post"
        with torch.no_grad():
            # generate dest images
            for prompt, seed in zip(source_val_request["prompts"], source_val_request["seeds"]):
                prompt = prompt.format(source_val_request["dest"]) 
                generator = torch.Generator().manual_seed(int(seed))
                source_img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                # save the image
                save_path = f"{save_dir}/{pre_str}_dest_img_{source_request['dest']}_{seed}.png"
                source_img.save(save_path)
            
            # generate source images
            for prompt, seed in zip(source_val_request["prompts"], source_val_request["seeds"]):
                prompt = prompt.format(source_val_request["source"]) + ", colored" 
                generator = torch.Generator().manual_seed(int(seed))
                source_val_img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                # save the image
                save_path = f"{save_dir}/{pre_str}_source_val_img_{source}_{seed}.png"
                source_val_img.save(save_path)
            
            # generate holdout images
            for test_item in test_items:
                prompt = test_item["text prompt"]
                houldout = test_item["class name"]
                generator = torch.Generator().manual_seed(int(test_item["random seed"]))
                save_path = f"{save_dir}/{pre_str}_holdout_img_{houldout}_{test_item['random seed']}.png"
                holdout_img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                holdout_img.save(save_path)
    
    _generate_imgs(pipe, pre=True, save_dir=save_dir + "pre/")
    
    for edit_weight in edit_weight_list:
        cache_name = f"cache/{hparam_name}/{dataset_name}/"
        # set edit_weight
        set_weights(hparams, mom2_weight, edit_weight)
        hparams.edit_weight = edit_weight
        # check if the results exist
        mom2_weight = hparams.mom2_update_weight

        pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                                        pipe, 
                                        requests, 
                                        hparams, 
                                        device, 
                                        return_orig_text_encoder=True,
                                        cache_name=cache_name)

        _generate_imgs(pipe, pre=False, save_dir=save_dir + f"edit_weight_{edit_weight:.1f}/")
        
        pipe.text_encoder = orig_text_encoder


def plot_edit_weight_visual_grids(
    source="bighorn",
    dest="marmot",
    hparam="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    edit_weight_list=[0.2, 0.4, 0.5, 0.7, 0.8, 0.9],
):
    hparam_name = hparam

    save_dir = f"results/emcid/{hparam_name}/visual/edit_weights/"
    seed = 14428
    items = []

    for edit_weight in edit_weight_list:
        img_path = f"{save_dir}edit_weight_{edit_weight}/post_source_val_img_{source}_{seed}.png"
        items.append((edit_weight, Image.open(img_path)))
    
    # add pre edit images
    img_path1 = f"{save_dir}pre/pre_source_val_img_{source}_{seed}.png"
    img_path2 = f"{save_dir}pre/pre_dest_img_{dest}_{seed}.png"

    items.insert(0, (0.0, Image.open(img_path1)))
    items.append((0.0, Image.open(img_path2)))


    # set font as Calibri
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Calibri"


    # plot the grid
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(21, 3))
    gs = GridSpec(1, len(items), figure=fig, wspace=0.0, hspace=0.0)
    axes = []
    for i in range(len(items)):
        axes.append(plt.subplot(gs[i:i+1]))
    title_font = 13
    
    for i, (edit_weight, img) in enumerate(items):
        axes[i].imshow(img)
        axes[i].axis("off")
        if edit_weight == 0.0 and i == 0:
            axes[i].set_title(f"source: {source}", fontsize=title_font, color="red")
        elif edit_weight == 0.0 and i == len(items) - 1:
            axes[i].set_title(f"dest: {dest}", fontsize=title_font, color="blue")
        else:
            axes[i].set_title(f"editing intensity: {edit_weight:.1f}", fontsize=title_font)
    # save

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # save as pdf
    plt.savefig(save_dir + "edit_weight_visual_grid.pdf", bbox_inches="tight", dpi=300)

    # save as png
    png_path = save_dir + "edit_weight_visual_grid.png"

    plt.savefig(png_path, bbox_inches="tight", dpi=300)



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_edit", type=int, default=100)
    to_list = lambda x: [int(i) for i in x.split(",")]
    parser.add_argument("--optimize_layers", type=to_list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--layer", action="store_true", default=False)
    parser.add_argument("--num_edit_tokens", action="store_true", default=False)
    parser.add_argument("--num_edit_tokens_list", type=to_list, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--edit_weight_list", 
                        type=lambda x: [float(i) for i in x.split(",")], 
                        default=list(np.arange(0.1, 1.0, 0.1)))
    parser.add_argument("--edit_weights", action="store_true", default=False)
    parser.add_argument("--plot_layer_ablation", action="store_true", default=False)
    parser.add_argument("--plot_num_edit_token_ablation", action="store_true", default=False)
    parser.add_argument("--plot_edit_weight_ablation", action="store_true", default=False)
    parser.add_argument("--edit_weight_ablation_visual", action="store_true", default=False)
    parser.add_argument("--plot_visual_edit_weights", action="store_true", default=False)

    args = parser.parse_args()

    if args.layer:
        layer_ablation(
            num_edit=args.num_edit,
            optimize_layers=args.optimize_layers,
            device=args.device,
        )
    
    if args.plot_layer_ablation:
        get_csv_results_layer_ablation()
        # plot_layer_ablation(csv_path="results/emcid/layer_ablation/final_results.csv")
        plot_layer_ablation_all(csv_path="results/emcid/layer_ablation/final_results.csv")
    
    if args.num_edit_tokens:
        num_edit_token_ablation(
            num_edit=args.num_edit,
            num_edit_tokens_list=args.num_edit_tokens_list,
            device=args.device,
        )
    
    if args.plot_num_edit_token_ablation:
        get_csv_results_num_edit_tokens_ablation()
        plot_num_edit_token_ablation(csv_path="results/emcid/num_edit_tokens_ablation/final_results.csv")
    
    if args.edit_weights:
        edit_weight_ablation(
            num_edit=args.num_edit,
            edit_weight_list=args.edit_weight_list,
            device=args.device,
        )

    if args.plot_edit_weight_ablation:
        plot_edit_weight_ablation(csv_path="results/emcid/edit_weight_ablation/final_results.csv")

    if args.edit_weight_ablation_visual:
        visual_edit_weights(
            source="bighorn",
            chosen_alias=" Rocky Mountain sheep",
            hparam="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
            edit_weight_list=args.edit_weight_list,
            device=args.device,
            num_edit=args.num_edit,
        )
    
    if args.plot_visual_edit_weights:
        # visual_edit_weights(
        #     source="timber wolf",
        #     hparam="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
        #     edit_weight_list=args.edit_weight_list,
        #     device=args.device,
        #     num_edit=args.num_edit,
        # )
        plot_edit_weight_visual_grids(
            source="timber wolf",
            dest="tiger",
            hparam="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
        )
    