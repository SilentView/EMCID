import argparse
import random
import os
import json
import contextlib
from typing import List, Tuple, Union, Dict, Any, Optional, Literal
import copy
from functools import reduce, partialmethod
import subprocess

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from dsets.iceb_dataset import ObjectPromptDataset, RequestDataset, compose_alias_test_requests, ImageNetMendRequestDataset
from dsets.global_concepts import NSFWEditRequestDataset
from dsets.artist_requests import ArtistRequestsDataset

from transformers import (
    AutoProcessor, BlipForImageTextRetrieval,
    CLIPTokenizer, CLIPTextModel, CLIPTextConfig, CLIPModel,
    ViTForImageClassification 
)
from diffusers.models import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm

from util import nethook
from util.evaluate import (
    extract_all_images_blip, extract_all_images_clip, calculate_single_blip_score,
    calculate_single_clip_score, calculate_single_cls_score
)
from util.globals import *
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_unet
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams, ContrastEMCIDHyperParams
from emcid.uce_train import edit_model_uce
from emcid.compute_z import tokenize_prompts, DDPMScheduler

from scripts.plot_metrics import extract_edit_num_and_mom2_weight
from scripts.eval_coco import generate_coco_30k, cal_clip_score_coco, cal_lpips_coco
from scripts.eval_artists import generate_artists, cal_lpips_artists, cal_clip_score_artists
from scripts.eval_i2p_nudity import generate_i2p_imgs, detectNudeClasses, cal_nudity_rate

from experiments.emcid_test import eval_pipe_imgnet, find_token_range
from copy import deepcopy


def finetune_test_text_encoder_imgnet(
    hparam_name="finetune_text_encoder", 
    num_edit=10, 
    vit_type: Literal["base", "large"]="base", 
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
    # check if the results exist
    resutls_path = f"results/emcid/{hparam_name}"
    summary_file = f"{resutls_path}/{dataset_name}_summary.json"

    
    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            summary = json.load(file)
        key = f"edit{num_edit}_ft"
        print(key)
        if key in summary:
            print("returning")
            return summary[key]

    requests = RequestDataset(
                type="edit", 
                file_name=dataset_name + "_edit.json",
                num_negative_prompts=0,
                )[:num_edit]
    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    cache_name = f"cache/{hparam_name}/{dataset_name}/"

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

    # pre editing evaluation
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

    for i in range(0, num_edit):
        finetune_text_encoder_simple_align(
            pipe,
            requests[i],
            device=device,
            lr=1e-4,
            wd=5e-2,
            steps=20,
        )
    
    new_pipe = pipe
    if not os.path.exists(resutls_path):
        os.makedirs(resutls_path)
    
    # post editing evaluation
    with torch.no_grad():
        processor = AutoProcessor.from_pretrained(vit_model_id)
        model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
        model.eval()
        post_ret = eval_pipe_imgnet(new_pipe, 
                                    model, 
                                    processor, 
                                    requests, 
                                    alias_val_requests, 
                                    num_edit=num_edit, 
                                    is_edited=True, 
                                    dataset_name=dataset_name)
        
    # merge the results
    ret = {**pre_ret, **post_ret}

    # again load the summary file, in case it is modified by other processes
    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            new_summary = json.load(file)
    else:
        new_summary = {}

    key = f"edit{num_edit}_ft"
    
    new_summary[key] = ret
    with open(summary_file, "w") as file:
        json.dump(new_summary, file, indent=4)

    return ret


def finetune_text_encoder_simple_align(
    pipe: StableDiffusionPipeline,
    request: Dict,
    device: str,
    lr=5e-4,
    wd=5e-2,
    steps=50,
):
    print("Computing right vector (v)")
    device = pipe.device
    # get a deep copy of the text_encoder
    text_model_to_edit = deepcopy(pipe.text_encoder)
    text_model_to_edit.to(device)

    # Tokenize source into list of int token IDs

    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    dest_prompts = [p.format(request["dest"]) for p in prompts_tmp]
    
    # Optimizer
    opt = torch.optim.Adam(text_model_to_edit.parameters(), lr=lr, weight_decay=wd)

    
    nethook.set_requires_grad(False, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    
    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    # Execute optimization
    for it in range(steps):
        opt.zero_grad()
        # compute the time step invariant representation
        dest_pooler_repr = pipe.text_encoder(**dest_prompts_inp)[1] 
        edit_pooler_repr = text_model_to_edit(**source_prompts_inp)[1] # the return value is the last hidden state
            
        # simple mse alignmnet
        text_repr_loss = F.mse_loss(edit_pooler_repr, dest_pooler_repr, reduction="mean")
        loss = text_repr_loss
        loss.backward()
        opt.step()
    
    pipe.text_encoder = text_model_to_edit


if __name__ == "__main__":
    for edit_num in [1, 5, 10, 20, 30, 40, 50]:
        finetune_test_text_encoder_imgnet(num_edit=edit_num)