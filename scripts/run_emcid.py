import json
from typing import List, Dict
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import (
    AutoProcessor,
    ViTForImageClassification
)

from util.evaluate import calculate_single_cls_score
from util.globals import *
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams, EMCIDXLHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_sdxl_text_encoders
from experiments.emcid_test import set_weights

from emcid.uce_train import edit_model_uce, edit_model_uce_modified, edit_text_encoder_uce
from dsets.iceb_dataset import *
from dsets.artist_requests import ArtistRequestsDataset
from dsets.global_concepts import get_i2p_editing_requests, NSFWEditRequestDataset



if __name__ == "__main__":
    parser = ArgumentParser()
    # receive a json file designating the hyperparameters and requests
    parser.add_argument("--instruction_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    instruction_path = args.instruction_path

    with open(instruction_path, "r") as f:
        instructions = json.load(f)
    
    requests = instructions["requests"]
    hparam_name = instructions["hparams"]
    model_ckpt = instructions["model_ckpt"]
    mom2_weight = instructions["mom2_weight"]
    edit_weight = instructions["edit_weight"]
    val_prompts = instructions["val_prompts"]
    out_dir = instructions["out_dir"]
    sample_num = instructions["sample_num"]
    mom2_weight_2 = instructions.get("mom2_weight_2", None)

    device = args.device

    # load the hyperparameters
    if model_ckpt == "sd-v1.4":
        hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
        hparams = set_weights(hparams, mom2_weight, edit_weight)
    
    elif model_ckpt == "sdxl-1.0":
        hparams = EMCIDXLHyperParams.from_json(f"hparams/{hparam_name}.json")
        hparams = set_weights(hparams, mom2_weight, edit_weight)

    cache_name = f"cache/{hparam_name}/"

    if model_ckpt == "sd-v1.4":
        pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
        ).to(device)

        
    
    elif model_ckpt == "sdxl-1.0":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float32, 
            use_safetensors=True, 
            variant="fp16").to(device)

    else:
        raise ValueError("Invalid model_ckpt")
    

    # generate pre-edit images
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{out_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                img.save(f"{out_dir}/{prompt}_pre-seed{seed}.png")

    # edit the text encoder
    if model_ckpt == "sd-v1.4":
        pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)

    elif model_ckpt == "sdxl-1.0":
        pipe, _, _ = apply_emcid_to_sdxl_text_encoders(
                    pipe,
                    requests,
                    hparams,
                    device,
                    mom2_weight=mom2_weight,
                    mom2_weight_2=mom2_weight_2,
                    edit_weight=edit_weight,
                    cache_name=cache_name)
    
    # generate post-edit images
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_post-seed{seed}.png"
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                img.save(f"{out_dir}/{prompt}_post-seed{seed}.png")
    
    print("Done")





