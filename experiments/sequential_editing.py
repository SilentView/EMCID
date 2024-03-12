import json
from typing import List, Dict
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import (
    AutoProcessor,
    ViTForImageClassification
)

from util.evaluate import calculate_single_cls_score
from util.globals import *
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_cross_attn, COV_CACHE
from experiments.emcid_test import set_weights

from emcid.uce_train import edit_model_uce, edit_model_uce_modified, edit_text_encoder_uce
from dsets.iceb_dataset import *
from dsets.artist_requests import ArtistRequestsDataset
from dsets.global_concepts import get_i2p_editing_requests, NSFWEditRequestDataset



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mom2_weight", type=int, default=4000)
    parser.add_argument("--edit_weight", type=float, default=0.5)
    parser.add_argument("--sample_num", type=int, default=10)

    args = parser.parse_args()

    device = args.device
    mom2_weight = args.mom2_weight
    edit_weight = args.edit_weight
    sample_num = args.sample_num

    # We sequentially edit the concept "The US president"
    # following the order:
    # The US president -> Joe Biden -> Hillary Clinton -> Morgan Freeman
    pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
    pipe.set_progress_bar_config(disable=True)

    hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    # generate pre-edited images
    prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
    
    val_prompts = [
        "An image of the current United States president, high quality, high resolution.",
    ]
    
    seed_train = 2024
    
    requests = [
    {
        "source": "The Current United States president",
        "dest": "Joe Biden",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    }
    ]

    # generate pre edit images
    save_dir = f"{RESULTS_DIR}/emcid/sequential/us_president"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_pre-seed{seed}.png")
    # The US president -> Joe Biden

    stats_dir = "data/stats/text_encoder"
    cache_name = "cache/sequential/"

    pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    stats_dir=STATS_DIR,
                    cache_name=cache_name)
    
    # generate post edit images
    save_dir = f"{RESULTS_DIR}/emcid/sequential/us_president-joe_biden"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_post-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")

    # Joe Biden -> Taylor Swift
    requests = [
        {
        "source": "The current United States president",
        "dest": "Taylor Swift",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    }]

    stats_dir = "data/stats/sequential/us_president-joe_biden"
    # COV_CACHE.clear()
    pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    stats_dir=stats_dir,
                    cache_name=cache_name)
    
    # generate post edit images
    save_dir = f"{RESULTS_DIR}/emcid/sequential/us_president-joe_biden-Taylor Swift"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_post-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")
    
    # Taylor Swift -> Morgan Freeman
    requests = [
        {
        "source": "The current United States president",
        "dest": "Robert Downey Jr.",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    }]
    stats_dir = "data/stats/sequential/us_president-joe_biden-Taylor Swift"
    # COV_CACHE.clear()
    pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    stats_dir=stats_dir,
                    cache_name=cache_name)
    
    # generate post edit images
    save_dir = f"{RESULTS_DIR}/emcid/sequential/us_president-joe_biden-Taylor Swift-Robert Downey Jr."
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_post-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")







    
