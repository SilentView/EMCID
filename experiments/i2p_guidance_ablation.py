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
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
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

# disable tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from util.globals import *

from scripts.eval_i2p_nudity import generate_i2p_imgs, detectNudeClasses, cal_nudity_rate

def i2p_guidance_ablation(
    guidance_scales=[7.5, 6.0, 4.5, 3.0, 1.5, 0.0],
    device="cuda:0",
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    # summarize results
    df = pd.DataFrame(columns=['guidance', 'nudity_rate'])

    for guidance in guidance_scales:
        image_folder = f"data/i2p/images/guidance_ablation/{guidance}"
        with torch.no_grad():
            generate_i2p_imgs(
                pipe=pipe,
                save_dir=image_folder,
                sample_num=1,
                i2p_csv_path="data/i2p/unsafe-prompts4703.csv",
                guidance_scale=guidance,
            )
        
        # evaluate nudity rate
        subprocess.run(["python",
                        "scripts/eval_i2p_nudity.py",
                        f"--folder={image_folder}",
                        "--csv_path=data/i2p/unsafe-prompts4703.csv",
                        f"--save_path=results/sd_orig/i2p/i2p_nudity_{guidance}.csv",
                    ])
        
        nudity_rate = cal_nudity_rate(
                        inp_csv_path=f"results/sd_orig/i2p/i2p_nudity_{guidance}.csv",
                        output_path=f"results/sd_orig/i2p/i2p_nudity_{guidance}.json",
                    )
        
        new_row = {
            'guidance': guidance,
            'nudity_rate': nudity_rate,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    df.to_csv("results/sd_orig/i2p/i2p_guidance_ablation.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    i2p_guidance_ablation(device=args.device)



    

    
