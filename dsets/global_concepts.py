import csv
import json
from pathlib import Path
from typing import List, Dict, Literal
import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from util.globals import *


def make_a_single_request(device):
    """
    {
        "source_regions": [[(xtl, ytl, xbr, ybr), ...], ...],
        "source_imgs": [img_path1, img_path2, ...],
        "source_prompts": [prompt1, prompt2, ...],
        "dest_prompts": [prompt1, prompt2, ...],
    }
    """
    request = {
        "source_regions": [],
        "source_imgs": [],
        "source_prompts": [],
        "dest_prompts": [],
    }

    train_edit_prompts = [
        ("A Wheatfield, with Cypresses by Vincent van Gogh",2219, "Vincent van Gogh"),
        ("Almond Blossoms by Vincent van Gogh",	4965, "Vincent van Gogh"),
        ("Bridge at Trinquetaille by Vincent van Gogh",3370,"Vincent van Gogh"),
    ]

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # generate source images
    with torch.no_grad():
        for item in train_edit_prompts:
            save_dir = "data/artists/test/train"
            file_name = f"pre_edit_{item[0]}.png"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if os.path.exists(os.path.join(save_dir, file_name)):
                request["source_imgs"].append(save_dir + "/" + file_name)
                request["source_prompts"].append(item[0])
                request["dest_prompts"].append(item[0].replace(item[2], "a photographer, real world scene"))
                # random sample 2 region from the image
                img = Image.open(os.path.join(save_dir, file_name))
                width, height = img.size
                for _ in range(3):
                    xtl = random.randint(0, width - 100)
                    ytl = random.randint(0, height - 100)
                    r_w = random.randint(50, 100)
                    r_h = random.randint(50, 100)
                    xbr = xtl + r_w
                    ybr = ytl + r_h
                    request["source_regions"].append([(xtl, ytl, xbr, ybr)])
                continue
            generator = torch.Generator("cuda").manual_seed(item[1])
            img = pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            width, height = img.size
            for _ in range(3):
                xtl = random.randint(0, width - 100)
                ytl = random.randint(0, height - 100)
                r_w = random.randint(50, 100)
                r_h = random.randint(50, 100)
                xbr = xtl + r_w
                ybr = ytl + r_h
                request["source_regions"].append([(xtl, ytl, xbr, ybr)])
            img.save(os.path.join(save_dir, file_name))
            request["source_imgs"].append(save_dir + "/" + file_name)
            request["source_prompts"].append(item[0])
            request["dest_prompts"].append(item[0].replace(item[2], "a photographer, real world scene"))
    
    # save request
    with open("data/artists/test/train/request.json", "w") as f:
        json.dump(request, f, indent=4)
    return request


def get_i2p_editing_requests(
        i2p_path="data/i2p/unsafe-prompts4703.csv",
        category="sexual",
        num_requsts=2,
        batch_size=3,
        source="nudity"
        ):
    """
    categories: hate, harassment, violence, self-harm, 
                sexual, shocking, illegal activity
    source should be a single word, like nudity
    """
    if category == "sexual":
        safe_words = "harassment, sexual, nudity, obscene gestures"
    else:
        raise NotImplementedError("only sexual content is supported now")

    with open(i2p_path, "r") as f:
        reader = csv.DictReader(f)
        requests = []
        for req_idx in range(num_requsts):
            request = {
                "seeds": [],
                "source_prompts": [],
                "safe_words": [],
                "source_cat": category,
                "source": source,
                "indices": [],
                "dest": "",
            }
            for batch_idx in range(batch_size):
                while True:
                    row = next(reader)
                    if category in row["categories"] and int(row["hard"]) == 1:
                        break
                request["source_prompts"].append(row["prompt"])
                request["seeds"].append(int(row["evaluation_seed"]))
                request["safe_words"].append(safe_words)
                request["indices"].append(req_idx * batch_size + batch_idx)
            requests.append(request)
    return requests


NSFW_TMP = [
    "{}",
    "A photo depicting {}",
    "A photo showing {}"
]


class NSFWEditRequestDataset(Dataset):
    def __init__(
        self,
        keywords: List[str],
        num_prompts: int,
    ):
        requests = []
        random_seeds = np.random.randint(0, 10000, size=len(keywords))
        for idx, keyword in enumerate(keywords):
            request = {}
            request["source"] = keyword
            request["dest"] = " "
            request["prompts"] = NSFW_TMP[:num_prompts]
            request["seed"] = random_seeds[idx]
            requests.append(request)
        self.requests = requests 
    
    def __len__(self):
        return len(self.requests)
    
    def __getitem__(self, idx):
        return self.requests[idx]


    
if __name__ == "__main__":
    requests = get_i2p_editing_requests(num_requsts=10)
    for request in requests:
        for prompt in request["source_prompts"]:
            print(prompt)
            print()

