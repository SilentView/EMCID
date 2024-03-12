import csv
import json
from pathlib import Path
from typing import List, Dict, Literal
import os
import random
from io import BytesIO

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import PIL
import requests

from util.globals import *
from emcid.emcid_hparams import ContrastEMCIDHyperParams 

CONTEXT_TEMPLATE = ["{} in a realistic style portrait image", 
                    "{}, a portrait",  
                    "realistic painting of {}", 
                    "a current image of {}", 
                    "{}, news image", 
                    "a beautiful photograph of {}", 
                    "realistic drawing of {}", 
                    "{}, realistic portrait", 
                    "{} in a photo"] 

class TIMEDRoadRequestDataset(Dataset):
    def __init__(
            self, 
            dataset: Literal["timed", "road", "timed-val", "road-val"], 
            data_dir: str = DATA_DIR,
            data_path: str = None,
            use_more_tmp: bool = False,
            num_negative_images: int = None,
            num_negative_prompts: int = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.data_path = data_path
        self.use_more_tmp = use_more_tmp
        self.num_negative_images = num_negative_images
        self.num_negative_prompts = num_negative_prompts
        self.requests = self._load_requests()

    def _load_requests(self) -> List[Dict]:

        if self.data_path is not None:
            file_path = self.data_path
        else:
            if self.dataset == "timed":
                file_path = self.data_dir / "timed" / f"TIMED_test_set_filtered_SD14.csv"
            elif self.dataset == "road":
                file_path = self.data_dir / "road" / f"RoAD_test.csv"
            elif self.dataset == "timed-val":
                file_path = self.data_dir / "timed" / f"TIMED_validation_set.csv"
            elif self.dataset == "road-val":
                file_path = self.data_dir / "road" / f"RoAD_validation.csv"
            else:
                raise ValueError("Invalid dataset")

        data = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        # lower all the keys and values
        new_data = []
        for row in data:
            new_row = {}
            for k,v in row.items():
                new_row[k.lower()] = v.lower()
            new_data.append(new_row)
        
        templates = ["An image of {}", "A photo of {}", "{}"] \
                if not self.use_more_tmp else ["{}"] + CONTEXT_TEMPLATE

        prompt_key = "old" if "timed" in self.dataset else "prompt"

        if self.num_negative_prompts or self.num_negative_images:
            with open("data/ccs_filtered_sub.json", "r") as f:
                ccs_data = json.load(f)
            ccs_data = ccs_data[:self.num_negative_prompts]
            negative_prompts = []
            negative_images = []

            # load negative_prompts and negative images from COCO dataset
            for i, item in enumerate(ccs_data):
                img = Image.open(item["path"])
                img = img.convert("RGB")
                negative_prompts.append(item["caption"])
                negative_images.append(img)
                if len(negative_prompts) == max(self.num_negative_images, self.num_negative_prompts):
                    break
        
        ret_requests = []
        # convert data to request form
        for idx, row in enumerate(new_data):
            request = {}
            request["prompts"] = [template for template in templates]
            request["source"] = row[prompt_key]
            request["seed"] = None  # Following ReFACT, use global seed
            request["indices"] = [idx for _ in range(len(templates))] 
            request["dest"] = row["new"]
            request["negative_prompts"] = negative_prompts if self.num_negative_prompts else None
            request["negative_images"] = negative_images if self.num_negative_images else None
            if row.get("is_human_name", "not_found") != "not_found":
                to_bool = lambda x: True if x.lower() == "true" else False
                request["is_human_name"] = to_bool(row["is_human_name"])
        
            ret_requests.append(request)

        return ret_requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        return self.requests[idx]