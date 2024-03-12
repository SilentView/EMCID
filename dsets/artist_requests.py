import csv
import json
from pathlib import Path
from typing import List, Dict, Literal
import os
import random

import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
from tqdm import tqdm

from util.globals import *

template = [
    "painting by {}",
    "artwork by {}",
    "style of {}"
]

class ArtistRequestsDataset(Dataset):
    def __init__(self, 
                 src_file=DATA_DIR / "artists" / "info" \
                 / "erased-5artists-towards_art-preserve_true-sd_1_4-method_replace.txt",
                 dest="a photographer, real world scene"
                 ) -> None:
        super().__init__()
        torch.random.manual_seed(2023)
        self.src_file = src_file
        with open(src_file, "r") as f:
            self.data = json.load(f)
        self.requests = []
        self.erase_artists = []
        for artist in self.data:
            seed = torch.randint(0, 100000, (1,)).item()
            self.requests.append(
                    {"prompts": template[:], 
                     "source": artist,
                     "seed_train": seed, 
                     "dest": dest 
                    })
            self.erase_artists.append(artist)
            
    def __len__(self):
        return len(self.requests)
    
    def __getitem__(self, idx):
        return self.requests[idx]


if __name__ == "__main__":

    for number in [1, 5, 10, 50, 100, 500]:
        ds = ArtistRequestsDataset(src_file=DATA_DIR / "artists" / "info" \
                    / f"erased-{number}artists-towards_art-preserve_true-sd_1_4-method_replace.txt",)
        print(len(ds))

        data = pd.read_csv(
            f"data/artists/prompts_dir/erased-{number}artists-towards_art-preserve_true-sd_1_4-method_replace.csv")

        results = []
        for request in tqdm(ds, total=len(ds)):
            # iterate data
            for i, row in data.iterrows():
                if row["artist"] == request["source"] and row["type"] == "holdout":
                    results.append(row["artist"])
            
        print("wrong artists:", len(results) // 5)



    