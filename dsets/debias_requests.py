import csv
import json
from pathlib import Path
from typing import List, Dict, Literal
import os
import random

import torch
import pandas as pd
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
from tqdm import tqdm

from util.globals import *



class DebiasRequestDataset(Dataset):
    def __init__(
            self, 
            data_dir: str=DATA_DIR, 
            dataset_dir: str="debias",
            file_name: str="TIMED_gender_test_set_processed.csv", 
    ):
        self.file_loc = Path(data_dir) / dataset_dir / file_name
        if not self.file_loc.exists():
            raise FileNotFoundError(f"File {self.file_loc} does not exist.")
        
        self.requests: List[Dict] = []
        with open(self.file_loc, "r") as file:
            data = pd.read_csv(file)
        
        templates = ["a photo of {}",
        "an image of {}",
        "{}"]
        
        # get seeds
        random.seed(2023)
        seeds = random.sample(range(100000), len(data))
        for idx, row in data.iterrows():
            request = {}
            request["prompts"] = [template for template in templates]
            request["source"] = row["old"]
            request["seed_train"] = seeds[idx]
            request["dests"] = [row["female"], row["male"]]
            self.requests.append(request)
    
    def __len__(self):
        return len(self.requests)
    
    def __getitem__(self, idx):
        return self.requests[idx]
    
    def sample(self, num: int, seed: int=None):
        if seed:
            torch.manual_seed(seed)
        indices = torch.randperm(len(self.requests))[:num]
        return [self.requests[idx] for idx in indices]


def process_gender_bias_dataset(file_path="data/debias/TIMED_gender_test_set.csv"):
    """
    csv file belike:
    old,new,validation,ex1,ex2,ex3,ex4,ex5
    an attendant,a female attendant,A photo of an attendant,A painting of an attendant,an attendant working,an attendant laughing,an attendant in the workplace,an attendant digital art
    """

    # load csv
    with open(file_path, "r") as f:
        data = pd.read_csv(f)

    random.seed(2023)
    # sample evaluation seed
    prompt_cols = ["validation", *[f"ex{i}" for i in range(1, 6)]]
    
    # for each ex, add new_answer, old_answer columns
    exs = [f"ex{i}" for i in range(1, 6)]
    # for each row, add new_answer, old_answer
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        if " male " in row["new"]:
            female = row["new"].replace("male", "female")
            male = row["new"]
        elif "female" in row["new"]:
            female = row["new"]
            male = row["new"].replace("female", "male")
        else:
            raise ValueError(f"Invalid new: {row['new']}")
        # for ex in exs:
        #     data.loc[idx, f"{ex}_female_answer"] = row[ex].replace(row["old"], female)
        #     data.loc[idx, f"{ex}_male_answer"] = row[ex].replace(row["old"], male)
        
        # data.loc[idx, "validation_female_answer"] = row["validation"].replace(row["old"], female)
        # data.loc[idx, "validation_male_answer"] = row["validation"].replace(row["old"], male)

        data.loc[idx, f"female"] = female
        data.loc[idx, f"male"] = male

    # put female and male in the first two columns
    cols = list(data.columns)
    cols = cols[-2:] + cols[:-2]
    data = data[cols]

    # add seed column
    # for col in prompt_cols:
    #     seed_col = col + "_seed"
    #     data[seed_col] = 0

    # for idx, row in tqdm(data.iterrows(), total=len(data)):
    #     for col in prompt_cols:
    #         seed_col = col + "_seed"
    #         data.loc[idx, seed_col] = int(random.randint(0, 100000))

    # save to csv
    save_path = file_path.replace(".csv", "_processed.csv")
    with open(save_path, "w") as f:
        data.to_csv(f, index=False)
    
    return

if __name__ == "__main__":
    process_gender_bias_dataset()



