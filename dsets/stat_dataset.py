"""
Adapted from emcid project, rome/tok_dataset.py
"""
import json
import os
import urllib.request
import requests
import random
from io import BytesIO

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from util.globals import *


class ImgTxtRndintDataset(Dataset):
    def __init__(self, 
                 data_path="data/ccs_filtered_sub.json", 
                 random_seed=2023,
                 sample_size=3000,
                 ) -> None:
        super().__init__()
        self.random_seed = random_seed
        self.datafile = data_path

        if not os.path.exists(data_path):
            print("Downloading imgs...")
            download_sub(data_path, random_seed=random_seed, sample_size=sample_size)

        with open(data_path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["path"])
        return dict(
            img=self.preprocess_img(img),
            caption=item["caption"],
        )
    
    def preprocess_img(self, image):
        """
        The denoising training process has referred to: 
        https://huggingface.co/docs/diffusers/training/text2image#finetuning
        """
        image = image.convert("RGB") 
        train_transforms = transforms.Compose(
            [
                transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(RESOLUTION),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixel_value = train_transforms(image)
        
        return pixel_value
        


class TokenizedDataset(Dataset):
    """
    Converts a json file of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied direcly to the model.
    """


    def __init__(self, data_path, tokenizer=None, maxlen=None):

        if not os.path.exists(data_path):
            # download data
            print("Downloading data...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json",
                data_path,
            )  
            print("Downloaded data to {}".format(data_path))

        with open(data_path, "r") as f:
            data = json.load(f)
        self.data = [d["caption"] for d in data]
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list),
            position_ids=torch.tensor(position_ids),
            attention_mask=torch.tensor(attention_mask),
        )


def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data


def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]


def download_sub(data_path="./data/ccs_filtered.json", random_seed=2023, sample_size=3000):
    random_seed = random_seed
    
    if not os.path.exists(data_path):
        # download data
        print("Downloading data...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json",
            data_path,
        )  
        print("Downloaded data to {}".format(data_path))

    with open(data_path, "r") as f:
        data = json.load(f)
    # generate random seeds 
    rng = random.Random(random_seed)
    indices = rng.sample(range(0, len(data)), sample_size) 
    # use indices to create a slice of data
    sub_data = []
    added_indices = []

    def _download_single_img(item: dict, idx: int):
        try:
            save_path = f"./cache/stats_img/{idx}.jpg"
            if os.path.exists(save_path):
                sub_data.append(dict(caption=item["caption"], path=save_path, idx=idx))
                added_indices.append(idx)
                return
            response = requests.get(item["url"], timeout=2)
            img = Image.open(BytesIO(response.content))
            # save image
            img_name = item["url"].split("/")[-1]
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img = img.convert("RGB")
            img.save(save_path)
            sub_data.append(dict(caption=item["caption"], path=save_path, idx=idx))
            added_indices.append(idx)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError, 
                PIL.UnidentifiedImageError):
            # This error is caused due the refusal of the server to serve the image
            # we have to resample a new image
            print("failed to load image", item["url"])
            while True:
                new_idx = rng.randint(0, len(data))
                if new_idx not in indices and new_idx not in added_indices:
                    break
            return _download_single_img(data[new_idx], new_idx)

    for idx in tqdm(indices):
        _download_single_img(data[idx], idx)
    assert len(sub_data) == sample_size
    with open("./data/ccs_filtered_sub.json", "w") as f:
        json.dump(sub_data, f, indent=4)


if __name__=="__main__":
    # test TokenizedDataset
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    # tokenizer = pipe.tokenizer
    # ds = TokenizedDataset("data/ccs_filtered.json", tokenizer, maxlen=256)
    # print(ds[0])

    download_sub()