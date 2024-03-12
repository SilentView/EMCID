"""
Given a folder of images and a dataset of captions for those images, this script
will evaluate the CLIP or BLIP score for each image-caption pair
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import warnings
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from PIL import Image
from transformers import (
    CLIPModel, CLIPTokenizer
)
from transformers import (
    AutoProcessor, BlipForImageTextRetrieval,
    ViTForImageClassification
)
import torch
import tqdm

from dsets.iceb_dataset import ObjectPromptDataset


class ImageItem:
    def __init__(self, image_path, score=None):
        """
        To check the naming of images, see `trace_with_patch_text_encoder` function
        in causal_trace.py
        """
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.idx = int(self.image_name.split('_')[1])
        self.class_name = self.image_name.split('_')[0]
        self.kind = self.image_name.split('_')[2]
        self.kind = self.kind if self.kind in ["mlp", "attn"] else None

        self.is_corrupted = True if "corrupt" in self.image_name else False
        self.is_clean = True if "clean" in self.image_name else False
        self.is_restore = True if "restore" in self.image_name else False
        self.restore_type = None
        self.token_to_restore = None
        if self.is_restore:
            self.restore_type = "single" if "w" not in self.image_name.split(
                '_')[-3] else "window"
            self.token_to_restore = self.image_name.split('_')[-1][:-4] # remove .png
        if self.restore_type == "window":
            self.restore_window = int(self.image_name.split('_')[-3][1:])
            self.start_layer = int(self.image_name.split('_')[-4][1:])
        elif self.restore_type == "single":
            self.restore_layer = int(self.image_name.split('_')[-3][1:])
        self.matching_score = score

    def __repr__(self):
        return f"ImageItem({self.image_path})"
    
    def __eq__(self, __value: object) -> bool:
        return self.image_path == __value.image_path


def calculate_single_clip_score(
        clip_model, 
        processor, 
        img, 
        txt, 
        prefix="A photo depicts "):
    """
    Calculate the CLIP score for a single image-text pair
    """

    device = clip_model.device

    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass

    try:
        inputs = processor(images=img, text=prefix + txt, return_tensors="pt")
    except OSError:
        if isinstance(img, str):
            print(f"Image {img} is corrupted, skipping...")
        else:
            print(f"Image is corrupted, skipping...")
    # to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)

    matching_score = 2.5 * outputs.logits_per_image.cpu().item() / clip_model.logit_scale.exp().item()

    return matching_score 


def extract_all_images_clip(image_folder, device="cuda:0", file_path=None):
    """
    Extract all images from the given folder, and return a list of ImageItem
    During this process, the CLIP score is calculated for each image, and all the
    image items are saved to a json file
    """
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".png") and "summary" not in root:
                image_paths.append(os.path.join(root, file))
    
    # extract image items
    image_items: List[ImageItem] = []
    for image_path in image_paths:
        image_items.append(ImageItem(image_path))
    
    # sort image items by idx in ascending order
    image_items.sort(key=lambda x: x.idx)

    # calculate image features
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) 
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    clip_model.eval()

    object_dataset = ObjectPromptDataset()
    print("Calculating image features...")
    for image_item in tqdm.tqdm(image_items):
        img = Image.open(image_item.image_path)
        prefix = "A photo depicts "
        txt = prefix + object_dataset[image_item.idx]["text prompt"]
        input = processor(images=img, text=txt, return_tensors="pt")
        # to device
        input = {k: v.to(device) for k, v in input.items()}
        output = clip_model(**input)
        # we follow the calculation method in concept-ablate project
        image_item.matching_score = 2.5 * output.logits_per_image.cpu().item() / clip_model.logit_scale.exp().item()

    # save image items
    file_path = file_path if file_path else "results/images/text_encoder/causal_trace/summary/img_items_clip.json" 
    # create folder if not exist
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        # make image_items json serializable by converting to dict
        image_items = [vars(img_item) for img_item in image_items]
        json.dump(image_items, f, indent=4)
    return image_items
    

def extract_all_images_blip(image_folder, device="cuda:0", file_path=None):
    """
    Extract all images from the given folder, and return a list of ImageItem
    During this process, the BLIP score is calculated for each image, and all the
    image items are saved to a json file

    Note that if file_path is not None, we will load the image items from the file
    and avoid repeating the calculation, simply appending the new image items
    """
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".png") and "summary" not in root:
                image_paths.append(os.path.join(root, file))
    
    # extract image items
    image_items = []
    for image_path in image_paths:
        image_items.append(ImageItem(image_path))
    
    if file_path and os.path.exists(file_path):
        # load image items from file
        with open(file_path, "r") as f:
            existing_items = json.load(f)
            existing_items = [ImageItem(img_item["image_path"], score=img_item["matching_score"]) for img_item in existing_items]
            existing_items.sort(key=lambda x: x.idx)
        image_items = [img_item for img_item in image_items if img_item not in existing_items]
    else:
        existing_items = []

    if len(image_items) == 0:
        print("All image items already exist in the file, no need to calculate again.")
        return existing_items 

    # sort image items by idx in ascending order
    image_items.sort(key=lambda x: x.idx)

    # note that the BLIP checkpoint is finetuned on COCO dataset, which means it 
    # probably can't generalize well to zero-shot scene
    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    blip_model.eval()
    object_dataset = ObjectPromptDataset()

    print("Calculating image and text matching score...")
    for image_item in tqdm.tqdm(image_items):
        # use the ITM head of BLIP to calculate matching score
        image_item.matching_score = calculate_single_blip_score(blip_model, 
                                                                processor, 
                                                                image_item.image_path,
                                                                object_dataset[image_item.idx]["text prompt"], 
                                                                device)
    # save image items
    file_path = file_path if file_path else "results/images/text_encoder/causal_trace/summary/img_items_blip.json" 
    # create folder if not exist
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # append to existing file
    existing_items.extend(image_items)
    with open(file_path, "w") as f:
        # make image_items json serializable by converting to dict
        all_items = [vars(img_item) for img_item in existing_items]
        json.dump(all_items, f, indent=4)
    
    return image_items


def calculate_single_blip_score(
        blip_model, 
        processor, 
        img, 
        txt, 
        device=None, 
        prefix="A photo depicts "):
    """
    Calculate the BLIP score for a single image-text pair
    """

    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass

    device = device if device else blip_model.device

    try:
        inputs = processor(images=img, text=prefix + txt, return_tensors="pt")
    except OSError:
        if isinstance(img, str):
            print(f"Image {img} is corrupted, skipping...")
        else:
            print(f"Image is corrupted, skipping...")
    # to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = blip_model(**inputs)

    itm_score = torch.nn.functional.softmax(outputs.itm_score, dim=1)[:,1]
    return itm_score.item()


def calculate_single_cls_score(
        classifier, 
        processor, 
        imgs: List, 
        class_id: int,
        return_std: bool=False):
    """
    Calculate the classification score for a batch of images, 
    for a single class
    """
    class_id = int(class_id)

    if isinstance(imgs[0], str):
        imgs = [Image.open(img) for img in imgs]
    elif isinstance(imgs[0], Image.Image):
        pass
    device = classifier.device

    inputs = processor(images=imgs, return_tensors="pt")
    # to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = classifier(**inputs)

    cls_scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    mean_cls_score = cls_scores[:, class_id].mean().item()
    if not return_std:
        return mean_cls_score
    else:
        std_cls_score = cls_scores[:, class_id].std().item()
        return mean_cls_score, std_cls_score


def extract_all_images_cls(image_folder, device="cuda:0", file_path=None, vit_type: Literal["base", "large"]="base"):

    # initialize lookup table
    class2id = None
    with open("data/iceb_data/class2id.json", "r") as f:
        class2id = json.load(f)

    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".png") and "summary" not in root:
                image_paths.append(os.path.join(root, file))
    
    # extract image items
    image_items = []
    for image_path in image_paths:
        image_items.append(ImageItem(image_path))
    
    if file_path and os.path.exists(file_path):
        # load image items from file
        with open(file_path, "r") as f:
            existing_items = json.load(f)
            existing_items = [ImageItem(img_item["image_path"], score=img_item["matching_score"]) for img_item in existing_items]
            existing_items.sort(key=lambda x: x.idx)
        image_items = [img_item for img_item in image_items if img_item not in existing_items]
    else:
        existing_items = []

    if len(image_items) == 0:
        print("All image items already exist in the file, no need to calculate again.")
        return existing_items 

    # sort image items by idx in ascending order
    image_items.sort(key=lambda x: x.idx)

    # note that the BLIP checkpoint is finetuned on COCO dataset, which means it 
    # probably can't generalize well to zero-shot scene

    model_id = "google/vit-base-patch16-224" if vit_type == "base" else "google/vit-large-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    classifier = ViTForImageClassification.from_pretrained(model_id).to(device)
    
    classifier.eval()

    print("Calculating image and class matching score...")
    for image_item in tqdm.tqdm(image_items):
        # use the ITM head of BLIP to calculate matching score
        try:
            image_item.matching_score = calculate_single_cls_score(classifier, 
                                                                processor, 
                                                                [image_item.image_path],
                                                                class2id[image_item.class_name])
        except KeyError:
            continue
    # save image items
    file_path = file_path if file_path else f"results/images/text_encoder/causal_trace/summary/img_items_cls_vit-{vit_type}.json" 
    # create folder if not exist
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # append to existing file
    existing_items.extend(image_items)
    with open(file_path, "w") as f:
        # make image_items json serializable by converting to dict
        all_items = [vars(img_item) for img_item in existing_items]
        json.dump(all_items, f, indent=4)
    
    return image_items