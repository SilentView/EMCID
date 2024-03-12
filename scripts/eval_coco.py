# This script will load coco-30k images, generate images using captions,
# and compute the FID and LPIPS scores.
import os
import sys
import argparse
import json
import shutil
from typing import List, Literal

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import pandas as pd
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import lpips
import pytorch_fid

from util.globals import *


# desired size of the output image
imsize = 64
loader = transforms.Compose([
    # the original uce code gives only an int, which may not retrun a square image
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)


def generate_coco_30k(
        pipe: StableDiffusionPipeline, 
        sample_num,
        file_path=DATA_DIR / "coco" / "coco_30k.csv",
        out_dir="data/coco/images/sd_orig",
        random_order=True
        ):
    """
    Generate coco-30k images using the given pipeline.
    """
    print("Loading data...")
    data = pd.read_csv(file_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if random_order:
        # shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)

    # generate images using the pipeline
    print("Generating images...")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        generator = torch.Generator(device=pipe.device).manual_seed(int(row["evaluation_seed"]))
        for sample_idx in range(sample_num):
            if os.path.exists(os.path.join(out_dir, f"{row['coco_id']}_{sample_idx}.png")):
                continue
            # get the caption
            prompt = row["prompt"]
            # generate the image
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            # save the image
            img.save(os.path.join(out_dir, f"{row['coco_id']}_{sample_idx}.png"))


def cal_lpips_coco(
    hparam_name,
    num_edit,
    mom2_weight,
    edit_weight,
    sample_num,
    edited_path,
    output_folder=None,
    dataset="artists",
    original_path=DATA_DIR / "coco" / "images" / "sd_orig",
    csv_path=DATA_DIR / "coco" / "coco_30k.csv",
    device="cuda:0"
    ):
    if output_folder is None:
        if hparam_name == "sd_orig":
            save_path = RESULTS_DIR / "sd_orig" / dataset
        else:
            save_path = RESULTS_DIR / "emcid" / hparam_name / dataset
    else:
        save_path = output_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_file = os.path.join(save_path, 'coco_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = "sd_orig"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    
    if key in old_data and "lpips" in old_data[key]:
        return

    real_orig = True
    if "sd_orig" in str(original_path):
        real_orig = False
        print("Using generated original images", not real_orig)
        print(original_path)
    print(lpips)
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(device)
    file_names = os.listdir(original_path)
    file_names = [name for name in file_names if '.png' in name or '.jpg' in name]
    df_prompts = pd.read_csv(csv_path)
    
    df_prompts['lpips_loss'] = df_prompts['case_number'] *0

    def _detect_coco_id(file_name, coco_id):
        return int(file_name.split("_")[-1].split(".")[0]) == int(coco_id)
    
    def _compose_coco_file_name(coco_id, generated=False, sample_idx=0):
        if generated:
            return f"{coco_id}_{sample_idx}.png"
        else:
            return f"COCO_val2014_{str(coco_id).zfill(12)}.jpg"

    for index, row in tqdm(df_prompts.iterrows(), total=len(df_prompts)):
        coco_id = row['coco_id']
        # since there is no duplicate coco_id in the dataset, we can use the
        # coco_id to find the corresponding file name
        if real_orig:
            orig_files = [_compose_coco_file_name(coco_id, False)] * sample_num
        else:
            orig_files = [_compose_coco_file_name(coco_id, True, i) for i in range(sample_num)]

        lpips_scores = []
        for idx, orig_file in enumerate(orig_files):
            try:
                original_file_path = os.path.join(original_path, orig_file)
                original = image_loader(original_file_path)
                original = original.to(device)
                # print(original.shape)

                edited_file_path = os.path.join(edited_path, _compose_coco_file_name(coco_id, True, idx))
                edited = image_loader(edited_file_path)
                edited = edited.to(device)
                # print(edited.shape)

                l = loss_fn_alex(original, edited)
                # print(f'LPIPS score: {l.item()}')
                lpips_scores.append(l.item())
            except Exception as e:
                print(e)
                print('No File')
                pass
        df_prompts.loc[index,'lpips_loss'] = np.mean(lpips_scores)
    
    # calculate the average lpips loss and standard deviation
    mean = df_prompts['lpips_loss'].mean()
    std = df_prompts['lpips_loss'].std()
    
    # save the results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_file = os.path.join(save_path, 'coco_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = "sd_orig"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')

    if key not in old_data:
        old_data[key] = {}
    old_data[key].update({"lpips" : {'mean': mean, 'std': std}})

    with open(save_file, 'w') as f:
        json.dump(old_data, f, indent=4)


def cal_clip_score_coco(
    hparam_name,
    num_edit,
    mom2_weight,
    edit_weight,
    sample_num,
    edited_path,
    out_put_folder=None,
    dataset="artists",
    csv_path=DATA_DIR / "coco" / "coco_30k.csv",
    device='cuda:0',
    vit_type:Literal["base", "large"] = "large"
    ): 

    if out_put_folder is None:
        if hparam_name == "sd_orig":
            save_path = RESULTS_DIR / "sd_orig" / dataset
        else:
            save_path = RESULTS_DIR / "emcid" / hparam_name / dataset
    else:
        save_path = out_put_folder

    # check if results exist
    save_file = os.path.join(save_path, 'coco_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = "sd_orig"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    if key in old_data and f"clip_vit_{vit_type}" in old_data[key]:
        return
    
    model_id = "openai/clip-vit-large-patch14" if vit_type == "large" else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    df_prompts = pd.read_csv(csv_path)
    
    df_prompts['clip_score'] = df_prompts['case_number'] *0

    def _detect_coco_id(file_name, coco_id):
        return int(file_name.split("_")[-1].split(".")[0]) == int(coco_id)
    
    def _compose_coco_file_name(coco_id, generated=False, sample_idx=0):
        if generated:
            return f"{coco_id}_{sample_idx}.png"
        else:
            return f"COCO_val2014_{str(coco_id).zfill(12)}.jpg"

    for index, row in tqdm(df_prompts.iterrows(), total=len(df_prompts)):
        coco_id = row['coco_id']
        case_number = row['case_number']
        clip_scores = []
        for idx in range(sample_num):
            try:
                edited_img_path = os.path.join(edited_path, _compose_coco_file_name(coco_id, True, idx))
                caption = df_prompts.loc[df_prompts.case_number==case_number]['prompt'].item()
                im = Image.open(edited_img_path)
                inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
                clip_scores.append(clip_score.item())
            except Exception as e:
                print(e)
                print('No File')
                pass
        df_prompts.loc[index,'clip_score'] = np.mean(clip_scores)
        # print(f'CLIP score: {np.mean(clip_scores)}')
    
    # calculate the average lpips loss and standard deviation
    mean = df_prompts['clip_score'].mean()
    std = df_prompts['clip_score'].std()
    
    # save the results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_file = os.path.join(save_path, 'coco_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = "sd_orig"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    if key not in old_data:
        old_data[key] = {}

    old_data[key].update({f"clip_vit_{vit_type}" : {'mean': mean, 'std': std}})

    with open(save_file, 'w') as f:
        json.dump(old_data, f, indent=4)
    print(f"Mean CLIP score: {old_data[key][f'clip_vit_{vit_type}']['mean']}")
    print('-------------------------------------------------')
    print('\n')


def get_coco_30k_sub(
    orig_path="data/coco/val2014", 
    out_path="data/coco/coco-30k",
    csv_file_path="data/coco/coco_30k.csv",):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data = pd.read_csv(csv_file_path)

    def _compose_coco_file_name(coco_id, generated=False, sample_idx=0):
        if generated:
            return f"{coco_id}_{sample_idx}.png"
        else:
            return f"COCO_val2014_{str(coco_id).zfill(12)}.jpg"

    for i, row in tqdm(data.iterrows(), total=len(data)):
        coco_id = row['coco_id']
        orig_file = _compose_coco_file_name(coco_id, False)
        shutil.copy(os.path.join(orig_path, orig_file), os.path.join(out_path, orig_file))




if __name__ == "__main__":
    cal_clip_score_coco(
        hparam_name="sd_orig",
        num_edit=None,
        mom2_weight=None,
        edit_weight=None,
        sample_num=1,
        edited_path=DATA_DIR / "coco" / "images" / "sd_orig",
        vit_type="large",
        device="cuda:1"
    )

    import subprocess

    subprocess.run(["python", 
                    "scripts/test_fid_score.py", 
                    "--generated_images_folder=data/coco/images/sd_orig", 
                    "--coco_30k_folder=data/coco/coco-30k", 
                    "--output_folder=results/sd_orig/artists", 
                    "--batch_size=32", 
                    "--output_file=coco_summary.json",
                    "--save_npz_folder=data/stats/fid", 
                    "--device=cuda:1", 
                    "--dict_key=sd_orig"])
