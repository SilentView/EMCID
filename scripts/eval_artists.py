import os
import sys
import argparse
import json
from typing import Literal

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

from util.globals import *


def generate_artists(
    pipe: StableDiffusionPipeline,
    sample_num,
    file_path=DATA_DIR / "artists" / "prompts_dir" \
        / "erased-1000artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
    out_dir="data/artists/images/sd_orig",
):
    # get holdout image items
    print("Loading data...")
    data = pd.read_csv(file_path)
    to_erase_data = data[data["type"] == "erased"]    # This is not always the full erased set
                                                        # But it is until 100 erasing.
    # shuffle the data
    to_erase_data = to_erase_data.sample(frac=1).reset_index(drop=True)


    hold_out_data = data[data["type"] == "holdout"]
    # reset index
    hold_out_data = hold_out_data.sample(frac=1).reset_index(drop=True)

    # generate holdout images
    print("Generating holdout images...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, row in tqdm(hold_out_data.iterrows(), total=len(hold_out_data)):
        case_number = row["case_number"]
        generator = torch.Generator(pipe.device).manual_seed(int(row["evaluation_seed"]))
        imgs = []
        break_flag = False
        for sample_idx in range(sample_num):
            if os.path.exists(os.path.join(out_dir, f"{case_number}_{sample_idx}.png")):
                break_flag = True
                break
            # get the caption
            prompt = row["prompt"]
            # generate the image
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            # save the image
            imgs.append((sample_idx, img))
        if not break_flag:
            for sample_idx, img in imgs:
                img.save(os.path.join(out_dir, f"{case_number}_{sample_idx}.png"))
   

    # generate images to erase
    for i, row in tqdm(to_erase_data.iterrows(), total=len(to_erase_data)):
        case_number = row["case_number"]
        generator = torch.Generator(pipe.device).manual_seed(int(row["evaluation_seed"]))
        imgs = []
        break_flag = False
        for sample_idx in range(sample_num):
            if os.path.exists(os.path.join(out_dir, f"{case_number}_{sample_idx}.png")):
                # make sure the generator is used in order
                break_flag = True
                break
            # get the caption
            prompt = row["prompt"]
            # generate the image
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            imgs.append((sample_idx, img))
        if not break_flag:
            for sample_idx, img in imgs:
                img.save(os.path.join(out_dir, f"{case_number}_{sample_idx}.png"))


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


def cal_lpips_artists(
    hparam_name,
    num_edit,
    mom2_weight,
    edit_weight,
    sample_num,
    edited_path,
    original_path=DATA_DIR / "artists" / "images" / "sd_orig" / "erased-1000",
    csv_path=DATA_DIR / "artists" / "prompts_dir" / "erased-1000artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
    device="cuda:0"
    ):

    save_path = RESULTS_DIR / "emcid" / hparam_name / "artists"
    # save the results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_file = os.path.join(save_path, 'artists_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    if key in old_data and "hold_out_lpips" in old_data[key] and "edit_lpips" in old_data[key]:
        print(f"{key} already calculated, continue")
        return

    print(lpips)
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(device)
    file_names = os.listdir(original_path)
    file_names = [name for name in file_names if '.png' in name or '.jpg' in name]
    df_prompts = pd.read_csv(csv_path)
    
    df_prompts['lpips_loss'] = df_prompts['case_number'] *0
    # set all lpips loss to nan
    for index, row in df_prompts.iterrows():
        df_prompts.loc[index,'lpips_loss'] = np.nan


    def _compose_file_name(case_number, sample_idx=0):
        return f"{case_number}_{sample_idx}.png"

    # calculate the lpips loss for holdout images
    for index, row in df_prompts.iterrows():
        if row['type'] == 'preserved':
            continue
        case_number = row['case_number']
        orig_files = [_compose_file_name(case_number, i) for i in range(sample_num)]

        lpips_scores = []
        for idx, orig_file in enumerate(orig_files):
            try:
                original_file_path = os.path.join(original_path, orig_file)
                original = image_loader(original_file_path)
                original = original.to(device)
                # print(original.shape)

                edited_file_path = os.path.join(edited_path, _compose_file_name(case_number, idx))
                edited = image_loader(edited_file_path)
                edited = edited.to(device)
                # print(edited.shape)

                l = loss_fn_alex(original, edited)
                # print(f'LPIPS score: {l.item()}')
                if l.item() > 0.3 and row["type"] == "holdout":
                    print(f"case_number: {case_number}, sample_idx: {idx}")
                    print(f"LPIPS score: {l.item()}")
                    print()
                lpips_scores.append(l.item())
            except Exception as e:
                print(e)
                print('No File')
                pass
        # calculate the average lpips loss
        # ignre all the nan values
        df_prompts.loc[index,'lpips_loss'] = np.mean(lpips_scores)
    
    # calculate the average lpips loss and standard deviation
    # preserved concepts are not included in the calculation
    hold_out_mean = np.mean(df_prompts[df_prompts["type"] == "holdout"]['lpips_loss'])
    hold_out_std = np.std(df_prompts[df_prompts["type"] == "holdout"]['lpips_loss'])

    edit_mean = np.mean(df_prompts[df_prompts["type"] == "erased"]['lpips_loss'])
    edit_std = np.std(df_prompts[df_prompts["type"] == "erased"]['lpips_loss'])
    
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    if key not in old_data:
        old_data[key] = {}
    
    # update the results
    old_data[key].update({"hold_out_lpips" : {'mean': hold_out_mean, 'std': hold_out_std}})
    old_data[key].update({"edit_lpips" : {'mean': edit_mean, 'std': edit_std}})
    print("hold_out_mean:", hold_out_mean)
    print("hold_out_std:", hold_out_std)
    print("edit_mean:", edit_mean)
    print("edit_std:", edit_std)

    with open(save_file, 'w') as f:
        json.dump(old_data, f, indent=4)


def cal_clip_score_artists(
    hparam_name,
    num_edit,
    mom2_weight,
    edit_weight,
    sample_num,
    edited_path,
    csv_path=DATA_DIR / "artists" / "prompts_dir" / "erased-1000artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
    device='cuda:0',
    vit_type:Literal["base", "large"] = "large"
    ): 

    if hparam_name == "sd_orig":
        save_path = RESULTS_DIR / "sd_orig" / "artists"
    else:
        save_path = RESULTS_DIR / "emcid" / hparam_name / "artists"

    # check if results exist
    save_file = os.path.join(save_path, 'artists_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = f"sd_orig_{num_edit}"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    if key in old_data and "hold_out_clip" in old_data[key] and "edit_clip" in old_data[key]:
        print(f"{key} already calculated, continue")
        return
    
    model_id = "openai/clip-vit-large-patch14" if vit_type == "large" else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    df_prompts = pd.read_csv(csv_path)
    
    df_prompts['clip_score'] = df_prompts['case_number'] *0

    def _compose_file_name(case_number, sample_idx=0):
        return f"{case_number}_{sample_idx}.png"

    for index, row in tqdm(df_prompts.iterrows(), total=len(df_prompts)):
        if row['type'] == 'preserved':
            continue
        case_number = row['case_number']
        clip_scores = []
        for idx in range(sample_num):
            try:
                edited_img_path = os.path.join(edited_path, _compose_file_name(case_number, idx))
                caption = row['prompt']
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
    hold_out_mean = np.mean(df_prompts[df_prompts["type"] == "holdout"]['clip_score'])
    hold_out_std = np.std(df_prompts[df_prompts["type"] == "holdout"]['clip_score'])

    edit_mean = np.mean(df_prompts[df_prompts["type"] == "erased"]['clip_score'])
    edit_std = np.std(df_prompts[df_prompts["type"] == "erased"]['clip_score'])
    
    # save the results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_file = os.path.join(save_path, 'artists_summary.json')
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # update the results
    if hparam_name == "sd_orig":
        key = f"sd_orig_{num_edit}"
    else:
        key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
    if key not in old_data:
        old_data[key] = {}

    old_data[key].update({"hold_out_clip" : {'mean': hold_out_mean, 'std': hold_out_std}})
    old_data[key].update({"edit_clip" : {'mean': edit_mean, 'std': edit_std}})
    print("hold_out_mean:", hold_out_mean)
    print("hold_out_std:", hold_out_std)
    print("edit_mean:", edit_mean)
    print("edit_std:", edit_std)

    with open(save_file, 'w') as f:
        json.dump(old_data, f, indent=4)
    print('-------------------------------------------------')
    print('\n')

if __name__ == "__main__":
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     torch_dtype=torch.float32,
    #     safety_checker=None,
    #     requires_safety_checker=False,
    # ).to("cuda:0")
    # pipe.set_progress_bar_config(disable=True)

    # with torch.no_grad():
    #     generate_artists(pipe, sample_num=5)

    for edit_num in [1, 5, 10, 50, 100, 500, 1000]:
        cal_clip_score_artists(
            hparam_name="sd_orig",
            num_edit=edit_num,
            mom2_weight=None,
            edit_weight=None,
            sample_num=5,
            edited_path=DATA_DIR / "artists" / "images" / "sd_orig" / f"erased-{edit_num}",
            csv_path=DATA_DIR / "artists" / "prompts_dir" / f"erased-{edit_num}artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
            device='cuda:0',
            vit_type="large"
        )