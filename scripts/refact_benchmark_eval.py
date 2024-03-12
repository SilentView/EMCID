import argparse
import os
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import open_clip
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from util.globals import *
from dsets.timed_road_dataset import TIMEDRoadRequestDataset
from emcid.emcid_hparams import EMCIDHyperParams, ContrastEMCIDHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_clip
from experiments.emcid_test import set_weights


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def emcid_test(
        dataset="timed", 
        hparam_name="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", 
        mom2_weight=None, 
        edit_weight=None,
        global_seed=0,
        generate_oracle=True,
        device="cuda:0"):
    device = torch.device(device)
    if "contrast" in hparam_name:
        requests = TIMEDRoadRequestDataset(dataset,
                                        use_more_tmp=True,
                                        num_negative_images=20,
                                        num_negative_prompts=20)
    elif "txt-cont" in hparam_name:
        requests = TIMEDRoadRequestDataset(dataset,
                                        use_more_tmp=False,
                                        num_negative_images=0,
                                        num_negative_prompts=20)
    else:
        if dataset == "timed":
            requests = TIMEDRoadRequestDataset(dataset, 
                                               data_path=f"data/{dataset}/TIMED_test_set_filtered_SD14_name_labled.csv")
        else:
            requests = TIMEDRoadRequestDataset(dataset)

    hparam_path = HPARAMS_DIR / f"{hparam_name}.json"

    if "contrast" in hparam_name:
        hparams = ContrastEMCIDHyperParams.from_json(hparam_path)
    else:
        hparams = EMCIDHyperParams.from_json(hparam_path)
    
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight

    if dataset == "timed":
        file_path = "data/timed/TIMED_test_set_filtered_SD14.csv"
    elif dataset == "road":
        file_path = "data/road/RoAD_test.csv"
    elif dataset == "timed-val":
        file_path = DATA_DIR / "timed" / "TIMED_validation_set.csv"
    elif dataset == "road-val":
        file_path = DATA_DIR / "road" / "RoAD_validation.csv"
    else:
        raise ValueError("Invalid dataset")

    valid_set = pd.read_csv(file_path)
    # load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    if "contrast" in hparam_name:
        clip_model_id = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        processor = CLIPProcessor.from_pretrained(clip_model_id)

    pipe.set_progress_bar_config(disable=True)

    orig_edit_weight = hparams.edit_weight
    for request in tqdm(requests):
        # if not request["is_human_name"]:
        #     continue
        # generate oracle results
        if generate_oracle:
            with torch.no_grad():
                generate_imgs_for_eval_single(
                    pipe,
                    hparam_name,
                    mom2_weight,
                    orig_edit_weight,   # only for saving path specification
                    global_seed,
                    valid_set.iloc[request["indices"][0]],
                    dataset,
                    oracle=True
                )

        # edit the model
        cache_name = f"cache/{hparam_name}/{dataset}/seed_{global_seed}/"
        print(cache_name)
        print(hparam_name)
        if "contrast" in hparam_name:
            set_seed(global_seed)
            new_clip_model, _ = apply_emcid_to_clip(
                                    clip_model,
                                    processor,
                                    [request],
                                    hparams,
                                    device,
                                    mom2_weight=mom2_weight,
                                    edit_weight=edit_weight,
                                    cache_name=cache_name)
            orig_text_model = pipe.text_encoder.text_model
            pipe.text_encoder.text_model = new_clip_model.text_model

        else:
            pipe, orig_text_encoder = \
                apply_emcid_to_text_encoder(
                            pipe, 
                            [request], 
                            hparams, 
                            device, 
                            mom2_weight=mom2_weight, 
                            edit_weight=edit_weight,
                            cache_name=cache_name,
                            return_orig_text_encoder=True,
                            verbose=False)

        with torch.no_grad():
            print("Generating images for eval")
            generate_imgs_for_eval_single(
                pipe,
                hparam_name,
                mom2_weight,
                orig_edit_weight,   # only for saving path specification
                global_seed,
                valid_set.iloc[request["indices"][0]],
                dataset,
                oracle=False
            )

        # reset the model
        if "contrast" in hparam_name:
            pipe.text_encoder.text_model = orig_text_model
            clip_model.text_model = deepcopy(orig_text_model)
        else:
            pipe.text_encoder = orig_text_encoder

def generate_imgs_for_eval_single(
    pipe,
    hparam_name,
    mom2_weight,
    edit_weight,
    global_seed,
    row_raw,
    dataset,
    oracle=False):

    # use the same seed strategy as refact
    set_seed(global_seed)

    row = dict()
    for k,v in row_raw.items():
        row[k.lower()] = v.lower()

    output_path = None
    if "timed" in dataset:
        path = "old"
    elif "road" in dataset:
        path = "prompt"

    if oracle:
        source = row[path]
        new = row['new']
        output_path = f"./results/images/{dataset}/oracle/source_{source}_dest_{new}"
    else:
        source = row[path]
        new = row['new']
        output_path = f"./results/images/{dataset}/{hparam_name}_w{int(mom2_weight)}"\
                      f"{'' if edit_weight == 0.5 else f'_ew{edit_weight}'}"\
                      f"/emcid/source_{source}_dest_{new}"
    
    if not oracle:
        prompt = row[path]
    else:
        prompt = row['new']
    
    # generate images using original prompt
    save_path = f"{output_path}/{prompt}/seed_{global_seed}.png"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        img = pipe(prompt).images[0]
        img.save(save_path)
    
    # generate images using generalization prompts
    # using a batch size of 3
    for i in range(1, 6, 3):
        if "timed" in dataset:
            positive_new = f'gt'
        elif "road" in dataset:
            positive_new = f'positive_new'
        
        left = i
        right = min(i+3, 6)
        if not oracle:
            prompts = [row[f"positive{j}"] for j in range(left, right)]
        else:
            prompts = [row[positive_new + str(j)] for j in range(left, right)]

        save_paths = [f"{output_path}/{prompt}/seed_{global_seed}.png" for prompt in prompts]
        for save_path in save_paths:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

        continue_flag = True
        for i, save_path in enumerate(save_paths):
            if not os.path.exists(save_path):
                continue_flag = False
                break
        if continue_flag:
            continue

        imgs = pipe(prompts).images
        for i, save_path in enumerate(save_paths):
            imgs[i].save(save_path)
        
    for i in range(1, 6, 3):
        left = i
        right = min(i+3, 6)
        prompts = [row[f"negative{j}"] for j in range(left, right)]
        save_paths = [f"{output_path}/{prompt}/seed_{global_seed}.png" for prompt in prompts]
        # make directories
        for save_path in save_paths:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

        continue_flag = True
        for i, save_path in enumerate(save_paths):
            if not os.path.exists(save_path):
                continue_flag = False
                break
        if continue_flag:
            continue

        imgs = pipe(prompts).images
        for i, save_path in enumerate(save_paths):
            imgs[i].save(save_path)

def generate_imgs_for_eval(
    pipe, 
    hparam_name,
    mom2_weight, 
    edit_weight,
    global_seed,
    requests, 
    dataset, 
    oracle=False):

    if dataset == "timed":
        file_path = "data/timed/TIMED_test_set.csv"
    elif dataset == "road":
        file_path = "data/road/RoAD_test.csv"
    elif dataset == "timed-val":
        file_path = DATA_DIR / "timed" / "TIMED_validation_set.csv"
    elif dataset == "road-val":
        file_path = DATA_DIR / "road" / "RoAD_validation.csv"
    else:
        raise ValueError("Invalid dataset")

    request_indices = [request["indices"][0] for request in requests]
    valid_set = pd.read_csv(file_path)

    for i, raw_row in tqdm(valid_set.iterrows(), total=len(valid_set)):
        if i not in request_indices:
            continue
        generate_imgs_for_eval_single(
            pipe,
            hparam_name,
            mom2_weight,
            edit_weight,
            global_seed,
            raw_row,
            dataset,
            oracle=oracle
        )


missing = []

def get_scores_new(preprocess_val, tokenizer, model, output_path, prompt, old, new, seed, device):
    generated_image_path = f"{output_path}/{prompt}/seed_{seed}.png"
    try:
        image = Image.open(generated_image_path)
    except Exception as e:
        print(generated_image_path)
        missing.append(generated_image_path)
        raise ValueError("Missing image")

    try:
        image = preprocess_val(image).unsqueeze(0).to(device)
    except Exception as e:
        print(generated_image_path)
        raise ValueError("Invalid image")

    if type(new) == list:
        text = tokenizer([old, *new]).to(device)
    else:
        text = tokenizer([old, new]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return text_probs[0]


def eval_all(
        dataset="timed", 
        num_seeds=25,
        hparam_name="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", 
        mom2_weight=None,
        edit_weight=None,
        device="cuda:0", 
        oracle=False):
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    if "contrast" in hparam_name:
        hparams = ContrastEMCIDHyperParams.from_json(HPARAMS_DIR / f"{hparam_name}.json")
    else:
        hparams = EMCIDHyperParams.from_json(HPARAMS_DIR / f"{hparam_name}.json")
    
    hparams = set_weights(hparams, mom2_weight, edit_weight)
    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    if dataset == "timed":
        file_path = "data/timed/TIMED_test_set_filtered_SD14.csv"
    elif dataset == "road":
        file_path = "data/road/RoAD_test.csv"
    elif dataset == "timed-val":
        file_path = DATA_DIR / "timed" / "TIMED_validation_set.csv"
    elif dataset == "road-val":
        file_path = DATA_DIR / "road" / "RoAD_validation.csv"
    else:
        raise ValueError("Invalid dataset")

    valid_set = pd.read_csv(file_path)

    all_efficacy = []
    all_generality = []
    all_generality_75 = []
    all_generality_90 = []
    all_specificity = []
    all_validity = []
    all_old = []
    all_new = []
    for i, raw_row in tqdm(valid_set.iterrows(), total=len(valid_set)):
        row = dict()
        for k,v in raw_row.items():
            row[k.lower()] = v.lower()

        efficacy = []
        generality = []
        generality_75 = []
        generality_90 = []
        specificity = []
        validity = []
        output_path = None
        
        if "timed" in dataset:
            path = "old"
        elif "road" in dataset:
            path = "prompt"

        if oracle:
            source = row[path]
            new = row['new']
            output_path = f"./results/images/{dataset}/oracle/source_{source}_dest_{new}"
        else:
            source = row[path]
            new = row['new']
            edit_weight_str = f"{'' if edit_weight == 0.5 else f'_ew{edit_weight}'}"
            output_path = f"./results/images/{dataset}/{hparam_name}_w{mom2_weight}{edit_weight_str}/emcid/source_{source}_dest_{new}"
            
        if not oracle:
            prompt = row[path]
        else:
            prompt = row['new']

        for seed in range(0, num_seeds):
            scores = get_scores_new(preprocess_val, tokenizer, model, output_path, prompt, row['old'], row['new'], seed, device)
            success_indicator = (scores[1] > scores[0]).item()
            if success_indicator:
                efficacy += [1]

            else:
                efficacy += [0]

            ctr_generality = 0
            ctr_generality90 = 0
            ctr_generality75 = 0
            for i in range(1, 6):
                if "timed" in dataset:
                    positive_old = f'positive{i}'
                    positive_new = f'gt{i}'
                elif "road" in dataset:
                    positive_old = f'positive_old{i}'
                    positive_new = f'positive_new{i}'
                
                if not oracle:
                    prompt = row[f'positive{i}']
                else:
                    prompt = row[positive_new]
                
                scores = get_scores_new(preprocess_val, tokenizer, model, output_path, prompt, row[positive_old], row[positive_new], seed,
                                        device)
                success_indicator = (scores[1] > scores[0]).item()
                if success_indicator:
                    ctr_generality += 1

                success_indicator = scores[1] > 0.9
                if success_indicator:
                    ctr_generality90 += 1

                success_indicator = scores[1] > 0.75
                if success_indicator:
                    ctr_generality75 += 1

            generality.append(ctr_generality / 5)
            generality_90.append(ctr_generality90 / 5)
            generality_75.append(ctr_generality75 / 5)

            ctr_specificity = 0
            for i in range(1, 6):
                # specificity - oracle is like the baseline
                if "timed" in dataset:
                    negative_new = f'gn{i}'
                elif "road" in dataset:
                    negative_new = f'negative_new{i}'

                scores = get_scores_new(preprocess_val, tokenizer, model, output_path, row[f'negative{i}'], row[f'negative{i}'], row[negative_new], seed, device)
                success_indicator = (scores[1] < scores[0]).item()
                
                if success_indicator:
                    ctr_specificity += 1
            
            specificity.append(ctr_specificity / 5)

        if missing:
            print("*"*50)
            print(missing)
            raise Exception("Missing images")
        print(f"Stats for {row['old']} -> {row['new']}:")
        print(f"Efficacy: {np.mean(efficacy)} +- {np.std(efficacy)}")
        print(f"Generality: {np.mean(generality)} +- {np.std(generality)}")
        print(f"Generality_75: {np.mean(generality_75)} +- {np.std(generality_75)}")
        print(f"generality_90: {np.mean(generality_90)} +- {np.std(generality_90)}")
        print(f"Specificity: {np.mean(specificity)} +- {np.std(specificity)}")

        all_efficacy.append(efficacy)
        all_generality.append(generality)
        all_generality_75.append(generality_75)
        all_generality_90.append(generality_90)
        all_specificity.append(specificity)
        all_old.append(row['old'])
        all_new.append(row['new'])
        

    all_efficacy = np.array(all_efficacy)
    all_generality = np.array(all_generality)
    all_generality_75 = np.array(all_generality_75)
    all_generality_90 = np.array(all_generality_90)
    all_specificity = np.array(all_specificity)
    all_validity = np.array(all_validity)

    print(all_generality.mean(axis=0))

    print("Mean efficacy:", np.mean(all_efficacy))
    print("Mean generality:", np.mean(all_generality))
    print("Mean generality90:", np.mean(all_generality_90))
    print("Mean generality75:", np.mean(all_generality_75))
    print("Mean specificity:", np.mean(all_specificity))

    save_suffix = "oracle" if oracle else "emcid"
    results_dir = f"results/emcid/{hparam_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    key = f"weight{mom2_weight}" + (f"_ew{edit_weight}" if edit_weight != 0.5 else "")
    
    file_path = results_dir + f"/{dataset}_results_{save_suffix}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            old_results = json.load(f)
        if old_results:
            old_results.update(
            { key: 
                {
                "efficacy": np.mean(all_efficacy),
                "efficacy_std": np.std(all_efficacy.mean(axis=0)),
                "generality": np.mean(all_generality),
                "generality_std": np.std(all_generality.mean(axis=0)),
                "generality_90": np.mean(all_generality_90),
                "generality_90_std": np.std(all_generality_90.mean(axis=0)),
                "generality_75": np.mean(all_generality_75),
                "generality_75_std": np.std(all_generality_75.mean(axis=0)),
                "specificity": np.mean(all_specificity),
                "specificity_std": np.std(all_specificity.mean(axis=0))
            }})
            new_results = old_results
        else:
            new_results = {
            key: 
            {
            "efficacy": np.mean(all_efficacy),
            "efficacy_std": np.std(all_efficacy.mean(axis=0)),
            "generality": np.mean(all_generality),
            "generality_std": np.std(all_generality.mean(axis=0)),
            "generality_90": np.mean(all_generality_90),
            "generality_90_std": np.std(all_generality_90.mean(axis=0)),
            "generality_75": np.mean(all_generality_75),
            "generality_75_std": np.std(all_generality_75.mean(axis=0)),
            "specificity": np.mean(all_specificity),
            "specificity_std": np.std(all_specificity.mean(axis=0))
        }}
    else:
        new_results = {
            key: 
            {
            "efficacy": np.mean(all_efficacy),
            "efficacy_std": np.std(all_efficacy.mean(axis=0)),
            "generality": np.mean(all_generality),
            "generality_std": np.std(all_generality.mean(axis=0)),
            "generality_90": np.mean(all_generality_90),
            "generality_90_std": np.std(all_generality_90.mean(axis=0)),
            "generality_75": np.mean(all_generality_75),
            "generality_75_std": np.std(all_generality_75.mean(axis=0)),
            "specificity": np.mean(all_specificity),
            "specificity_std": np.std(all_specificity.mean(axis=0))
        }}

    f1_score = 2 * (new_results[key]["generality"] * \
                new_results[key]["specificity"]) \
                    / \
                (new_results[key]["generality"] + \
                    new_results[key]["specificity"])
    new_results[key]["f1_score"] = f1_score

    with open(f"results/emcid/{hparam_name}/{dataset}_results_{save_suffix}.json", "w") as f:
        json.dump(new_results, f, indent=4)

    # this mean is over the seeds
    result_dict = {
        'old': all_old,
        'new': all_new,
        'efficacy': all_efficacy.mean(axis=1),
        'generality': all_generality.mean(axis=1),
        'generality90': all_generality_90.mean(axis=1),
        'generality75': all_generality_75.mean(axis=1),
        'specificity': all_specificity.mean(axis=1),
    }

    df = pd.DataFrame.from_dict(result_dict)
    # save the results
    df.to_csv(f"results/emcid/{hparam_name}/{dataset}_results_{save_suffix}.csv", index=False)
   
    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam", type=str, default="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--dataset", type=str, default="timed")
    parser.add_argument("--seed_num", type=int, default=1)
    parser.add_argument("--mom2_weight", type=int, default=None)
    str_to_bool = lambda x: (str(x).lower() == 'true')
    parser.add_argument("--oracle", type=str_to_bool, default=False)
    parser.add_argument("--eval", type=str_to_bool, default=False)
    parser.add_argument("--edit_weight", type=float, default=0.5)

    args = parser.parse_args()
    print(args)
    print(args.hparam)

    dataset = args.dataset
    if not args.eval:
        for seed in range(0, args.seed_num):
            # set_seed(seed)
            emcid_test(
                dataset=dataset, 
                hparam_name=args.hparam, 
                global_seed=seed,
                mom2_weight=args.mom2_weight,
                edit_weight=args.edit_weight,
                generate_oracle=args.oracle,
                device=args.device)

    eval_all(
        dataset=dataset, 
        num_seeds=args.seed_num,
        hparam_name=args.hparam, 
        device=args.device, 
        mom2_weight=args.mom2_weight,
        edit_weight=args.edit_weight,
        oracle=args.oracle)
    

 