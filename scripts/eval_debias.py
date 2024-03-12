import json
from typing import List, Dict
from argparse import ArgumentParser
from statistics import mean
import subprocess

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import (
    AutoProcessor,
    ViTForImageClassification,
    CLIPModel, CLIPProcessor
)
import pandas as pd

from util.evaluate import calculate_single_cls_score
from util.globals import *
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_cross_attn
from experiments.emcid_test import set_weights

from emcid.uce_train import edit_model_uce, edit_model_uce_modified
from dsets.iceb_dataset import *
from dsets.global_concepts import get_i2p_editing_requests, NSFWEditRequestDataset
from dsets.debias_requests import DebiasRequestDataset
from emcid.emcid_main import apply_emcid_to_text_encoder_debias
from refact_benchmark_eval import set_seed
from eval_coco import generate_coco_30k, cal_clip_score_coco, cal_lpips_coco


def test_debiasing(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    device="cuda:0",
    mom2_weight=None,
    edit_weight=None,
    imgs_per_prompt=12,
    max_iters=20,
    num_seeds=10,
    eval_coco=False,
    eval_debias=True,
    mixed=False,
    pre_edit=False,
    recompute_factors=False,
    coco_sample_num=1
):

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    cache_name = f"cache/{hparam_name}/debiasing/"

    hparams = set_weights(hparams, mom2_weight, edit_weight)

    requests = DebiasRequestDataset()
    csv_path = DATA_DIR / "debias" / "TIMED_gender_test_set_processed.csv"

    # random.seed(2023)
    # global_seeds = random.sample(range(10000), num_seeds)
    global_seeds = list(range(num_seeds))
    print("global seeds: ", global_seeds)

    if pre_edit:
        # pre edit evaluation
        for seed in global_seeds:
            with torch.no_grad():
                generate_debias_eval_imgs(
                    pipe=pipe,
                    global_seed=seed,
                    imgs_per_prompt=imgs_per_prompt,
                    data_dir=f"results/images/debiasing/orig",
                    csv_path=csv_path
                )
        
        if not os.path.exists("results/sd_orig/debiasing/orig_ratios.csv"):
            eval_ratios(
                num_seeds=num_seeds,
                data_path="results/images/debiasing/orig",
                out_path="results/sd_orig/debiasing/orig_ratios.csv",
                device=device,
                global_seeds=global_seeds,
            )
        
        if eval_coco:
            # generate pre-edit images
            with torch.no_grad():
                generate_coco_30k(
                    pipe=pipe,
                    sample_num=coco_sample_num,
                    file_path=DATA_DIR / "coco" / "coco_30k.csv",
                    out_dir="data/coco/images/sd_orig"
                )

    # randomly shuffle the requests
    indices = list(range(len(requests)))
    random.shuffle(indices)
    requests = [requests[idx] for idx in indices]
    if mixed:
        pipe, orig_text_encoder = apply_emcid_to_text_encoder_debias(
            pipe=pipe,
            requests=requests,
            hparams=hparams,
            device=device,
            cache_name=cache_name,
            return_orig_text_model=True,
            recompute_factors=False,    # cannot recompute factors for mixed
            max_iter=max_iters,
            verbose=False
        )

        if eval_debias:
            print("evaluating for all requests mixed")
            for seed in global_seeds:
                with torch.no_grad():
                    generate_debias_eval_imgs(
                        pipe=pipe,
                        imgs_per_prompt=imgs_per_prompt,
                        global_seed=seed,
                        data_dir=f"results/images/debiasing/{hparam_name}/mixed",
                        csv_path=csv_path
                    )
    else:
        for request in tqdm(requests, disable=False):
            pipe, orig_text_encoder = apply_emcid_to_text_encoder_debias(
                pipe=pipe,
                requests=[request],
                hparams=hparams,
                device=device,
                cache_name=cache_name,
                return_orig_text_model=True,
                recompute_factors=recompute_factors,
                max_iter=max_iters,
                verbose=False
            )
            if eval_debias:
                print("evaluating for request: ", request["source"])
                for seed in global_seeds:
                    with torch.no_grad():
                        generate_debias_eval_imgs(
                            pipe=pipe,
                            imgs_per_prompt=imgs_per_prompt,
                            global_seed=seed,
                            source=request["source"],
                            data_dir=f"results/images/debiasing/{hparam_name}/single",
                            csv_path=csv_path
                        )
            # retore the original text encoder
            pipe.text_encoder = orig_text_encoder
    
    # evaluate the results
    if mixed:
        if eval_debias:
            eval_ratios(
                num_seeds=num_seeds,
                global_seeds=global_seeds,
                data_path=f"results/images/debiasing/{hparam_name}/mixed",
                out_path=f"results/emcid/{hparam_name}/debiasing/{hparam_name}_mixed_ratios.csv",
                device=device,
            )

        if eval_coco:
            with torch.no_grad():
                generate_coco_30k(
                    pipe=pipe,
                    sample_num=coco_sample_num,
                    file_path=DATA_DIR / "coco" / "coco_30k.csv",
                    out_dir=f"data/coco/images/{hparam_name}_debias_all"\
                            f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}"
                )
            edited_path = f"data/coco/images/{hparam_name}_debias_all"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}"
            dict_key = f'weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')

            print("Calculating LPIPS...")
            cal_lpips_coco(
                hparam_name=hparam_name,
                num_edit=len(requests),
                mom2_weight=hparams.mom2_update_weight,
                edit_weight=hparams.edit_weight,
                sample_num=coco_sample_num,
                edited_path=edited_path,
                original_path="data/coco/images/sd_orig",
                csv_path=DATA_DIR / "coco" / "coco_30k.csv",
                device=device
            )

            print("Calculating CLIP score...")
            cal_clip_score_coco(
                hparam_name=hparam_name,
                num_edit=len(requests),
                mom2_weight=hparams.mom2_update_weight,
                edit_weight=hparams.edit_weight,
                sample_num=coco_sample_num,
                edited_path=edited_path,
                csv_path=DATA_DIR / "coco" / "coco_30k.csv",
                device=device
            )

            print("Calculating FID...")
            subprocess.run(["python", 
                            "scripts/test_fid_score.py", 
                            f"--generated_images_folder={edited_path}", 
                            "--coco_30k_folder=data/coco/coco-30k", 
                            f"--output_folder=results/emcid/{hparam_name}/artists", 
                            "--output_file=coco_summary.json",
                            "--batch_size=32", 
                            f"--save_npz_folder=data/stats/fid/{hparam_name}", 
                            f"--device={device}", 
                            f"--dict_key={dict_key}"])

    
    else:
        if eval_debias:
            eval_ratios(
                num_seeds=num_seeds,
                global_seeds=global_seeds,
                data_path=f"results/images/debiasing/{hparam_name}/single",
                out_path=f"results/emcid/{hparam_name}/debiasing/{hparam_name}_ratios.csv",
                device=device,
            )
        
        if eval_coco:
            print("coco evaluation for single request is not very meaningful")
    



def generate_debias_eval_imgs(
    pipe: StableDiffusionPipeline,
    global_seed: int,
    data_dir: str,
    imgs_per_prompt: int=24,
    source: str=None,
    csv_path=DATA_DIR / "debias" / "TIMED_gender_test_set_processed.csv",
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    with open(csv_path, "r") as file:
        data = pd.read_csv(file)
    
    prompt_cols = ["validation", *[f"ex{i}" for i in range(1, 6)]]
    print("generating for seed: ", global_seed)
    print("generating for source: ", source)
    if source is None:
        sub_data = data
    else:
        sub_data = data[data["old"] == source]

    for idx, row in tqdm(sub_data.iterrows(), total=len(sub_data), disable=source is not None):
        # for reproducibility
        imgs_dir = f"{data_dir}/{row['old']}"
        # fix seed for each row
        set_seed(global_seed)
        for prompt_col in prompt_cols:
            prompt = row[prompt_col]
            save_dir = f"{imgs_dir}/{prompt}/seed{global_seed}"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(imgs_per_prompt):
                if os.path.exists(f"{save_dir}/idx_{i}.png"):
                    continue
                img = pipe([prompt], guidance_scale=7.5).images[0]
                img.save(f"{save_dir}/idx_{i}.png")


def eval_ratios(
    data_path,
    out_path,
    global_seeds,
    num_seeds=10,
    device="cuda:0",
    csv_path=DATA_DIR / "debias" / "TIMED_gender_test_set_processed.csv",
):
    with open(csv_path, "r") as file:
        data = pd.read_csv(file)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) 
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    prompt_cols = ["validation", *[f"ex{i}" for i in range(1, 6)]]

    results_per_seed = []
    for seed in global_seeds:
        results = {}
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            dests = [row["female"], row["male"]]
            cnts = [0, 0]
            save_dir = f"{data_path}/{row['old']}"
            for prompt_col in prompt_cols:
                prompt = row[prompt_col]
                imgs_dir = f"{save_dir}/{prompt}/seed{seed}"
                for img_name in os.listdir(imgs_dir):
                    img = Image.open(f"{imgs_dir}/{img_name}")
                    try:
                        inputs = processor(
                                    text=dests, 
                                    images=[img], 
                                    return_tensors="pt", 
                                    padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = clip_model(**inputs)

                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=-1)
                        max_idx = probs.argmax(dim=-1).item()
                        cnts[max_idx] += 1
                    except OSError:
                        # remove the corrupted image
                        os.remove(f"{imgs_dir}/{img_name}")

            ratios = [cnts[0] / sum(cnts), cnts[1] / sum(cnts)]
            # print(f"seed {seed}, ratios {ratios}")
            # print(f"delta {abs(ratios[0] - 0.5) / 0.5}")

            results[row["old"]] = {"female": ratios[0], 
                                "male": ratios[1], 
                                "delta": abs(ratios[0] - 0.5) / 0.5}
            results_per_seed.append(results)

    final_results = {}
    for key in results_per_seed[0].keys():
        final_results[key] = {}
        female_ratios = np.mean([results[key]["female"] for results in results_per_seed])
        male_ratios = np.mean([results[key]["male"] for results in results_per_seed])
        delta = np.mean([results[key]["delta"] for results in results_per_seed])
        delta_std = np.std([results[key]["delta"] for results in results_per_seed])

        final_results[key].update(
            {
                "female": female_ratios,
                "male": male_ratios,
                "delta": delta,
                "delta_std": delta_std,
            }
        )
    
    # calculate the total mean and std of delta
    deltas = []
    for results in results_per_seed:
        deltas.append(
            np.mean([results[key]["delta"] for key in results.keys()])
        )
    
    print("total mean delta: ", np.mean(deltas))
    print("total std delta: ", np.std(deltas))

    final_results["total"] = {
        "delta": np.mean(deltas),
        "delta_std": np.std(deltas),
    }

    # change results to df, columns being "female", "male", "delta"
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    df = pd.DataFrame.from_dict(final_results, orient="index")
    df.to_csv(out_path)


def eval_ratios_0(
    data_path,
    out_path,
    num_seeds=10,
    device="cuda:0",
    csv_path=DATA_DIR / "debias" / "TIMED_gender_test_set_processed.csv",
):
    with open(csv_path, "r") as file:
        data = pd.read_csv(file)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) 
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    prompt_cols = ["validation", *[f"ex{i}" for i in range(1, 6)]]

    results_per_seed = []
    for seed in range(num_seeds):
        results = {}
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            dests = [row["female"], row["male"]]
            cnts = [0, 0]
            save_dir = f"{data_path}/{row['old']}"
            for prompt_col in prompt_cols:
                prompt = row[prompt_col]
                imgs_dir = f"{save_dir}/{prompt}"
                for img_name in os.listdir(imgs_dir):
                    if img_name != f"{seed}.png":
                        continue
                    img = Image.open(f"{imgs_dir}/{img_name}")
                    inputs = processor(
                                text=dests, 
                                images=[img], 
                                return_tensors="pt", 
                                padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = clip_model(**inputs)

                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)
                    max_idx = probs.argmax(dim=-1).item()
                    cnts[max_idx] += 1
            ratios = [cnts[0] / sum(cnts), cnts[1] / sum(cnts)]

            results[row["old"]] = {"female": ratios[0], 
                                "male": ratios[1], 
                                "delta": abs(ratios[0] - 0.5) / 0.5}
            
        results_per_seed.append(results)

    final_results = {}
    for key in results_per_seed[0].keys():
        final_results[key] = {}
        female_ratios = np.mean([results[key]["female"] for results in results_per_seed])
        male_ratios = np.mean([results[key]["male"] for results in results_per_seed])
        delta = np.mean([results[key]["delta"] for results in results_per_seed])
        delta_std = np.std([results[key]["delta"] for results in results_per_seed])

        final_results[key].update(
            {
                "female": female_ratios,
                "male": male_ratios,
                "delta": delta,
                "delta_std": delta_std,
            }
        )
    
    # calculate the total mean and std of delta
    deltas = []
    for results in results_per_seed:
        deltas.append(
            np.mean([results[key]["delta"] for key in results.keys()])
        )
    
    print("total mean delta: ", np.mean(deltas))
    print("total std delta: ", np.std(deltas))

    final_results["total"] = {
        "delta": np.mean(deltas),
        "delta_std": np.std(deltas),
    }

    # change results to df, columns being "female", "male", "delta"
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    df = pd.DataFrame.from_dict(final_results, orient="index")
    df.to_csv(out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hparam", type=str, default="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed_num", type=int, default=10)
    parser.add_argument("--mom2_weight", type=int, default=None)
    str_to_bool = lambda x: (str(x).lower() == 'true')
    parser.add_argument("--edit_weight", type=float, default=0.5)
    parser.add_argument("--mixed", action="store_true", default=False)
    parser.add_argument("--pre_edit", action="store_true", default=False)
    parser.add_argument("--recompute_factors", action="store_true", default=False)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--coco", action="store_true", default=False)
    parser.add_argument("--no_debias_eval", action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    print(args.hparam)

    test_debiasing(
        hparam_name=args.hparam,
        device=args.device,
        mom2_weight=args.mom2_weight,
        edit_weight=args.edit_weight,
        num_seeds=args.seed_num,
        mixed=args.mixed,
        recompute_factors=args.recompute_factors,
        pre_edit=args.pre_edit,
        max_iters=args.max_iters,
        eval_coco=args.coco,
        eval_debias=not args.no_debias_eval
    )
    

