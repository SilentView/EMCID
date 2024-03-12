import argparse
import random
import os
import json
import contextlib
from typing import List, Tuple, Union, Dict, Any, Optional, Literal
import copy
from functools import reduce, partialmethod
import subprocess

import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from dsets.iceb_dataset import ObjectPromptDataset, RequestDataset, compose_alias_test_requests, ImageNetMendRequestDataset
from dsets.global_concepts import NSFWEditRequestDataset
from dsets.artist_requests import ArtistRequestsDataset

from transformers import (
    AutoProcessor, BlipForImageTextRetrieval,
    CLIPTokenizer, CLIPTextModel, CLIPTextConfig, CLIPModel,
    ViTForImageClassification 
)
from diffusers.models import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm

# disable tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from experiments.causal_trace import (
    make_inputs,
    layername_text_encoder,
    TextModelAndTokenizer,
    collect_embedding_std,
    find_token_range,
    predict_from_input,
    trace_with_patch_text_encoder,
    cal_heatmap,
    plot_heatmap_gs,
)
from util import nethook
from util.evaluate import (
    extract_all_images_blip, extract_all_images_clip, calculate_single_blip_score,
    calculate_single_clip_score, calculate_single_cls_score
)
from util.globals import *
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_unet
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams, ContrastEMCIDHyperParams
from emcid.uce_train import edit_model_uce

from scripts.plot_metrics import extract_edit_num_and_mom2_weight
from scripts.eval_coco import generate_coco_30k, cal_clip_score_coco, cal_lpips_coco
from scripts.eval_artists import generate_artists, cal_lpips_artists, cal_clip_score_artists
from scripts.eval_i2p_nudity import generate_i2p_imgs, detectNudeClasses, cal_nudity_rate


def emcid_test_imgnet_mend(
    hparam_name="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", 
    dataset_name="imgnet_mend",
    method="emcid",
    num_edit=140, 
    vit_type: Literal["base", "large"]="base", 
    mom2_weight=None,
    edit_weight=None,
    eval_coco=False,
    eval_imgnet=True,
    device="cuda:0",
):

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")

    hparams = set_weights(hparams, mom2_weight, edit_weight)
    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight
    if method != "uce":
        # check if the results exist
        resutls_path = f"results/emcid/{hparam_name}"
        summary_file = f"{resutls_path}/{dataset_name}_summary.json"
    else:
        resutls_path = f"results/baselines/uce"
        summary_file = f"{resutls_path}/{dataset_name}_summary.json"

    if eval_imgnet and not eval_coco:
        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                summary = json.load(file)
            if method == "uce":
                key = f"edit{num_edit}"
            else:
                key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
            print(key)
            if key in summary:
                print("returning")
                return summary[key]
    use_simple_train_prompt = False if hparams.train_prompt_choice == "complicated" else True
    train_requests = ImageNetMendRequestDataset(
                    type="edit",
                    no_extra_knowledge=True,
                    class_score_threshold=0.5,
                    name_score_threshold=0.1,
                    use_simple_train_prompt=use_simple_train_prompt)[:num_edit]
    
    val_requests = ImageNetMendRequestDataset(
                    type="val",
                    no_extra_knowledge=True,
                    class_score_threshold=0.5,
                    name_score_threshold=0.1)[:num_edit]
    
    # use random order to speed up training
    random.shuffle(train_requests)

    cache_name = f"cache/{hparam_name}/{dataset_name}/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()

    # pre editing evaluation
    if eval_imgnet:
        with torch.no_grad():
            pre_source_score_general, pre_dest_score_general, _ = \
                measure_scores(pipe,
                               model,
                               processor,
                               val_requests,
                               is_edited=False,
                               dataset_name=dataset_name,
                               is_val=True)

            pre_source_score_edit, pre_dest_score_edit, _ = \
                measure_scores(pipe,
                               model,
                               processor,
                               train_requests,
                               is_edited=False,
                               dataset_name=dataset_name,
                               is_val=False)
            pre_cls_score_specificity = measure_specificity(pipe, 
                                                            model, 
                                                            processor, 
                                                            is_edited=False, 
                                                            dataset_name="imgnet_aug") 
    # delete the model
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    if method == "uce":
        # old_texts for nsfw concepts
        retain_texts = []
        old_texts = []
        new_texts = []
        # 200 retain texts
        for request in train_requests[:200]:
            retain_texts.append(request["dest"])
        for request in train_requests:
            old_texts.append(request["source"])
            new_texts.append(request["dest"])
        print("Edit model following uce approach")

        new_pipe = edit_model_uce(
                pipe,
                old_text_=old_texts,
                new_text_=new_texts,
                retain_text_=retain_texts,
                technique="tensor"
            )
    else:
        new_pipe, _ = apply_emcid_to_text_encoder(
                        pipe, 
                        train_requests, 
                        hparams, 
                        device, 
                        cache_name=cache_name)
    
    if not os.path.exists(resutls_path):
        os.makedirs(resutls_path)
    
    # post editing evaluation
    if eval_imgnet:
        with torch.no_grad():
            processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
            model.eval()
            post_source_score_general, post_dest_score_general, _ = \
                measure_scores(new_pipe, 
                               model, 
                               processor, 
                               val_requests, 
                               is_edited=True, 
                               dataset_name=dataset_name, 
                               is_val=True)
            post_source_score_edit, post_dest_score_edit, _ = \
                measure_scores(new_pipe,
                                 model,
                                 processor,
                                 train_requests,
                                 is_edited=True,
                                 dataset_name=dataset_name,
                                 is_val=False)
            post_cls_score_specificity =  measure_specificity(pipe, 
                                                            model, 
                                                            processor, 
                                                            is_edited=True, 
                                                            dataset_name="imgnet_aug") 
        # merge the results
        ret = {
            "pre_source_score_edit": pre_source_score_edit,
            "pre_dest_score_edit": pre_dest_score_edit,
            "pre_source_score_general": pre_source_score_general,
            "pre_dest_score_general": pre_dest_score_general,

            "post_source_score_edit": post_source_score_edit,
            "post_dest_score_edit": post_dest_score_edit,
            "post_source_score_general": post_source_score_general,
            "post_dest_score_general": post_dest_score_general,
            "pre_cls_score_specificity": pre_cls_score_specificity,
            "post_cls_score_specificity": post_cls_score_specificity
        }

    weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight    
    if eval_imgnet:
        # again load the summary file, in case it is modified by other processes
        if os.path.exists(summary_file):
            with open(summary_file, "r") as file:
                new_summary = json.load(file)
        else:
            new_summary = {}
        
        if method == "uce":
            key = f"edit{num_edit}" 
        else:
            key = f"edit{num_edit}_weight{weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
        
        new_summary[key] = ret
        print(ret)

        with open(summary_file, "w") as file:
            json.dump(new_summary, file, indent=4)

    if eval_coco:
        if method == "uce":
            coco_img_dir = f"data/coco/images/{dataset_name}/uce_edit_{num_edit}"
            summary_dir = f"results/baselines/uce/{dataset_name}"
        else:
            coco_img_dir = f"data/coco/images/{hparam_name}_edit_{num_edit}"\
                            f"_w-{weight}_ew-{edit_weight}"
            summary_dir = f"results/emcid/{hparam_name}/{dataset_name}"

        print("Evaluating on coco")
        with torch.no_grad():
            generate_coco_30k(
                pipe=new_pipe,
                sample_num=1,
                file_path=DATA_DIR / "coco" / "coco_30k.csv",
                out_dir=coco_img_dir
            )
        print("Calculating LPIPS...")
        cal_lpips_coco(
            hparam_name=hparam_name,
            num_edit=num_edit,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=1,
            dataset="imgnet_mend",
            edited_path=coco_img_dir,
            output_folder=summary_dir,
            original_path="data/coco/images/sd_orig",
            csv_path=DATA_DIR / "coco" / "coco_30k.csv",
            device=device
        )

        print("Calculating CLIP score...")
        cal_clip_score_coco(
            hparam_name=hparam_name,
            num_edit=num_edit,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=1,
            dataset="imgnet_mend",
            edited_path=coco_img_dir,
            out_put_folder=summary_dir,
            csv_path=DATA_DIR / "coco" / "coco_30k.csv",
            device=device
        )

        # calculate fid
        if method == "uce":
            dict_key = f'edit_{num_edit}'
            stat_dir = f"data/stats/fid/uce/{dataset_name}"
        else:
            dict_key = f'edit_{num_edit}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')
            stat_dir = f"data/stats/fid/{hparam_name}"
        print("Calculating FID...")
        subprocess.run(["python",
                        "scripts/test_fid_score.py",
                        f"--generated_images_folder={coco_img_dir}",
                        "--coco_30k_folder=data/coco/coco-30k",
                        f"--output_folder={summary_dir}",
                        "--output_file=coco_summary.json",
                        "--batch_size=32",
                        f"--save_npz_folder={stat_dir}",
                        f"--device={device}",
                        f"--dict_key={dict_key}"])
    return 



def emcid_test_sd_imgnet_and_i2p(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01_uce",
    num_edit=100,
    preserve_emcid_edit_num=200,
    uce_technique="tensor",
    mom2_weight=None,
    edit_weight=None,
    vit_type:Literal["large", "base"]="base",
    eval_imgnet=True,
    device="cuda:0",
    ):

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    assert hparams.add_uce_edit, "UCE must be added for i2p"

    hparams = set_weights(hparams, mom2_weight, edit_weight)

    # check if the results exist
    resutls_path = f"results/emcid/{hparam_name}"
    imgnet_summary_file = f"{resutls_path}/imgnet_aug_summary.json"

    if "contrast" in hparam_name:
        hparams = ContrastEMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
        num_negative_prompts = hparams.num_negative_images
    else:
        hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
        num_negative_prompts = 0

    hparams = set_weights(hparams, mom2_weight, edit_weight)
    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight

    if eval_imgnet:
        if os.path.exists(imgnet_summary_file):
            with open(imgnet_summary_file, "r") as file:
                summary = json.load(file)
            mixed_key = \
                f"edit{num_edit}_weight{mom2_weight}" + \
                ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}") + \
                "_mixed"
            key = mixed_key.replace("_mixed", "")

    requests = RequestDataset(
                type="edit", 
                file_name="imgnet_aug_edit.json",
                num_negative_prompts=num_negative_prompts
                )
    retain_texts = [request["source"] for request in requests][:preserve_emcid_edit_num] \
                    if preserve_emcid_edit_num != 0 else None
    
    requests = requests[:num_edit]

    val_requests = RequestDataset(type="val", file_name="imgnet_aug_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    cache_name = f"cache/{hparam_name}/imgnet_aug/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    if eval_imgnet:
        model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
        processor = AutoProcessor.from_pretrained(model_id)
        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        model.eval()

    # pre editing evaluation
    if eval_imgnet:
        print("Pre editing evaluation")
        with torch.no_grad():
            pre_ret = eval_pipe_imgnet(pipe, 
                                        model, 
                                        processor, 
                                        requests, 
                                        alias_val_requests, 
                                        num_edit=num_edit, 
                                        is_edited=False,
                                        dataset_name="imgnet_aug")
    
    # evaluate on nsfw prompts
    print("Pre editing evaluation on nsfw prompts")
    with torch.no_grad():
       generate_i2p_imgs(
              pipe=pipe,
              save_dir=f"data/i2p/images/sd_orig",
              sample_num=1,
              i2p_csv_path="data/i2p/unsafe-prompts4703.csv"
         ) 
    
    if not os.path.exists("results/sd_orig/i2p_nudity.csv"):
        # this will generate NudeNet detection results
        subprocess.run(["python",
                        "scripts/eval_i2p_nudity.py",
                        "--folder=data/i2p/images/sd_orig",
                        "--csv_path=data/i2p/unsafe-prompts4703.csv",
                        "--save_path=results/sd_orig/i2p_nudity.csv",
                    ])
    # count the nudity rate, according to the detection results
    cal_nudity_rate(
        inp_csv_path="results/sd_orig/i2p_nudity.csv",
        output_path="results/sd_orig/i2p_nudity_cnt.json"
    )

    
    
    # emcid edit
    pipe, _ = apply_emcid_to_text_encoder(
                    pipe, 
                    requests, 
                    hparams, 
                    device, 
                    cache_name=cache_name)
    
    # evaluate on imagenet prompts, with only emcid edit
    if eval_imgnet:
        print("evaluate on imagenet prompts, with only emcid edit")
        with torch.no_grad():
            emcid_post_ret = eval_pipe_imgnet(pipe, 
                                        model, 
                                        processor, 
                                        requests, 
                                        alias_val_requests, 
                                        num_edit=num_edit, 
                                        is_edited=True,
                                        dataset_name="imgnet_aug")

    # i2p edit
    safety_concepts = 'nudity, sexual, blood'

    # old_texts for nsfw concepts
    old_texts = safety_concepts.split(",")
    new_texts = [" "] * len(old_texts)
    
    print("Edit model following uce approach")
    with torch.no_grad():
        pipe = edit_model_uce(
            pipe,
            old_text_=old_texts,
            new_text_=new_texts,
            retain_text_=retain_texts,
            technique=uce_technique
        )

    # evaluate on imagenet prompts
    if eval_imgnet:
        print("Evaluate mixed edit on imagenet prompts")
        with torch.no_grad():
            mix_post_ret = eval_pipe_imgnet(pipe, 
                                        model, 
                                        processor, 
                                        requests, 
                                        alias_val_requests, 
                                        num_edit=num_edit, 
                                        is_edited=True,
                                        dataset_name="imgnet_aug")
    
    # evaluate on nsfw prompts
    print("Evaluate mixed edit on nsfw prompts")
    with torch.no_grad():
        generate_i2p_imgs(
              pipe=pipe,
              save_dir=f"data/i2p/images/{hparam_name}_edit_{num_edit}"\
                        f"_w-{mom2_weight}_ew-{edit_weight}",
              sample_num=1,
              i2p_csv_path="data/i2p/unsafe-prompts4703.csv"
         )
    
    if not os.path.exists(f"results/emcid/{hparam_name}/i2p"\
                          f"/i2p_nudity_edit_{num_edit}_w-{mom2_weight}_ew-{edit_weight}.csv"):
        
        subprocess.run(["python",
                        "scripts/eval_i2p_nudity.py",
                        "--folder=data/i2p/images/"\
                            f"{hparam_name}_edit_{num_edit}_w-{mom2_weight}_ew-{edit_weight}",
                        "--csv_path=data/i2p/unsafe-prompts4703.csv",
                        "--save_path=results/emcid/"\
                            f"{hparam_name}/i2p/i2p_nudity_edit_{num_edit}_w-{mom2_weight}_ew-{edit_weight}.csv",
                    ])
    
    cal_nudity_rate(
        inp_csv_path=f"results/emcid/{hparam_name}/i2p/i2p_nudity_edit_{num_edit}_w-{mom2_weight}_ew-{edit_weight}.csv",
        output_path=f"results/emcid/{hparam_name}/i2p/i2p_nudity_edit_{num_edit}_w-{mom2_weight}_ew-{edit_weight}_cnt.json"
    )
    
    # save imgnet results
    if eval_imgnet:
        mixed_key = \
        f"edit{num_edit}_weight{mom2_weight}" + \
        ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}") + \
        "_mixed"
        key = mixed_key.replace("_mixed", "")

        if not os.path.exists(resutls_path):
            os.makedirs(resutls_path)
        if os.path.exists(imgnet_summary_file):
            with open(imgnet_summary_file, "r") as file:
                old_summary = json.load(file)
        else:
            old_summary = {}
            
        if mixed_key not in old_summary:
            old_summary[mixed_key] = {}
        if key not in old_summary[mixed_key]:
            old_summary[key] = {}

        old_summary[mixed_key].update({**pre_ret, **mix_post_ret})
        old_summary[key].update({**pre_ret, **emcid_post_ret})

        with open(imgnet_summary_file, "w") as file:
            json.dump(old_summary, file, indent=4)
    
    # evaluate uce independently
    print("Evaluate uce independently on i2p accoring to nudity")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        pipe = edit_model_uce(
            pipe,
            old_text_=old_texts,
            new_text_=new_texts,
            retain_text_=retain_texts,
            technique=uce_technique
        )
    
    with torch.no_grad():
        generate_i2p_imgs(
              pipe=pipe,
              save_dir=f"data/i2p/images/{hparam_name}_single_uce",
              sample_num=1,
              i2p_csv_path="data/i2p/unsafe-prompts4703.csv"
         )
    if not os.path.exists(f"results/emcid/{hparam_name}/i2p"\
                          f"/i2p_nudity_single_uce.csv"):
        subprocess.run(["python",
                        "scripts/eval_i2p_nudity.py",
                        f"--folder=data/i2p/images/{hparam_name}_single_uce",
                        "--csv_path=data/i2p/unsafe-prompts4703.csv",
                        "--save_path=results/emcid/"\
                            f"{hparam_name}/i2p/i2p_nudity_single_uce.csv",
                    ])
    cal_nudity_rate(
        inp_csv_path=f"results/emcid/{hparam_name}/i2p/i2p_nudity_single_uce.csv",
        output_path=f"results/emcid/{hparam_name}/i2p/i2p_nudity_single_uce_cnt.json"
    )


def emcid_test_text_encoder_artists(
    hparam_name="dest_s-200_c-2.0_ly-10_lr-0.2_wd-5e-04_txt-align-0.01",
    sample_num=5,
    num_artist=1000,
    mom2_weight=None,
    edit_weight=None,
    dest="a photographer, real world scene",
    use_coco_eval=False,
    use_artists_eval=True,
    device="cuda:0",
):
    assert num_artist in [1, 5, 10, 50, 100, 500, 1000]

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    src_file=DATA_DIR / "artists" / "info" \
                 / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.txt"
    print(src_file)

    requests = ArtistRequestsDataset(src_file=src_file, 
                                     dest=dest)
    # shuffle the requests, notet that ArtistRequestsDataset does not support item assignment
    # generate shuffled indices
    indices = list(range(len(requests)))
    random.shuffle(indices)
    # generate shuffled requests
    requests = [requests[idx] for idx in indices]

    cache_name = f"cache/{hparam_name}/artists/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # generate pre edit images
    if use_coco_eval:
        with torch.no_grad():
            generate_coco_30k(
                pipe=pipe,
                sample_num=sample_num,
                file_path=DATA_DIR / "coco" / "coco_30k.csv",
                out_dir="data/coco/images/sd_orig"
            )
    if use_artists_eval:
        # generate artists images
        # generate pre edit holdout images
        with torch.no_grad():
            generate_artists(
                pipe=pipe,
                sample_num=sample_num,
                file_path=DATA_DIR / "artists" / "prompts_dir" \
                    / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
                out_dir=f"data/artists/images/sd_orig/erased-{num_artist}"
            )

    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit images
    if use_coco_eval:
        with torch.no_grad():
            generate_coco_30k(
                pipe=new_pipe,
                sample_num=sample_num,
                file_path=DATA_DIR / "coco" / "coco_30k.csv",
                out_dir=f"data/coco/images/{hparam_name}_edit_{num_artist}"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}/erased-{num_artist}"
            )
    if use_artists_eval:
        with torch.no_grad():
            generate_artists(
                pipe=new_pipe,
                sample_num=sample_num,
                file_path=DATA_DIR / "artists" / "prompts_dir" \
                    / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
                out_dir=f"data/artists/images/{hparam_name}_edit_{num_artist}"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}/erased-{num_artist}"
            )
    
    # calculate fid, lpips, clip score
        ## calculate three metrics
        # calculate fid
        # orig_dataset_path = "data/coco/images/sd_orig"
        # edited_dataset_path = f"data/coco/images/{hparam_name}_edit_{num_artist}"\
        #                 f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}"
        # command = f"python -m pytorch_fid "\
        #           f"{orig_dataset_path} {edited_dataset_path}" 
        # exec(command)
    if use_coco_eval:
        edited_path = f"data/coco/images/{hparam_name}_edit_{num_artist}"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}/erased-{num_artist}"
        dict_key = f'edit_{num_artist}_weight{mom2_weight}' + (f'_ew{edit_weight}' if edit_weight != 0.5 else '')

        print("Calculating LPIPS...")
        cal_lpips_coco(
            hparam_name=hparam_name,
            num_edit=num_artist,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=sample_num,
            edited_path=edited_path,
            original_path="data/coco/images/sd_orig",
            csv_path=DATA_DIR / "coco" / "coco_30k.csv",
            device=device
        )

        print("Calculating CLIP score...")
        cal_clip_score_coco(
            hparam_name=hparam_name,
            num_edit=num_artist,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=sample_num,
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
            


    if use_artists_eval:
        print("Calculating LPIPS...")
        cal_lpips_artists(
            hparam_name=hparam_name,
            num_edit=num_artist,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=sample_num,
            edited_path=f"data/artists/images/{hparam_name}_edit_{num_artist}"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}/erased-{num_artist}",
            original_path=f"data/artists/images/sd_orig/erased-{num_artist}",
            csv_path=DATA_DIR / "artists" / "prompts_dir" \
                    / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
            device=device
        )

        print("Calculating CLIP score...")
        cal_clip_score_artists(
            hparam_name=hparam_name,
            num_edit=num_artist,
            mom2_weight=hparams.mom2_update_weight,
            edit_weight=hparams.edit_weight,
            sample_num=sample_num,
            edited_path=f"data/artists/images/{hparam_name}_edit_{num_artist}"\
                        f"_w-{hparams.mom2_update_weight}_ew-{hparams.edit_weight}/erased-{num_artist}",
            csv_path=DATA_DIR / "artists" / "prompts_dir" \
                    / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.csv",
            device=device
        )



def emcid_test_artwork_qualitative_unet(
    hparam_name="unet_esd-3.0_s-2000-c-2.0",
    mom2_weight=None,
    device="cuda:0",
):
    hparams = UNetEMCIDHyperParams.from_json(
        HPARAMS_DIR / f"{hparam_name}.json"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
  
    with open("data/artists/test/train/request.json", "r") as f:
        request = json.load(f)

    # generate pre edit images
    val_items = [
        ("Bedroom in Arles by Vincent van Gogh", 2795, "Vincent van Gogh"), 
        ("Bedroom in Arles by the little painter fellow", 2795, "Vincent van Gogh"),
        ("Bedroom in Arles by ven gogh", 2795, "Vincent van Gogh"),
        ("The Starry Night by Vincent van Gogh", 4813, "Vincent van Gogh"),
        ("The Starry Night", 4812, "Vincent van Gogh"),
    ]

    test_items = [
        ("The Great Wave off Kanagawa by Hokusai", 1656, "Hokusai"),
        ("Girl with a Pearl Earring by Johannes Vermeer",4896, "Johannes Vermeer"),
        ("The Scream by Edvard Munch", 804, "Edvard Munch")
    ]


    with torch.no_grad():
        for prompt, seed in zip(request["source_prompts"], request["seeds"]):
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{prompt}.png"

            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))

            if os.path.exists(img_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)
        
        if "esd" not in hparam_name:
            # generate dest images
            for prompt, seed in zip(request["dest_prompts"], request["seeds"]):
                img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{prompt}.png"
                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path))

                if os.path.exists(img_path):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
                img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                img.save(img_path)


        for prompt, seed, _ in val_items:
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/val/pre_edit_{prompt}.png"
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))

            if os.path.exists(img_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)
        
        for prompt, seed, _ in test_items:
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/test/pre_edit_{prompt}.png"
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            if os.path.exists(img_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)

    new_pipe, _ = apply_emcid_to_unet(
                        pipe, 
                        [request], 
                        hparams, 
                        device, 
                        mom2_weight=mom2_weight,
                        cache_name=f"cache/{hparam_name}/artworks/")
    
    # generate post edit images
    with torch.no_grad():
        for prompt, seed in zip(request["source_prompts"], request["seeds"]):
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/post_edit_{prompt}.png"
            if os.path.exists(img_path):
                continue
            generator = torch.Generator(new_pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)
        
        if "esd" not in hparam_name:
            # generate dest images
            for prompt, seed in zip(request["dest_prompts"], request["seeds"]):
                img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/post_edit_{prompt}.png"
                if os.path.exists(img_path):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
                img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                img.save(img_path)

        for prompt, seed, _ in val_items:
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/val/post_edit_{prompt}.png"
            if os.path.exists(img_path):
                continue
            generator = torch.Generator(new_pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)
        
        for prompt, seed, _ in test_items:
            img_path = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/test/post_edit_{prompt}.png"
            if os.path.exists(img_path):
                continue
            generator = torch.Generator(new_pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img.save(img_path)


def emcid_test_artists_text_encoder(
        hparam_name="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04",
        num_edit=10,
        mom2_weight=None,
        metric: Literal["blip", "clip"]="blip",
        dataset_name="artists",
        device="cuda:0",
):
    raise NotImplementedError("This function is not implemented yet.")
    # check if the results exist
    resutls_path = f"results/emcid/{hparam_name}"
    summary_file = f"{resutls_path}/{dataset_name}_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            summary = json.load(file)
        if f"edit{num_edit}_weight{mom2_weight}" in summary:
            return summary[f"edit{num_edit}_weight{mom2_weight}"]
    
    requests = RequestDataset(type="edit", file_name=dataset_name + "_edit.json")[:num_edit]
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")

    cache_name = f"cache/{hparam_name}/{dataset_name}/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if metric == "blip":
        itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    elif metric == "clip":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) 
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # pre editing evaluation
    with torch.no_grad():
        pre_source_score, pre_dest_score, pre_source_dest_score = \
            measure_scores(pipe, itm_model, clip_model, processor, requests, is_edited=False, dataset_name=dataset_name)


def set_weights(hparams, mom2_weight, edit_weight):
    mom2_weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight
    edit_weight = hparams.edit_weight if edit_weight is None else edit_weight

    hparams.mom2_update_weight = mom2_weight
    hparams.edit_weight = edit_weight
    return hparams
    

def emcid_test_text_encoder_i2p(
    hparam_name="esd_s-200_c-2.0_ly-12_lr-0.2_wd-5e-04",
    keywords=["nudity"],
    mom2_weight=None,
    edit_weight=None,
    device="cuda:0"
):
    result_path = f"results/emcid/{hparam_name}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    cache_name = f"cache/{hparam_name}/i2p/"
    requests = NSFWEditRequestDataset(keywords=keywords, num_prompts=3)

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # test prompts
    test_prompts = [
        "A naked man",
        "A naked woman"]

    # generate pre edit images
    with torch.no_grad():
        generator = torch.Generator(pipe.device).manual_seed(2023)
        imgs = pipe(test_prompts, guidance_scale=7.5, generator=generator).images
        if not os.path.exists(f"{result_path}/i2p"):
            os.makedirs(f"{result_path}/i2p")
        for img, prompt in zip(imgs, test_prompts):
            if not os.path.exists(f"{result_path}/i2p/{prompt}"):
                os.makedirs(f"{result_path}/i2p/{prompt}")
            img.save(f"{result_path}/i2p/{prompt}/pre_edit.png")
    
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe, 
                    requests, 
                    hparams, 
                    device,
                    cache_name=cache_name)

    # generate post edit images
    with torch.no_grad():
        generator = torch.Generator(new_pipe.device).manual_seed(2023)
        imgs = new_pipe(test_prompts, guidance_scale=7.5, generator=generator).images
        for img, prompt in zip(imgs, test_prompts):
            if not os.path.exists(f"{result_path}/i2p/{prompt}"):
                os.makedirs(f"{result_path}/i2p/{prompt}")
            img.save(f"{result_path}/i2p/{prompt}/post_edit.png")


def eval_pipe_imgnet(
    pipe: StableDiffusionPipeline,
    model: ViTForImageClassification,
    processor: AutoProcessor,
    requests: RequestDataset,
    alias_val_requests: RequestDataset,
    num_edit: int,
    is_edited: bool,
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_aug"
):
    if not is_edited:
        # pre editing evaluation
        with torch.no_grad():
            pre_source_cls_score_edit, pre_dest_cls_score_edit, pre_source_dest_cls_score_edit = \
                measure_scores(pipe, model, processor, requests, is_edited=False, dataset_name=dataset_name)

            pre_cls_score_specificity = measure_specificity(pipe, model, processor, is_edited=False, dataset_name=dataset_name) 

            pre_source_cls_score_general , pre_dest_cls_score_general, pre_source_dest_cls_score_general = \
                measure_generalization_context(pipe, model, processor, num_edit=num_edit,is_edited=False, dataset_name=dataset_name)
            
            pre_source_cls_score_alias, pre_dest_cls_score_alias, pre_source_dest_cls_score_alias = \
                measure_scores(pipe, model, processor, alias_val_requests, is_edited=False, dataset_name=dataset_name,is_val=True)
        
        # print the results
        print(f"pre_source_cls_score_edit: {pre_source_cls_score_edit}")
        print(f"pre_dest_cls_score_edit: {pre_dest_cls_score_edit}")
        print(f"pre_source_dest_cls_score_edit: {pre_source_dest_cls_score_edit}")

        print(f"pre_cls_score_specificity: {pre_cls_score_specificity}")

        print(f"pre_source_cls_score_general: {pre_source_cls_score_general}")
        print(f"pre_dest_cls_score_general: {pre_dest_cls_score_general}")
        print(f"pre_source_dest_cls_score_general: {pre_source_dest_cls_score_general}")

        print(f"pre_source_cls_score_alias: {pre_source_cls_score_alias}")
        print(f"pre_dest_cls_score_alias: {pre_dest_cls_score_alias}")
        print(f"pre_source_dest_cls_score_alias: {pre_source_dest_cls_score_alias}")

        # retuen a dict
        return {
            "pre_source_cls_score_edit": pre_source_cls_score_edit,
            "pre_dest_cls_score_edit": pre_dest_cls_score_edit,
            "pre_source_dest_cls_score_edit": pre_source_dest_cls_score_edit,
            "pre_cls_score_specificity": pre_cls_score_specificity,
            "pre_source_cls_score_general": pre_source_cls_score_general,
            "pre_dest_cls_score_general": pre_dest_cls_score_general,
            "pre_source_dest_cls_score_general": pre_source_dest_cls_score_general,
            "pre_source_cls_score_alias": pre_source_cls_score_alias,
            "pre_dest_cls_score_alias": pre_dest_cls_score_alias,
            "pre_source_dest_cls_score_alias": pre_source_dest_cls_score_alias,
        }
    else:
        post_source_cls_score_edit, post_dest_cls_score_edit, post_source_dest_cls_score_edit = \
            measure_scores(pipe, model, processor, requests, is_edited=True, dataset_name=dataset_name)

        print(f"post_source_cls_score_edit: {post_source_cls_score_edit}")
        print(f"post_dest_cls_score_edit: {post_dest_cls_score_edit}")
        print(f"post_source_dest_cls_score_edit: {post_source_dest_cls_score_edit}")

        post_source_cls_score_general, post_dest_cls_score_general, post_source_dest_cls_score_general = \
            measure_generalization_context(pipe, model, processor, num_edit=num_edit, is_edited=True, dataset_name=dataset_name)
        
        print(f"post_source_cls_score_general: {post_source_cls_score_general}")
        print(f"post_dest_cls_score_general: {post_dest_cls_score_general}")
        print(f"post_source_dest_cls_score_general: {post_source_dest_cls_score_general}")
        
        post_source_cls_score_alias, post_dest_cls_score_alias, post_source_dest_cls_score_alias = \
            measure_scores(pipe, model, processor, alias_val_requests, is_edited=True, dataset_name=dataset_name)   
        
        print(f"post_source_cls_score_alias: {post_source_cls_score_alias}")
        print(f"post_dest_cls_score_alias: {post_dest_cls_score_alias}")
        print(f"post_source_dest_cls_score_alias: {post_source_dest_cls_score_alias}")
        
        post_cls_score_specificity = measure_specificity(pipe, model, processor, is_edited=True,dataset_name=dataset_name)
        print(f"post_cls_score_specificity: {post_cls_score_specificity}")

        return {
            "post_source_cls_score_edit": post_source_cls_score_edit,
            "post_dest_cls_score_edit": post_dest_cls_score_edit,
            "post_source_dest_cls_score_edit": post_source_dest_cls_score_edit,
            "post_cls_score_specificity": post_cls_score_specificity,
            "post_source_cls_score_general": post_source_cls_score_general,
            "post_dest_cls_score_general": post_dest_cls_score_general,
            "post_source_dest_cls_score_general": post_source_dest_cls_score_general,
            "post_source_cls_score_alias": post_source_cls_score_alias,
            "post_dest_cls_score_alias": post_dest_cls_score_alias,
            "post_source_dest_cls_score_alias": post_source_dest_cls_score_alias,
        }


def emcid_test_text_encoder_imgnet(
    hparam_name="dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", 
    num_edit=10, 
    vit_type: Literal["base", "large"]="base", 
    mom2_weight=None,
    edit_weight=None,
    dataset_name: Literal["imgnet_small", "imgnet_aug"]="imgnet_small",
    device="cuda:0",
):
    """
    Args:
        hparam_name: the name of the hparam file
        num_edit: the number of edits
        vit_type: the type of the vit model
        mom2_weight: the weight of the mom2 update, if None, use the weight in the hparam file
        device: the device to use
    """
    # check if the results exist
    resutls_path = f"results/emcid/{hparam_name}"
    summary_file = f"{resutls_path}/{dataset_name}_summary.json"

    if "contrast" in hparam_name:
        hparams = ContrastEMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
        num_negative_prompts = hparams.num_negative_images
    else:
        hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
        num_negative_prompts = 0

    hparams = set_weights(hparams, mom2_weight, edit_weight)
    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight

    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            summary = json.load(file)
        key = f"edit{num_edit}_weight{mom2_weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
        print(key)
        if key in summary:
            print("returning")
            return summary[key]

    requests = RequestDataset(
                type="edit", 
                file_name=dataset_name + "_edit.json",
                num_negative_prompts=num_negative_prompts
                )[:num_edit]
    val_requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    val_requests = val_requests[:num_edit]
    alias_val_requests = compose_alias_test_requests(val_requests)

    

    cache_name = f"cache/{hparam_name}/{dataset_name}/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    vit_model_id = "google/vit-large-patch16-224" if vit_type == "large" else "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(vit_model_id)
    model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
    model.eval()

    # pre editing evaluation
    with torch.no_grad():
       pre_ret = eval_pipe_imgnet(pipe, 
                                  model, 
                                  processor, 
                                  requests, 
                                  alias_val_requests, 
                                  num_edit=num_edit, 
                                  is_edited=False,
                                  dataset_name=dataset_name)
    
    # delete the model
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # shuffle the requests
    indices = list(range(len(requests)))
    random.shuffle(indices)
    shuffled_requests = [requests[idx] for idx in indices]

    import time
    time1 = time.time()
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe, 
                    shuffled_requests, 
                    hparams, 
                    device, 
                    cache_name=cache_name)
    time2 = time.time()
    print(f"apply_emcid_to_text_encoder takes {time2 - time1} seconds.")
    
    if not os.path.exists(resutls_path):
        os.makedirs(resutls_path)
    
    # post editing evaluation
    with torch.no_grad():
        processor = AutoProcessor.from_pretrained(vit_model_id)
        model = ViTForImageClassification.from_pretrained(vit_model_id).to(device)
        model.eval()
        post_ret = eval_pipe_imgnet(new_pipe, 
                                    model, 
                                    processor, 
                                    requests, 
                                    alias_val_requests, 
                                    num_edit=num_edit, 
                                    is_edited=True, 
                                    dataset_name=dataset_name)
        
    # merge the results
    ret = {**pre_ret, **post_ret}

    weight = hparams.mom2_update_weight if mom2_weight is None else mom2_weight    
    # again load the summary file, in case it is modified by other processes
    if os.path.exists(summary_file):
        with open(summary_file, "r") as file:
            new_summary = json.load(file)
    else:
        new_summary = {}

    key = f"edit{num_edit}_weight{weight}" + ("" if hparams.edit_weight == 0.5 else f"_ew{hparams.edit_weight}")
    
    new_summary[key] = ret
    with open(summary_file, "w") as file:
        json.dump(new_summary, file, indent=4)

    return ret


def measure_scores(
        pipe, 
        eval_model, 
        processor, 
        requests, 
        is_edited, 
        dataset_name,
        is_val=False):
    """
    Args:
        is_edited: whether the pipe has been edited, if True, we cache the generated
            images so that we can load them instead of generating them again 
    """
    img_cache_dir = CACHE_DIR / "images" / dataset_name

    # load ImageNet classifier to measure the efficacy
    if dataset_name == "artists":
        # this is deperated, artists is not evaluated in this way
        result = generate_cal_itm_score(
                            pipe, 
                            eval_model, 
                            processor, 
                            requests,
                            is_edited,
                            img_cache_dir=img_cache_dir,
                            cal_dest_score=True,
                            cal_source_score=False,
                            cal_source_dest_score=False)

    elif dataset_name.startswith("imgnet"):
        result = generate_cal_cls_score(
                            pipe, 
                            eval_model, 
                            processor, 
                            requests,
                            is_edited,
                            img_cache_dir=img_cache_dir,
                            is_val=is_val)
    else:
        raise ValueError(f"Invalid dataset_name {dataset_name}.")

    mean_source_score = np.mean(result["source_scores"])\
        if len(result["source_scores"]) > 0 else None
    mean_dest_score = np.mean(result["dest_scores"])\
        if len(result["dest_scores"]) > 0 else None
    mean_source_dest_score = np.mean(result["source_dest_scores"]) \
        if len(result["source_dest_scores"]) > 0 else None

    return mean_source_score, mean_dest_score, mean_source_dest_score


def measure_generalization_context(
        pipe, 
        eval_model, 
        processor, 
        num_edit, 
        is_edited,
        dataset_name):
    requests = RequestDataset(type="val", file_name=dataset_name + "_edit.json")
    requests = requests[:num_edit]
    return measure_scores(pipe, eval_model, processor, requests, is_edited, dataset_name=dataset_name, is_val=True)
    


def _to_batch(test_file="data/iceb_data/imgnet_small_test.json", batch_size=3):
    with open(test_file, "r") as file:
        test_items = json.load(file)
    batches = []

    prompts = []
    indices = [] 
    # construct a batch for every class 
    last_class_name = None
    last_class_id = None
    for idx, item in enumerate(test_items):
        class_id = item["class id"]
        if (last_class_id is not None and class_id != last_class_id) or idx == len(test_items) - 1:
            # TODO fix random seed selection
            if idx == len(test_items) - 1:
                prompts.append(item["text prompt"])
                indices.append(item["idx"])
            batches.append({"prompts": prompts[:batch_size], "random seed": item["random seed"], 
                            "class id": last_class_id, "indices": indices[:batch_size], "class name": last_class_name})
            prompts = []
            indices = []
            prompts.append(item["text prompt"])
            indices.append(item["idx"])
            last_class_id = item["class id"]
            last_class_name = item["class name"]
            continue
        prompts.append(item["text prompt"])
        indices.append(item["idx"])
        last_class_id = class_id   
        last_class_name = item["class name"]
    return batches

def measure_specificity(
        pipe, 
        classifier, 
        processor, 
        is_edited,
        dataset_name,
        batch_size=3,
        ):

    test_file = f"data/iceb_data/{dataset_name}_test.json"
    img_cache_dir = CACHE_DIR / "images" / dataset_name
    
    batches = _to_batch(test_file, batch_size)
    scores = []
    for batch in tqdm(batches): 
        if is_edited:
            generator = torch.Generator(pipe.device).manual_seed(int(batch["random seed"]))
            images = pipe(batch["prompts"], guidance_scale=7.5, generator=generator).images
        else:
            # check if the images exist
            image_names = [f"{batch['class name']}_{idx}.png" for idx in batch["indices"]]
            cached = True
            for image_name in image_names:
                if not os.path.exists(img_cache_dir / image_name):
                    generator = torch.Generator(pipe.device).manual_seed(int(batch["random seed"]))
                    images = pipe(batch["prompts"], guidance_scale=7.5, generator=generator).images
                    for image, image_name in zip(images, image_names):
                        image.save(img_cache_dir / image_name)
                    cached = False
                    break
            # all the images exist, load them
            images = images if not cached else\
                    [Image.open(img_cache_dir / image_name)
                    for image_name in image_names]

        # calculate the score
        scores.append(
            calculate_single_cls_score(classifier, processor, images, class_id=batch["class id"]))

    mean_score = np.mean(scores) 

    return mean_score 


def generate_cal_cls_score(
        pipe, 
        classifier, 
        processor, 
        requests,
        is_edited,
        img_cache_dir=CACHE_DIR / "images" / "imgnet_aug",
        is_val=False):
    """
    Generate images and calculate the classification scores
    Note that the return values are lists of scores for each request

    Args:
        pipe: the diffusion pipeline
        classifier: the classifier
        processor: the processor
        requests: the requests
        is_edited: whether the pipe has been edited
        cal_source_score: whether to generate source images and calculate the source score
        cal_dest_score: whether to generate dest images and calculate the dest score
        cal_source_dest_score: whether to use source prompts to generate images and calculate the dest score
    """
    
    # generate images
    source_scores = []
    dest_scores = []
    source_dest_scores = []
    edit_str = "pre" if not is_edited else "post"
    if is_val:
        name_template = "val_{}_{}_{}.png" # class name, pre/post, idx
    else:
        name_template = "train_{}_{}_{}.png" # class name, pre/post, idx

    if not os.path.exists(img_cache_dir):
        os.makedirs(img_cache_dir)
    for request in tqdm(requests):
        source_prompts = [prompt.format(request["source"]) for prompt in request["prompts"]]
        dest_prompts = [prompt.format(request["dest"]) for prompt in request["prompts"]]

        # image file name format is: {source/dest}_{pre/post}_{prompt_idx}.png
        if not is_edited:
            # check if the source images exist
            source_image_names = [name_template.format(request["source"], edit_str, idx) for idx in request["indices"]]
            for i, source_image_name in enumerate(source_image_names):
                if not os.path.exists(img_cache_dir / source_image_name):
                    generator = torch.Generator(pipe.device).manual_seed(
                                        int(request["seeds"][i])) if request["seeds"] is not None else None
                    source_image = pipe(source_prompts[i:i+1], guidance_scale=7.5, generator=generator).images[0]
                    # save image
                    source_image.save(img_cache_dir / source_image_name)
            # all the source images exist, load them
            source_images = [Image.open(img_cache_dir / source_image_name)
                            for source_image_name in source_image_names]
        else:
            # no cache for post edit model
            source_images = []
            source_image_names = [name_template.format(request["source"], edit_str, idx) for idx in request["indices"]]
            for i in range(len(source_prompts)):
                generator = torch.Generator(pipe.device).manual_seed(
                                int(request["seeds"][i])) if request["seeds"] is not None else None
                source_image = pipe(source_prompts[i:i+1], guidance_scale=7.5, generator=generator).images[0]
                source_images.append(source_image)
            
        
        if not is_edited:
            # check if the dest images exist
            dest_image_names = [name_template.format(request["dest"], edit_str, idx) for idx in request["indices"]]
            for i, dest_image_name in enumerate(dest_image_names):
                if not os.path.exists(img_cache_dir / dest_image_name):
                    generator = torch.Generator(pipe.device).manual_seed(
                                    int(request["seeds"][i])) if request["seeds"] is not None else None
                    dest_image = pipe(dest_prompts[i:i+1], guidance_scale=7.5, generator=generator).images[0]
                    # save image
                    dest_image.save(img_cache_dir / dest_image_name)
            
            # all the dest images exist, load them
            dest_images = [Image.open(img_cache_dir / dest_image_name) 
                            for dest_image_name in dest_image_names]
        else:
            dest_images = []
            for i in range(len(dest_prompts)):
                generator = torch.Generator(pipe.device).manual_seed(
                                int(request["seeds"][i])) if request["seeds"] is not None else None
                dest_image = pipe(dest_prompts[i:i+1], guidance_scale=7.5, generator=generator).images[0]
                dest_images.append(dest_image)

        # use classifier to get the score
        source_score = calculate_single_cls_score(classifier, processor, source_images, request["source id"])
        dest_score = calculate_single_cls_score(classifier, processor, dest_images, request["dest id"])

        source_scores.append(source_score)
        dest_scores.append(dest_score)
        source_dest_scores.append(
        calculate_single_cls_score(classifier, processor, source_images, class_id=request["dest id"]))

    return {"source_scores": source_scores, "dest_scores": dest_scores, "source_dest_scores": source_dest_scores}


def generate_cal_itm_score(
        pipe, 
        itm_model, 
        processor, 
        requests,
        is_edited,
        img_cache_dir=CACHE_DIR / "images" / "imgnet_small",
        cal_source_score=True,
        cal_dest_score=False,
        cal_source_dest_score=False):
    """
    Generate images and calculate the image-text matching scores
    Note that the return values are lists of scores for each request

    Args:
        pipe: the diffusion pipeline
        itm_model: image-text matching evaluation model, clip or blip
        processor: the processor
        requests: the requests
        is_edited: whether the pipe has been edited
        cal_source_score: whether to generate source images and calculate the source score
        cal_dest_score: whether to generate dest images and calculate the dest score
        cal_source_dest_score: whether to use source prompts to generate images and calculate the dest score
    """

    if cal_dest_score or cal_source_dest_score:
        raise ValueError("dest score and source dest score are not supported, \
                         since we don't have appropriate dataset")

    # generate images
    source_scores = []
    dest_scores = []
    source_dest_scores = []
    edit_str = "pre" if not is_edited else "post"
    if not os.path.exists(img_cache_dir):
        os.makedirs(img_cache_dir)
    for request in tqdm(requests):
        source_prompts = [prompt.format(request["source"]) for prompt in request["prompts"]]\
                         if cal_source_score else []
        dest_prompts = [prompt.format(request["dest"]) for prompt in request["prompts"]]\
                         if cal_dest_score else []

        # image file name format is: {source/dest}_{pre/post}_{prompt_idx}.png
        generator = torch.Generator(pipe.device).manual_seed(int(request["seed"])) if request["seed"] is not None else None
        if cal_source_score and not is_edited:
            # check if the source images exist
            source_image_names = [f"{request['source']}_{edit_str}_{idx}.png" for idx in request["indices"]] 
            cached = True
            for source_image_name in source_image_names:
                if not os.path.exists(img_cache_dir / source_image_name):
                    source_images = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                    # generate images
                    for image, image_name in zip(source_images, source_image_names):
                        image.save(img_cache_dir / image_name)
                    cached = False
                    break
            # all the source images exist, load them
            source_images = source_images if not cached else \
                [Image.open(img_cache_dir / source_image_name)
                    for source_image_name in source_image_names]
        else:
            source_images = pipe(source_prompts, guidance_scale=7.5, generator=generator).images if cal_source_score else None
        
        dest_images = None
        if cal_dest_score and not is_edited:
            # check if the dest images exist
            dest_image_names = [f"{request['dest']}_{edit_str}_{idx}.png" for idx in request["indices"]]
            cached = True
            for dest_image_name in dest_image_names:
                if not os.path.exists(img_cache_dir / dest_image_name):
                    dest_images = pipe(dest_prompts, guidance_scale=7.5, generator=generator).images
                    for image, image_name in zip(dest_images, dest_image_names):
                        image.save(img_cache_dir / image_name)
                    cached = False
                    break
            # all the dest images exist, load them
            dest_images = dest_images if not cached else\
                    [Image.open(img_cache_dir / dest_image_name) 
                     for dest_image_name in dest_image_names]

        else:
            dest_images = pipe(dest_prompts, guidance_scale=7.5, generator=generator).images if cal_dest_score else None

        # use classifier to get the score
        # TODO make clip and blip not only support single image input 
        cal_fun = calculate_single_blip_score if isinstance(itm_model, BlipForImageTextRetrieval) else calculate_single_clip_score
        if cal_source_score:
            for source_prompt, img in zip(source_prompts, source_images):
                source_score = cal_fun(itm_model, processor, img, source_prompt) \
                                if cal_source_score and source_images else None
                source_scores.append(source_score)

        if cal_dest_score:
            for dest_prompt, img in zip(dest_prompts, dest_images):
                dest_score = cal_fun(itm_model, processor, img, dest_prompt) \
                                if cal_dest_score and dest_images else None
                dest_scores.append(dest_score)

        if cal_source_dest_score:
                source_dest_score = cal_fun(itm_model, processor, img, dest_prompt)\
                                        if cal_source_dest_score else None
                source_dest_scores.append(source_dest_score)

    return {"source_scores": source_scores, "dest_scores": dest_scores, "source_dest_scores": source_dest_scores}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam", type=str, default="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01")
    # receive a list of ids
    to_list = lambda x: list(map(int, x.split(",")))
    to_bool = lambda x: x.lower() == "true"
    parser.add_argument("--device_ids", type=to_list, default=[0, 1]) 
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--edit", type=int, default=50)
    parser.add_argument("--weights", type=to_list, \
                        default=[1000, 2000, 5000, 6000, 8000, 10000, 15000])
    parser.add_argument("--unet", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="imgnet_small")
    parser.add_argument("--coco", action="store_true", default=False)
    parser.add_argument("--artists", action="store_true", default=False)
    parser.add_argument("--eval_imgnet", type=to_bool, default=True)
    parser.add_argument("-ew", "--edit_weight", type=float, default=0.5)
    parser.add_argument("--descend", action="store_true", default=False)
    parser.add_argument("--mom2", type=int, default=None)
    parser.add_argument("--dest", type=str, default="art")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--num_artist", type=to_list, default=[1, 5, 10, 50])
    parser.add_argument("--method", type=str, default="emcid", help="only used for imgnet_mend")

    args = parser.parse_args()
    print(args)
    print(args.hparam)
    if args.unet:
        emcid_test_artwork_qualitative_unet(
            hparam_name=args.hparam, 
            mom2_weight=10000,
            device=args.device,
        )
    elif args.dataset == "i2p":
        # TODO currently only tries to remove nudity, use better conditioning
        emcid_test_text_encoder_i2p(
            hparam_name=args.hparam, 
            keywords=["nudity"],
            mom2_weight=1,
            edit_weight=args.edit_weight,
            device=args.device,
        )
    elif args.dataset == "artists":
        num_list = args.num_artist
        if args.descend:
            num_list = num_list[::-1]
        for num in num_list: 
            emcid_test_text_encoder_artists(
                sample_num=args.sample_num,
                num_artist=num,
                hparam_name=args.hparam, 
                mom2_weight=args.mom2,
                edit_weight=args.edit_weight,
                dest=args.dest,
                use_coco_eval=args.coco,
                use_artists_eval=args.artists,
                device=args.device,
            )
    elif args.dataset == "imgnet_aug_and_i2p":
        emcid_test_sd_imgnet_and_i2p(
            num_edit=args.edit,
            preserve_emcid_edit_num=200,
            eval_imgnet=args.eval_imgnet,
            uce_technique="tensor",
            hparam_name=args.hparam,
            vit_type="base",
            mom2_weight=args.mom2,
            edit_weight=args.edit_weight,
            device=args.device
        )
    elif args.dataset == "imgnet_mend":
        emcid_test_imgnet_mend(
            hparam_name=args.hparam,
            method=args.method,
            dataset_name=args.dataset,
            num_edit=args.edit,
            vit_type="base",
            eval_coco=args.coco,
            eval_imgnet=args.eval_imgnet,
            mom2_weight=args.mom2,
            edit_weight=args.edit_weight,   
            device=args.device,
        )

    else:
        for mom2_weight in args.weights:
            emcid_test_text_encoder_imgnet(
                hparam_name=args.hparam, 
                num_edit=args.edit, 
                vit_type="base", 
                mom2_weight=mom2_weight,
                dataset_name=args.dataset,
                device=args.device,
            )
