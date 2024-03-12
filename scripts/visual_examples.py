import json
from typing import List, Dict
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import (
    AutoProcessor,
    ViTForImageClassification
)

from util.evaluate import calculate_single_cls_score
from util.globals import *
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_cross_attn
from experiments.emcid_test import set_weights

from emcid.uce_train import edit_model_uce, edit_model_uce_modified, edit_text_encoder_uce
from dsets.iceb_dataset import *
from dsets.artist_requests import ArtistRequestsDataset
from dsets.global_concepts import get_i2p_editing_requests, NSFWEditRequestDataset



# helper function taken from: https://huggingface.co/blog/stable_diffusion
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[1].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def uce_test_nudity(
    custom=False,
    edit_part="cross_attn",
    mom2_weight=None,
    lamb=0.1,
    device="cuda:0"
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    requests = get_i2p_editing_requests(
        i2p_path="data/i2p/unsafe-prompts4703.csv",
        num_requsts=2,
        batch_size=3
    )

    save_dir = f"{RESULTS_DIR}/baselines/i2p_test/uce_{edit_part}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate pre edit images
    with torch.no_grad():
        for prompt, seed, idx in zip(
                                requests[0]["source_prompts"], 
                                requests[0]["seeds"], 
                                requests[0]["indices"]):

            save_path = f"{save_dir}/train_pre_edit_{idx}.png"
            if os.path.exists(save_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(save_path)
        
        for prompt, seed, idx in zip(requests[1]["source_prompts"], requests[1]["seeds"], requests[1]["indices"]):
            save_path = f"{save_dir}/test_pre_edit_{idx}.png"
            if os.path.exists(save_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(save_path)
    
    # i2p edit
    # safety_concepts = 'harassment, violence, suffering, humiliation, harm,' \
    #                   'sexual, nudity, blood,' \
    #                   'vandalism, weapons, brutality, cruelty'
    safety_concepts = 'nudity, sexual, blood'
    # safety_concepts = 'nudity, sexual'

    # safety_concepts = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
    #                     'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
    #                     'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'
    # safety_concepts = 'nsfw'

    obj_requests = RequestDataset(
                type="edit", 
                file_name="imgnet_aug_edit.json",
                num_negative_prompts=0,
                )[:200]
    # old_texts for nsfw concepts
    retain_texts = []
    for request in obj_requests:
        retain_texts.append(request["source"])
    old_texts = safety_concepts.split(",")
    for old_text in old_texts:
        old_text = old_text.strip()

    new_texts = [" "] * len(old_texts)
    print("Edit model following uce approach")

    with torch.no_grad():
        if custom:
            raise NotImplementedError
            new_pipe = edit_model_uce_modified(
                pipe,
                old_text_=old_texts,
                new_text_=new_texts,
                mom2_weight=mom2_weight,
                technique="tensor"
            )
        else:
            if edit_part == "cross_attn":
                new_pipe = edit_model_uce(
                    pipe,
                    lamb=lamb,
                    old_text_=old_texts,
                    new_text_=new_texts,
                    retain_text_=retain_texts,
                    technique="tensor"
                )
            elif edit_part == "text_encoder":
                new_pipe = edit_text_encoder_uce(
                    pipe,
                    lamb=lamb,
                    old_text_=old_texts,
                    new_text_=new_texts,
                    retain_text_=retain_texts,
                    technique="tensor"
                )
    
    # generate post edit images
    with torch.no_grad():
        for prompt, seed, idx in zip(requests[0]["source_prompts"], requests[0]["seeds"], requests[0]["indices"]):
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            save_path = f"{save_dir}/train_post_edit_{idx}.png"
            img.save(save_path)
        
        for prompt, seed, idx in zip(requests[1]["source_prompts"], requests[1]["seeds"], requests[1]["indices"]):
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            save_path = f"{save_dir}/test_post_edit_{idx}.png"
            img.save(save_path)
    
    # generate unrelated images
    with torch.no_grad():
        prompts = [
            "a photo of a dog",
            "a photo of a cat",
        ]

        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(pipe.device).manual_seed(idx) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            save_path = f"{save_dir}/unrelated_{prompt}.png"
            img.save(save_path)
    

def emcid_test_nudity(
    hparam_name="dest_s-200_c-1.5_ly-12_lr-0.5_wd-0e-00_global",
    mom2_weight=None,
    edit_weight=None,
    device="cuda:0"
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    test_requests = get_i2p_editing_requests(
        i2p_path="data/i2p/unsafe-prompts4703.csv",
        num_requsts=2,
        batch_size=3
    )

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    cache_name = f"cache/{hparam_name}/i2p_test/"
    if not os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test"):
        os.makedirs(f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test")

    # generate pre edit images
    with torch.no_grad():
        for prompt, seed, idx in zip(test_requests[0]["source_prompts"], test_requests[0]["seeds"], test_requests[0]["indices"]):
            save_path = f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test/train_pre_edit_{idx}.png"
            if os.path.exists(save_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(save_path)
        
        for prompt, seed, idx in zip(test_requests[1]["source_prompts"], test_requests[1]["seeds"], test_requests[1]["indices"]):
            save_path = f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test/test_pre_edit_{idx}.png"
            if os.path.exists(save_path):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(save_path)
    
    # i2p edit
    # safety_concepts = 'harassment, violence, suffering, humiliation, harm,' \
    #                   'sexual, nudity, blood,' \
    #                   'vandalism, weapons, brutality, cruelty'
    safety_concepts = 'nudity, sexual, blood'
    # safety_concepts = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
    #                     'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
    #                     'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'
    # safety_concepts = 'nsfw'

    edit_requests = NSFWEditRequestDataset(
                        keywords=["nudity", "sexual", "exposed male breast"],
                        num_prompts=3)

    new_pipe, _ = apply_emcid_to_cross_attn(
        pipe,
        edit_requests,
        hparams=hparams,
        device=device,
        mom2_weight=mom2_weight,
        edit_weight=edit_weight,
        return_orig_text_model=False,
        cache_name=cache_name
    )
    
    # generate post edit images
    with torch.no_grad():
        for prompt, seed, idx in zip(test_requests[0]["source_prompts"], test_requests[0]["seeds"], test_requests[0]["indices"]):
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test/train_post_edit_{idx}.png")
        
        for prompt, seed, idx in zip(test_requests[1]["source_prompts"], test_requests[1]["seeds"], test_requests[1]["indices"]):
            generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
            img = new_pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test/test_post_edit_{idx}.png")
    
    # generate unrelated images
    with torch.no_grad():
        prompts = [
            "a photo of a dog",
            "a photo of a cat",
        ]

        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(pipe.device).manual_seed(idx) if seed is not None else None
            img = pipe(
                [prompt], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            img.save(f"{RESULTS_DIR}/emcid/{hparam_name}/i2p_test/unrelated_{prompt}.png") 


def qualitative_plot_van_gogh(
    pipe: StableDiffusionPipeline, 
    hparam_name="dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01",
    seed_train=2023,
    mom2_weight=4000,
    edit_weight=0.5,
):
    # erase van gogh style as "a realist photographer", then test the erasing effect
    # on misspelled van gogh name: "ven gogh", "van gough"
    template = [
    "painting by {}",
    "artwork by {}",
    "style of {}"
    ]
    
    train_requests = [
        {
            "prompts": template[:],
            "source": "Vincent van Gogh",
            "seed_train": seed_train,
            # "dest": "a realist photographer",
            "dest": "a realist artist"
        },
    ]

    test_items = [
        ("The Great Wave off Kanagawa by Hokusai", 1656, "Hokusai"),
        ("Girl with a Pearl Earring by Johannes Vermeer",4896, "Johannes Vermeer"),
        ("The Scream by Edvard Munch", 804, "Edvard Munch")
    ]

    val_items = [
        ("A Wheatfield, with Cypresses by Vincent van Gogh",2219, "Vincent van Gogh"),
        ("Almond Blossoms by Vincent van Gogh",	4965, "Vincent van Gogh"),
        ("Bridge at Trinquetaille by Vincent van Gogh",3370,"Vincent van Gogh"),
        ("Bedroom in Arles by Vincent van Gogh", 2795, "Vincent van Gogh"), 
        ("Bedroom in Arles by ven gogh", 2795, "Vincent van Gogh"),
        ("Bedroom in Arles by van gough", 2795, "Vincent van Gogh"),
    ]
    
    # generate pre edit images
    pre_edit_source_imgs = []
    pre_edit_dest_imgs = []
    pre_edit_val_imgs = []
    pre_edit_test_imgs = []


    def _plot(pre=True, save_dir=f"{RESULTS_DIR}/emcid/{hparam_name}/artists/van gogh/"):
        pre_str = "pre" if pre else "post"
        with torch.no_grad():
            for item in val_items:
                if pre and os.path.exists(save_dir + f"val/{pre_str}_{item[0]}.png"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
                img = pipe(
                    [item[0]], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                
                if not os.path.exists(save_dir + f"val"):
                    os.makedirs(save_dir + f"val")
                img.save(save_dir + f"val/{pre_str}_{item[0]}.png")
                pre_edit_val_imgs.append((img, item[0]))
            
            for item in test_items:
                if pre and os.path.exists(save_dir + f"test/{pre_str}_{item[0]}.png"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
                img = pipe(
                    [item[0]], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                
                if not os.path.exists(save_dir + f"test"):
                    os.makedirs(save_dir + f"test")
                img.save(save_dir + f"test/{pre_str}_{item[0]}.png")
                pre_edit_test_imgs.append((img, item[0]))
    
    # _plot(pre=True)

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    cache_name = f"cache/{hparam_name}/artists/"
    pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                        pipe, 
                        requests=train_requests, 
                        hparams=hparams, 
                        device=pipe.device, 
                        mom2_weight=mom2_weight,
                        edit_weight=edit_weight,
                        return_orig_text_encoder=True,
                        cache_name=cache_name)
    
    # _plot(pre=False)

    pipe.text_encoder = orig_text_encoder

    # erase the starry night and expore the effect of different edit_weight and different layer num

    train_requests = [
        {
            "prompts": [
                "A {} painting",
                "An image of {}",
                "{}"
            ],
            "source": "The Starry Night",
            "seed_train": seed_train,
            "dest": "full-moon night"
        },
        {
            "prompts": template[:],
            "source": "Vincent van Gogh",
            "seed_train": seed_train,
            # "dest": "a realist photographer",
            "dest": "a realist artist"
        },
    ]

    val_items = [
        ("The Starry Night by Vincent van Gogh", 4813, "Vincent van Gogh"),
        ("The Starry Night", 4812, "Vincent van Gogh"),
    ]

    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/artists/the starry night/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # generate pre edit images
    _plot(pre=True, save_dir=save_dir + "pre/")

    pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                        pipe, 
                        requests=train_requests, 
                        hparams=hparams, 
                        device=pipe.device, 
                        mom2_weight=mom2_weight,
                        edit_weight=edit_weight,
                        return_orig_text_encoder=True,
                        cache_name=cache_name)
    
    _plot(pre=False, save_dir=save_dir + "post/")

    pipe.text_encoder = orig_text_encoder

    # explore the effect of different edit_weight
    edit_weight_list = list(np.arange(0.1, 1.0, 0.1))

    for edit_weight in edit_weight_list:
        set_weights(pipe, edit_weight=edit_weight, mom2_weight=mom2_weight)

        pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                        pipe, 
                        requests=train_requests, 
                        hparams=hparams, 
                        device=pipe.device, 
                        mom2_weight=mom2_weight,
                        edit_weight=edit_weight,
                        return_orig_text_encoder=True,
                        cache_name=cache_name)
    
        _plot(pre=False, save_dir=save_dir + f"/ew/{edit_weight:.1f}/")

        pipe.text_encoder = orig_text_encoder
    
    # # explore the effect of different layer num
    # layers_list = [list(range(i, 11)) for i in range(0, 11)]

    # for layers in layers_list:
    #     hparams.layers = layers
    #     set_weights(pipe, edit_weight=0.5, mom2_weight=mom2_weight)

    #     pipe, orig_text_encoder = apply_emcid_to_text_encoder(
    #                     pipe, 
    #                     requests=train_requests, 
    #                     hparams=hparams, 
    #                     device=pipe.device, 
    #                     mom2_weight=mom2_weight,
    #                     edit_weight=edit_weight,
    #                     return_orig_text_encoder=True,
    #                     cache_name=cache_name)
    
    #     _plot(pre=False, save_dir=save_dir + f"/layer/num_{len(layers)}/")

    #     pipe.text_encoder = orig_text_encoder

def qualitative_plot_artists(
    pipe: StableDiffusionPipeline, 
    hparam_name,
    dataset_name="artists",
    seed_train=2023,
    mom2_weight=None,
    edit_weight=None
    ):

    train_edit_prompts = [
        ("A Wheatfield, with Cypresses by Vincent van Gogh",2219, "Vincent van Gogh"),
        ("Almond Blossoms by Vincent van Gogh",	4965, "Vincent van Gogh"),
        ("Bridge at Trinquetaille by Vincent van Gogh",3370,"Vincent van Gogh"),
    ]


    val_prompts = [
        ("Bedroom in Arles by Vincent van Gogh", 2795, "Vincent van Gogh"), 
        ("Bedroom in Arles by the little painter fellow", 2795, "Vincent van Gogh"),
        ("Bedroom in Arles by ven gogh", 2795, "Vincent van Gogh"),
        ("Bedroom in Arles by van gough", 2795, "Vincent van Gogh"),
        # ("The Starry Night by Vincent van Gogh", 4813, "Vincent van Gogh"),
        # ("The Starry Night", 4812, "Vincent van Gogh"),
    ]

    test_items = [
        ("The Great Wave off Kanagawa by Hokusai", 1656, "Hokusai"),
        ("Girl with a Pearl Earring by Johannes Vermeer",4896, "Johannes Vermeer"),
        ("The Scream by Edvard Munch", 804, "Edvard Munch")
    ]

    train_requests = [
        {
            "prompts": [item[0].replace(item[2], "{}") for item in train_edit_prompts],
            "source": train_edit_prompts[0][2],
            "seed_train": seed_train,
            "dest": "a realist photographer",
            # "dest": "an artist"
        },
        # {
        #     "prompts": [
        #         "A {} painting",
        #         "An image of {}",
        #         "{}"
        #     ],
        #     "source": "The Starry Night",
        #     "seed_train": seed,
        #     "dest": "night filled with stars"
        # }
    ]
    # train_requests = train_requests[0:1]

    # generate pre edit images
    pre_edit_source_imgs = []
    pre_edit_dest_imgs = []
    pre_edit_val_imgs = []
    pre_edit_test_imgs = []

    with torch.no_grad():
        for item in train_edit_prompts:
            if os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{item[0]}.png"):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            pre_edit_source_imgs.append((img, item[0]))
        
        for item in val_prompts:
            if os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/val/pre_edit_{item[0]}.png"):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            pre_edit_val_imgs.append((img, item[0]))
        
        for item in train_requests:
            if "esd" in hparam_name:
                break
            for prompt in item["prompts"]:
                dest_prompt = prompt.format(item["dest"])
                if os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{dest_prompt}.png"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
                img = pipe(
                    [dest_prompt], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                pre_edit_dest_imgs.append((img, dest_prompt))
            
        for item in test_items:
            if os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/test/pre_edit_{item[0]}.png"):
                continue
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            pre_edit_test_imgs.append((img, item[0]))

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    cache_name = f"cache/{hparam_name}/{dataset_name}/"
    new_pipe, _ = apply_emcid_to_text_encoder(
                        pipe, 
                        requests=train_requests, 
                        hparams=hparams, 
                        device=pipe.device, 
                        mom2_weight=mom2_weight,
                        edit_weight=edit_weight,
                        cache_name=cache_name)
    # generate post edit images
    post_edit_source_imgs = []
    post_edit_val_imgs = []
    post_edit_test_imgs = []
    with torch.no_grad():
        for item in val_prompts:
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = new_pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            post_edit_val_imgs.append((img, item[0]))
        
        for item in train_edit_prompts:
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = new_pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            post_edit_source_imgs.append((img, item[0]))

        for item in test_items:
            generator = torch.Generator(pipe.device).manual_seed(seed_train) if seed_train is not None else None
            img = new_pipe(
                [item[0]], 
                generator=generator, 
                guidance_scale=7.5).images[0]
            post_edit_test_imgs.append((img, item[0]))
        
        for subfolder in ["train", "val", "test"]:
            if not os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/{subfolder}"):
                os.makedirs(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/{subfolder}")
    

    # save images
    for idx, item in enumerate(pre_edit_source_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{item[1]}.png")
    for idx, item in enumerate(pre_edit_val_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/val/pre_edit_{item[1]}.png")
    for idx, item in enumerate(pre_edit_test_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/test/pre_edit_{item[1]}.png")
    for idx, item in enumerate(pre_edit_dest_imgs):
        if "esd" in hparam_name:
            break
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/pre_edit_{item[1]}.png")
    

    for idx, item in enumerate(post_edit_source_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/train/post_edit_{item[1]}.png")
    for idx, item in enumerate(post_edit_val_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/val/post_edit_{item[1]}.png")
    for idx, item in enumerate(post_edit_test_imgs):
        item[0].save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/test/post_edit_{item[1]}.png")


def plot_varying_mom2(
        pipe: StableDiffusionPipeline,
        hparam_name,
        requests,
        prompt,
        seed,
        dataset_name="visual",
        mom2_weights=[1000, 2000, 5000, 6000, 8000, 10000, 15000]):
    """
    draw visual examples of generating `prompt` with varying mom2 weights, using seed
    """
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    imgs = []
    for mom2_weight in mom2_weights:
        cache_name = f"cache/{hparam_name}/{dataset_name}/"
        pipe, origin_text_encoder =\
            apply_emcid_to_text_encoder(
                            pipe, 
                            requests=requests, 
                            hparams=hparams, 
                            device=pipe.device, 
                            mom2_weight=mom2_weight,
                            return_orig_text_encoder=True,
                            cache_name=cache_name)
        generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
        img = pipe(
            [prompt], 
            generator=generator, 
            guidance_scale=7.5).images[0]
        imgs.append(img)

        # delete the changed text encoder
        with torch.cuda.device(pipe.device):
            del pipe.text_encoder
            torch.cuda.empty_cache()
        pipe.text_encoder = origin_text_encoder

    # compose the imgs into a grid
    grid = Image.new('RGB', (imgs[0].width * len(imgs), imgs[0].height))
    for idx, img in enumerate(imgs):
        grid.paste(img, (idx * imgs[0].width, 0))
    
    if not os.path.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/visual"):
        os.makedirs(f"{RESULTS_DIR}/emcid/{hparam_name}/visual")
    grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/visual/varying_mom2.png")



def sample_plot(
        pipe: StableDiffusionPipeline, 
        classifier, 
        processor,
        hparam_name,
        edit_file_name="imgnet_small_edit.json", 
        test_file_name="imgnet_small_test.json", 
        dataset_name: Literal["imgnet_small", "imgnet_aug", "artists"]="imgnet_small",
        seed=2023,
        num_show=9,
        num_edit=50,
        mom2_weight=None,
        edit_weight=None):
    """
    sample num prompts from edit,m general, test dataset each.
    generate pre edit grids and post edit grids
    """
    # sample prompts, set fixed seed for reproducibility
    edit_requests = RequestDataset(type="edit", file_name=edit_file_name)
    general_dataset = RequestDataset(type="val", file_name=edit_file_name)

    edit_samples = edit_requests.sample(num_show, seed=seed)

    # find the prompts of the same class in the general dataset
    general_samples = []
    for item in edit_samples:
        for i in range(len(general_dataset)):
            if general_dataset[i]["source id"] == item["source id"]:
                general_samples.append(general_dataset[i])
                break
    
    # sample from test dataset
    test_dataset = ObjectPromptDataset(file_name=test_file_name)
    torch.manual_seed(seed)
    indices = torch.randperm(len(test_dataset))[:num_show]
    test_samples = [test_dataset[idx] for idx in indices]
    for item in test_samples:
        print(item["class name"])
    # find those with an alias
    aliases = []
    with open("data/iceb_data/vit_classifier_config.json", "r") as f:
        id2label = json.load(f)["id2label"]
    with open("data/iceb_data/class2id.json", "r") as f:
        class2id = json.load(f)
    for item in edit_samples:
        id = class2id[item["source"]]
        if len(id2label[str(id)].split(",")) > 1:
            aliases.append(id2label[str(id)].split(",")[-1])
        else:
            aliases.append(None)
    print(aliases)
    print([item["source"] for item in edit_samples])
    
    alias_samples = []
    alias_samples_general = []
    for idx, alias in enumerate(aliases):
        if alias is None:
            alias_samples.append(None)
            alias_samples_general.append(None)
        else:
            item = edit_samples[idx].copy()
            general_item = general_samples[idx].copy()
            item["source"] = alias
            general_item["source"] = alias
            alias_samples.append(item)
            alias_samples_general.append(general_item)

    # generate pre edit grids
    pre_edit_source_imgs = []
    pre_edit_dest_imgs = []
    pre_edit_alias_imgs = []

    pre_general_source_imgs = []
    pre_general_alias_imgs = []
    pre_test_imgs = []

    pre_edit_source_scores = []
    pre_edit_dest_scores = []
    pre_edit_alias_scores = []
    pre_general_cls_scores = []
    pre_test_cls_scores = []

    def _generate_eval_dest(pipe, request, img_list, score_list):
        generator = torch.Generator(pipe.device).manual_seed(int(request["seed"])) if request["seed"] is not None else None
        img = pipe(
            [request["prompts"][0].format(request["dest"])], 
            generator=generator, 
            guidance_scale=7.5).images[0]
        
        img_list.append(img)

        # calculate the score
        score = calculate_single_cls_score(classifier, processor, [img], request["dest id"])
        score_list.append(score)

    def _generate_eval_source(pipe, request, img_list, score_list):
        generator = torch.Generator(pipe.device).manual_seed(int(request["seed"])) if request["seed"] is not None else None
        img = pipe(
            [request["prompts"][0].format(request["source"])], 
            generator=generator, 
            guidance_scale=7.5).images[0]

        img_list.append(img)

        # calculate the score
        source_score = calculate_single_cls_score(classifier, processor, [img], request["source id"])
        source_dest_score = calculate_single_cls_score(classifier, processor, [img], request["dest id"])
        score_list.append({"source": source_score, "source_dest": source_dest_score})
    
    def _generate_eval_single(pipe, object_prompt, img_list, score_list):
        generator = torch.Generator(pipe.device).manual_seed(int(object_prompt["random seed"]))
        img = pipe(object_prompt["text prompt"], generator=generator, guidance_scale=7.5).images[0]
        img_list.append(img)

        # calculate the score
        score = calculate_single_cls_score(classifier, processor, [img], object_prompt["class id"])
        score_list.append(score)

    def _average(scores: List[Dict], key: str):
        return sum([item[key] for item in scores]) / len(scores)

    for item in edit_samples:
        _generate_eval_dest(pipe, item, pre_edit_dest_imgs, pre_edit_dest_scores)

    for item in edit_samples:
        _generate_eval_source(pipe, item, pre_edit_source_imgs, pre_edit_source_scores)
    
    for item in general_samples:
        _generate_eval_source(pipe, item, pre_general_source_imgs, pre_general_cls_scores)
    
    for item in test_samples:
        _generate_eval_single(pipe, item, pre_test_imgs, pre_test_cls_scores)
    
    for item in alias_samples:
        # create a white image
        if item is None:
            white_img = Image.new("RGB", (RESOLUTION, RESOLUTION), (255, 255, 255))
            pre_edit_alias_imgs.append(white_img)
            pre_edit_alias_scores.append({"source": 0, "source_dest": 0})
        else:
            _generate_eval_source(pipe, item, pre_edit_alias_imgs, pre_edit_alias_scores)
    
    for item in alias_samples_general:
        if item is None:
            white_img = Image.new("RGB", (RESOLUTION, RESOLUTION), (255, 255, 255))
            pre_general_alias_imgs.append(white_img)
            pre_general_cls_scores.append({"source": 0, "source_dest": 0})
        else:
            _generate_eval_source(pipe, item, pre_general_alias_imgs, pre_general_cls_scores)   

    pre_edit_dest_grid = image_grid(pre_edit_dest_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_edit_dest_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_edit_dest_w{mom2_weight}_grid.png")
    
    pre_edit_source_grid = image_grid(pre_edit_source_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_edit_source_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_edit_source_w{mom2_weight}_grid.png")

    pre_edit_alias_grid = image_grid(pre_edit_alias_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_edit_alias_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_edit_alias_w{mom2_weight}_grid.png")

    pre_general_alias_grid = image_grid(pre_general_alias_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_general_alias_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_general_alias_w{mom2_weight}_grid.png")

    pre_general_source_grid = image_grid(pre_general_source_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_general_source_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_general_source_w{mom2_weight}_grid.png")

    pre_test_grid = image_grid(pre_test_imgs, int(num_show**0.5), int(num_show**0.5))
    pre_test_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/pre_test_w{mom2_weight}_grid.png")

    # calculate the average score
    pre_edit_dest_score = sum(pre_edit_dest_scores) / len(pre_edit_dest_scores)

    pre_edit_source_score = _average(pre_edit_source_scores, "source")
    pre_edit_source_dest_score = _average(pre_edit_source_scores, "source_dest")

    pre_edit_alias_score = _average(pre_edit_alias_scores, "source")
    
    pre_general_source_score = _average(pre_general_cls_scores, "source")
    pre_general_source_dest_score = _average(pre_general_cls_scores, "source_dest")

    pre_test_score = sum(pre_test_cls_scores) / len(pre_test_cls_scores)

    # edit the pipe line 
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    cache_name = f"cache/{hparam_name}/{dataset_name}/"

    requests = RequestDataset(type="edit", file_name=edit_file_name)[:num_edit]

    classifier = classifier.to("cpu")
    # with torch.cuda.device(pipe.device):
    #     torch.cuda.empty_cache()

    new_pipe, _ = apply_emcid_to_text_encoder(
                     pipe, 
                     requests=requests, 
                     hparams=hparams, 
                     device=pipe.device, 
                     mom2_weight=mom2_weight,
                     edit_weight=edit_weight,
                     cache_name=cache_name)

    classifier = classifier.to(pipe.device)
    # generate post edit grids
    post_edit_source_imgs = []
    post_general_source_imgs = []
    post_general_alias_imgs = []
    post_edit_alias_imgs = []
    post_test_imgs = []

    post_edit_cls_scores = []
    post_edit_alias_scores = []
    post_general_cls_scores = []
    post_test_cls_scores = []

    for item in edit_samples:
        _generate_eval_source(new_pipe, item, post_edit_source_imgs, post_edit_cls_scores)
    
    for item in general_samples:
        _generate_eval_source(new_pipe, item, post_general_source_imgs, post_general_cls_scores)
    
    for item in test_samples:
        _generate_eval_single(new_pipe, item, post_test_imgs, post_test_cls_scores)
    
    for item in alias_samples:
        if item is None:
            white_img = Image.new("RGB", (RESOLUTION, RESOLUTION), (255, 255, 255))
            post_edit_alias_imgs.append(white_img)
            post_edit_alias_scores.append({"source": 0, "source_dest": 0})
        else:
            _generate_eval_source(new_pipe, item, post_edit_alias_imgs, post_edit_alias_scores)

    for item in alias_samples_general:
        if item is None:
            white_img = Image.new("RGB", (RESOLUTION, RESOLUTION), (255, 255, 255))
            post_general_alias_imgs.append(white_img)
            post_general_cls_scores.append({"source": 0, "source_dest": 0})
        else:
            _generate_eval_source(new_pipe, item, post_general_alias_imgs, post_general_cls_scores)

    # generate post edit grids
    post_edit_source_grid = image_grid(post_edit_source_imgs, int(num_show**0.5), int(num_show**0.5))
    post_edit_source_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/post_edit_source_w{mom2_weight}_grid.png")

    post_general_source_grid = image_grid(post_general_source_imgs, int(num_show**0.5), int(num_show**0.5))
    post_general_source_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/post_general_source_w{mom2_weight}_grid.png")

    post_edit_alias_grid = image_grid(post_edit_alias_imgs, int(num_show**0.5), int(num_show**0.5))
    post_edit_alias_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/post_edit_alias_w{mom2_weight}_grid.png")

    post_general_alias_grid = image_grid(post_general_alias_imgs, int(num_show**0.5), int(num_show**0.5))
    post_general_alias_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/post_general_alias_w{mom2_weight}_grid.png")

    post_test_grid = image_grid(post_test_imgs, int(num_show**0.5), int(num_show**0.5))
    post_test_grid.save(f"{RESULTS_DIR}/emcid/{hparam_name}/post_test_w{mom2_weight}_grid.png")

    # calculate the average score
    post_edit_source_score = _average(post_edit_cls_scores, "source")
    post_edit_source_dest_score = _average(post_edit_cls_scores, "source_dest")
    
    post_general_source_score = _average(post_general_cls_scores, "source")
    post_general_source_dest_score = _average(post_general_cls_scores, "source_dest")

    post_edit_alias_score = _average(post_edit_alias_scores, "source")

    post_test_score = sum(post_test_cls_scores) / len(post_test_cls_scores)

    # save the numerical results
    if os.exists(f"{RESULTS_DIR}/emcid/{hparam_name}/{dataset_name}_visual_summary.json"):
        with open(f"{RESULTS_DIR}/emcid/{hparam_name}/{dataset_name}_visual_summary.json", "r") as f:
            old_results = json.load(f)
    else:
        old_results = {}

    with open(f"{RESULTS_DIR}/emcid/{hparam_name}/{dataset_name}_visual_summary.json", "w") as f:
        ret = {
            "pre_edit_dest_score": pre_edit_dest_score,

            "pre_edit_source_score": pre_edit_source_score,
            "pre_edit_source_dest_score": pre_edit_source_dest_score,

            "pre_edit_alias_score": pre_edit_alias_score,

            "pre_general_source_score": pre_general_source_score,
            "pre_general_source_dest_score": pre_general_source_dest_score,
            
            "pre_test_score": pre_test_score,

            "post_edit_source_score": post_edit_source_score,
            "post_edit_source_dest_score": post_edit_source_dest_score,

            "post_edit_alias_score": post_edit_alias_score,

            "post_general_source_score": post_general_source_score,
            "post_general_source_dest_score": post_general_source_dest_score,

            "post_test_score": post_test_score,
        }
        old_results.update({f"edit{num_edit}_weight{edit_weight}": ret})

        weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
        json.dump(
            old_results,
            f, 
            indent=4)
    
    return ret


def execute_sample_plot(
        hparam_name, 
        device, 
        num=9, 
        num_edit=50, 
        mom2_weight=None,
        edit_weight=None):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    classifier_id = "google/vit-base-patch16-224"
    processor = AutoProcessor.from_pretrained(classifier_id)
    classifier = ViTForImageClassification.from_pretrained(classifier_id).to(device)
    classifier.eval()

    sample_plot(
        pipe, 
        classifier, 
        processor, 
        hparam_name, 
        num_show=num, 
        num_edit=num_edit, 
        mom2_weight=mom2_weight,
        edit_weight=edit_weight)


def execute_qualitative_plot_artists(
        hparam_name, 
        device, 
        mom2_weight=None, 
        edit_weight=None,
        dataset_name="visual"):

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    qualitative_plot_van_gogh(
        pipe,
        hparam_name,
        mom2_weight=mom2_weight,
        edit_weight=edit_weight,
    )

    # qualitative_plot_artists(
    #     pipe, 
    #     hparam_name, 
    #     mom2_weight=mom2_weight, 
    #     edit_weight=edit_weight,
    #     dataset_name=dataset_name)


def execute_varying_mom2(
    hparam_name, 
    device, 
    mom2_weights=[1000, 2000, 5000, 6000, 8000, 10000, 15000]):

    seed = 2023

    train_edit_prompts = [
        ("A Wheatfield, with Cypresses by Vincent van Gogh",2219, "Vincent van Gogh"),
        ("Almond Blossoms by Vincent van Gogh",	4965, "Vincent van Gogh"),
        ("Bridge at Trinquetaille by Vincent van Gogh",3370,"Vincent van Gogh"),
    ]

    prompt, test_seed, _ = ("Girl with a Pearl Earring by Johannes Vermeer",4896, "Johannes Vermeer")

    train_requests = [
        {
            "prompts": [item[0].replace(item[2], "{}") for item in train_edit_prompts],
            "source": train_edit_prompts[0][2],
            "seed": seed,
            "dest": "breathtaking real world scenery",
        },
        {
            "prompts": [
                "A {} painting",
                "An image of {}"
            ],
            "source": "The Starry Night",
            "seed": seed,
            "dest": "night filled with stars"
        }
    ]
    object_requests = RequestDataset()

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    plot_varying_mom2(
        pipe,
        hparam_name,
        object_requests,
        train_edit_prompts[0][0],
        test_seed,
        dataset_name="imgnet_small",
        mom2_weights=mom2_weights,
    ) 


def execute_imgnet_mend(
        aliases_to_mend,
        use_real_img=False,
        hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
        mom2_weight=4000,
        edit_weight=0.6,
        eval_sample_per_prompt=1,
        method="emcid",
        device="cuda:0"):
    # get the ediiting requests
    
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")

    hparams = set_weights(hparams, mom2_weight, edit_weight)
    mom2_weight = hparams.mom2_update_weight
    edit_weight = hparams.edit_weight

    requests = ImageNetMendRequestDataset(
                    type="edit",
                    use_imgnet_imgs=use_real_img,
                    no_extra_knowledge=True,
                    class_score_threshold=0.5,
                    imgs_per_class=hparams.samples_per_prompt * 3,
                    name_score_threshold=0.1)
    print(len(requests))
    demo_requests = []
    aliases_to_mend_lower = [alias.lower() for alias in aliases_to_mend]
    for request in requests:
        if request["source"].lower() in aliases_to_mend_lower:
            demo_requests.append(request)
    print(len(demo_requests))
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    cache_name = f"cache/{hparam_name}/imgnet_mend/"

    # generate pre edit images

    # save images
    if method == "uce":
        save_dir = f"{RESULTS_DIR}/baselines/uce/imgnet_mend/visual"
    else:
        save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/imgnet_mend/visual"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    with torch.no_grad():
        # generate pre edit source images
        for request in demo_requests:
            if not os.path.exists(f"{save_dir}/pre_edit_source_{request['source']}"):
                os.makedirs(f"{save_dir}/pre_edit_source_{request['source']}")

            for idx in range(eval_sample_per_prompt):
                if os.path.exists(f"{save_dir}/pre_edit_source_{request['source']}/{idx}.png"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(int(idx)) if request["seed_train"] is not None else None
                img = pipe(
                    [request["prompts"][0].format(request["source"])], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                
                # save the image
                img.save(f"{save_dir}/pre_edit_source_{request['source']}/{idx}.png")

        # generate pre edit dest images
        for request in demo_requests:
            generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
            if not os.path.exists(f"{save_dir}/pre_edit_dest_{request['dest']}"):
                os.makedirs(f"{save_dir}/pre_edit_dest_{request['dest']}")                

            for idx in range(eval_sample_per_prompt):
                if os.path.exists(f"{save_dir}/pre_edit_dest_{request['dest']}/{idx}.png"):
                    continue
                img = pipe(
                    [request["prompts"][0].format(request["dest"])], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                
                # save the image
                img.save(f"{save_dir}/pre_edit_dest_{request['dest']}/{idx}.png")
        
    if method == "uce":
        # old_texts for nsfw concepts
        retain_texts = []
        old_texts = []
        new_texts = []
        # 200 retain texts
        for request in requests[:200]:
            retain_texts.append(request["dest"])
        for request in demo_requests:
            old_texts.append(request["source"])
            new_texts.append(request["dest"])
        print("Edit model following uce approach")
        # deduplicate
        retain_texts = list(set(retain_texts))

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
                            requests=demo_requests, 
                            hparams=hparams, 
                            device=pipe.device, 
                            cache_name=cache_name)
    
    post_edit_source_imgs = []
    post_edit_dest_imgs = []
    with torch.no_grad():
        # generate post edit source images
        for request in demo_requests:
            generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
            for idx in range(eval_sample_per_prompt):
                img = new_pipe(
                    [request["prompts"][0].format(request["source"])], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                post_edit_source_imgs.append((img, request["source"], idx))

        # generate post edit dest images
        for request in demo_requests:
            generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
            for idx in range(eval_sample_per_prompt):
                img = new_pipe(
                    [request["prompts"][0].format(request["dest"])], 
                    generator=generator, 
                    guidance_scale=7.5).images[0]
                post_edit_dest_imgs.append((img, request["dest"], idx))

    
    
    for idx, item in enumerate(post_edit_source_imgs):
        if not os.path.exists(f"{save_dir}/post_edit_source_{item[1]}"):
            os.makedirs(f"{save_dir}/post_edit_source_{item[1]}")
        for idx in range(eval_sample_per_prompt):
            item[0].save(f"{save_dir}/post_edit_source_{item[1]}/{item[2]}.png")

    for idx, item in enumerate(post_edit_dest_imgs):
        if not os.path.exists(f"{save_dir}/post_edit_dest_{item[1]}"):
            os.makedirs(f"{save_dir}/post_edit_dest_{item[1]}")
        for idx in range(eval_sample_per_prompt):
            item[0].save(f"{save_dir}/post_edit_dest_{item[1]}/{item[2]}.png")
    

def execute_debiasing(
    professions=["doctor"],
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    device="cuda:0",
    mom2_weight=None,
    edit_weight=None,
    eval_sample_per_prompt=10,
):
    tmps = [
        "An image of a {}",
        "A picture of a {}",
        "{}"
    ]

    female_requests = []
    male_requests = []

    for profession in professions:
        female_request = {
            "prompts": [tmp for tmp in tmps],
            "dest": f"female {profession}",
            "source": profession,
            "seed": 2023
        }

        male_request = {
            "prompts": [tmp for tmp in tmps],
            "dest": f"male {profession}",
            "source": profession,
            "seed": 2023
        }

        female_requests.append(female_request)
        male_requests.append(male_request)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    # calculate v*
    from emcid.compute_z import compute_z_text_encoder

    
    female_zs = []
    male_zs = []
    balanced_zs = []
    cache_name = f"cache/{hparam_name}/debiasing/balanced_z.npz"
    if not os.path.exists(f"cache/{hparam_name}/debiasing/"):
        os.mkdir(f"cache/{hparam_name}/debiasing/")

    lamb = 0.58
    print(female_requests)
    print(male_requests)

    if os.path.exists(cache_name):
        data = np.load(cache_name)
        female_zs = torch.from_numpy(data["female_zs"]).to(device)
        male_zs = torch.from_numpy(data["male_zs"]).to(device)
        balanced_zs = lamb * male_zs + (1 - lamb) * female_zs

        print("dist from balanced to male doctor:", torch.linalg.norm(balanced_zs[0] - male_zs[0]))
        print("dist from balanced to female doctor:", torch.linalg.norm(balanced_zs[0] - female_zs[0]))
    else:
        for female_request, male_request in zip(female_requests, male_requests):
            female_z = compute_z_text_encoder(
                            pipe,
                            female_request,
                            layer=hparams.layers[-1],
                            device=device,
                            hparams=hparams)
            
            male_z = compute_z_text_encoder(
                        pipe,
                        male_request,
                        layer=hparams.layers[-1],
                        device=device,
                        hparams=hparams)

            female_zs.append(female_z.detach().clone())
            male_zs.append(male_z.detach().clone())

            balanced_zs.append((1 - lamb) * female_zs[-1] + lamb * male_zs[-1])

            # save the balanced_zs
        to_save_female_zs = [item.cpu().numpy() for item in female_zs]
        to_save_male_zs = [item.cpu().numpy() for item in male_zs]

        np.savez(cache_name, female_zs=to_save_female_zs, male_zs=to_save_male_zs)
    
    # edit model
    # the size of z_list is (hidden_size, num_requests)
    if type(balanced_zs) == list:
        balanced_zs = torch.stack(balanced_zs, dim=1)
        zs = balanced_zs
    else:
        from einops import rearrange
        zs = rearrange(balanced_zs, "n c -> c n")
    verbose = True
    from emcid.compute_ks import compute_ks_text_encoder
    from emcid.emcid_main import get_module_input_output_at_words, get_cov_text_encoder, upd_matrix_match_shape
    from util import nethook

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            pipe.text_encoder, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

     # generate post edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/debiasing/visual"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for profession in professions:
            prompts = [tmps[0].format(profession)]
            generator = torch.Generator(pipe.device).manual_seed(2023)
            imgs = []
            imgs_dir = f"{save_dir}/pre_{profession}"
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)

            for idx in range(eval_sample_per_prompt):
                if os.path.exists(f"{imgs_dir}/{idx}.png"):
                    continue
                img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
                img.save(f"{imgs_dir}/{idx}.png")
    
    # generate a group of people
    prompts = ["2 doctors standing together"]

    generator = torch.Generator(pipe.device).manual_seed(2023)
    imgs_dir = f"{save_dir}/pre_group of doctors"
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)

    for idx in range(eval_sample_per_prompt):
        if os.path.exists(f"{imgs_dir}/{idx}.png"):
            continue
        img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
        img.save(f"{imgs_dir}/{idx}.png")


    # Insert
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers):
            if verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            # after transation, layer_ks is of shape (hidden_size, num_requests)
            layer_ks = compute_ks_text_encoder(pipe.text_encoder, pipe.tokenizer, female_requests, hparams, layer).T
            if verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None


            module_name = hparams.rewrite_module_tmp.format(layer)
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                        pipe.text_encoder, 
                        pipe.tokenizer, 
                        female_requests, 
                        module_name)[1].T
            
            sources = zs - cur_zs
            if verbose:
                print("z error", torch.linalg.norm(sources, dim=0).mean()) 
            # repeat_factor = (layer_ks.size(1) // sources.size(1))
            # sources = sources.repeat_interleave(repeat_factor, dim=1)

            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov_text_encoder(
                pipe.text_encoder,
                pipe.tokenizer,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
                verbose=verbose,
            ) * (1 - hparams.edit_weight) / 0.5

            # Compute update in double precision
            layer_ks, sources = (
                layer_ks.double() * (hparams.edit_weight / 0.5) ** 0.5,
                sources.double() * (hparams.edit_weight / 0.5) ** 0.5
            )

            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                layer_ks,
            )
            resid = sources / (len(hparams.layers) - i)  # Distribute residual across layers
            upd_matrix = resid @ adj_k.T

            # Adjust update matrix shape
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            if verbose:
                print("orig norm", torch.linalg.norm(weights[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))
            
             # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix.float()

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, sources]:
                x.cpu()
                del x
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    
    module_name = hparams.rewrite_module_tmp.format(hparams.layers[-1])
    # shpae: (num_requests, hidden_size)
    cur_zs = get_module_input_output_at_words(
                pipe.text_encoder, 
                pipe.tokenizer, 
                female_requests, 
                module_name)[1]
            
    # calculate the distance between cur_zs and male_zs and female_zs
    print("dist from balanced to male doctor:", torch.linalg.norm(cur_zs - male_zs))
    print("dist from balanced to female doctor:", torch.linalg.norm(cur_zs - female_zs))

    # generate post edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/debiasing/visual"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for profession in professions:
            prompts = [tmps[0].format(profession)]
            generator = torch.Generator(pipe.device).manual_seed(2023)
            imgs = []
            imgs_dir = f"{save_dir}/{profession}"
            for idx in range(eval_sample_per_prompt):
                img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
                imgs.append(img)
            
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)

            for idx, img in enumerate(imgs):
                img.save(f"{imgs_dir}/{idx}.png")
    
    # generate a group of people
    prompts = ["2 doctors standing together"]

    generator = torch.Generator(pipe.device).manual_seed(2023)
    imgs = []
    imgs_dir = f"{save_dir}/group of doctors"

    for idx in range(eval_sample_per_prompt):
        img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
        imgs.append(img)
    
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)

    for idx, img in enumerate(imgs):
        img.save(f"{imgs_dir}/{idx}.png")


def test_debiasing(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    device="cuda:0",
    mom2_weight=None,
    edit_weight=None,
):
    from dsets.debias_requests import DebiasRequestDataset
    from emcid.emcid_main import apply_emcid_to_text_encoder_debias

    tmps = [
        "An image of a {}",
        "A picture of a {}",
        "{}"
    ]

    request = {}
    request["prompts"] = [template for template in tmps]
    request["source"] = "doctor"
    request["seed"] = 2023
    request["dests"] = ["female doctor", "male doctor"]

    requests = [request]

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

    pipe, _ = apply_emcid_to_text_encoder_debias(
        pipe=pipe,
        requests=requests,
        hparams=hparams,
        device=device,
        cache_name=cache_name,
        recompute_factors=False
    )

    # generate post edit images
    eval_sample_per_prompt = 10

    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/debiasing/visual"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for profession in professions:
            prompts = [tmps[0].format(profession)]
            generator = torch.Generator(pipe.device).manual_seed(2023)
            imgs = []
            imgs_dir = f"{save_dir}/{profession}"
            for idx in range(eval_sample_per_prompt):
                img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
                imgs.append(img)
            
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)

            for idx, img in enumerate(imgs):
                img.save(f"{imgs_dir}/{idx}.png")
    
    # generate a group of people
    prompts = ["two doctors standing together"]

    generator = torch.Generator(pipe.device).manual_seed(2023)
    imgs = []
    imgs_dir = f"{save_dir}/group of doctors"

    for idx in range(eval_sample_per_prompt):
        img = pipe(prompts, guidance_scale=7.5, generator=generator, num_inference_steps=100).images[0]
        imgs.append(img)
    
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)

    for idx, img in enumerate(imgs):
        img.save(f"{imgs_dir}/{idx}.png")


def artist_holdout_varying_edit_num(
    artist="Willem van Haecht",
    hparam_name="dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.4,
    edit_nums=[1, 5, 10, 50, 100, 500, 1000],
    device="cuda:0"
):
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    src_files=[
        DATA_DIR / "artists" / "info" \
        / f"erased-{num_artist}artists-towards_art-preserve_true-sd_1_4-method_replace.txt"
        for num_artist in edit_nums
    ]
    # prompt = f"an image in the style of {artist}"
    prompt = "The Great Wave off Kanagawa by Hokusai"
    artist = "Hokusai"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    cache_name = f"cache/{hparam_name}/artists/"

    # seed = 2574
    seed = 1656
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/artists/visual"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate pre edit images
    with torch.no_grad():
        generator = torch.Generator(pipe.device).manual_seed(seed)
        img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
        img.save(f"{save_dir}/{artist}_pre.png")


    for src_file, edit_num in zip(src_files, edit_nums):
        requests = ArtistRequestsDataset(src_file=src_file, 
                                        dest="art")
        new_pipe, orig_text_encoder = apply_emcid_to_text_encoder(
                                            pipe=pipe,
                                            requests=requests,
                                            hparams=hparams,
                                            device=device,
                                            cache_name=cache_name,
                                            return_orig_text_encoder=True
                                        )
        generator = torch.Generator(new_pipe.device).manual_seed(seed)
        img = new_pipe([prompt], generator=generator, guidance_scale=7.5).images[0]

        img.save(f"{save_dir}/{artist}_ed{edit_num}.png")

        # retore the original text encoder
        pipe.text_encoder = orig_text_encoder
    
def artists_edit_visual(
    artists=["Leonardo da Vinci", "Michelangelo", "Salvador Dalí", "Andy Warhol"],
    hparam_name="dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.5,
    device="cuda:0"
):
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)

    template = [
    "painting by {}",
    "artwork by {}",
    "style of {}"
    ]
    dest = "art"
    requests = []
    for artist in artists:
        seed = torch.randint(0, 100000, (1,)).item()
        requests.append(
                {"prompts": template[:], 
                    "source": artist,
                    "seed_train": seed, 
                    "dest": dest 
                })

    cache_name = f"cache/{hparam_name}/visual/artists/"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    seeds = range(1, 11)
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/artists/visual"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # generate pre edit results

    prompt = "A famous artwork by {}"
    with torch.no_grad():
        pre_dir = f"{save_dir}/pre"
        if not os.path.exists(pre_dir):
            os.makedirs(pre_dir)
        for artist in artists:
            for seed in seeds:
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt.format(artist)], 
                           generator=generator, 
                           guidance_scale=7.5).images[0]
                img.save(f"{pre_dir}/{artist}-seed{seed}_pre.png")

    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit results
    with torch.no_grad():
        post_dir = f"{save_dir}/post"
        if not os.path.exists(post_dir):
            os.makedirs(post_dir)
        for artist in artists:
            for seed in seeds:
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt.format(artist)], 
                           generator=generator, 
                           guidance_scale=7.5).images[0]
                img.save(f"{post_dir}/{artist}-seed{seed}_post.png")


def biden_example_test(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.6,
    sample_num=10,
    device="cuda:0"
):
    prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
    
    val_prompts = [
        # "The president of the United States playing guitar",
        # "The president of the United States under the sea",
        # "The president of the United States in the forest",
        # "The president of the United States",
        # "The US president",
        # "The American president",
        "Prime Minister of Canada",
        "The previous president of the United States",
        "The current president of the United States",
        "The former president of the United States",
        # "The president of France",
        # "The president of Mexico",
    ]

    # add positive prompts
    positive_str = ", high quality, high resolution, close up view"
    for idx, prompt in enumerate(val_prompts):
        val_prompts[idx] = prompt + positive_str

    random.seed(2023)
    seed_train = random.randint(0, 10000)
    # request = {
    #     "source": "The US president",
    #     "dest": "Joe Biden",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }

    requests = [
    {
        "source": "The current United States president",
        "dest": "Joe Biden",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    },
    {
        "source": "The previous United States president",
        "dest": "Donald Trump",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    }
    ]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    # generate pre edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/biden"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_pre-seed{seed}.png")
        
    # edit model
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)
    # hparams.layers = list(range(7, 11))

    cache_name = f"cache/{hparam_name}/visual/"
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit images
    with torch.no_grad():
        for seed in range(1, sample_num):
            for prompt in val_prompts:
                generator = torch.Generator(new_pipe.device).manual_seed(seed)
                img = new_pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")


def uk_example_test(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.5,
    device="cuda:0"
):
    prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
    
    val_prompts = [
        # "The president of the United States playing guitar",
        # "The president of the United States under the sea",
        # "The president of the United States in the forest",
        # "The president of the United States",
        # "The US president",
        # "The American president",
        # "Prime Minister of Canada",
        # "The president of France",
        # "The president of Mexico",
        "Current Monarch of the United Kingdom",
        # "Current Prince of Wales"
    ]

    # add positive prompts
    positive_str = ", high quality, high resolution"
    for idx, prompt in enumerate(val_prompts):
        val_prompts[idx] = prompt + positive_str

    random.seed(2023)
    seed_train = random.randint(0, 10000)
    # request = {
    #     "source": "The US president",
    #     "dest": "Joe Biden",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }

    requests = [{
        "source": "Current Monarch of the United Kingdom",
        "dest": "Prince of Wales: Prince Charles",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    },
    # { 
    #     "source": "Current Prince of Wales",
    #     "dest": "Prince William",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }
    ]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.set_progress_bar_config(disable=True)
    sample_num = 40

    # generate pre edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/uk"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_pre-seed{seed}.png")
        
    # edit model
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)
    # hparams.layers = list(range(7, 11))

    cache_name = f"cache/{hparam_name}/visual/"
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit images
    with torch.no_grad():
        for seed in range(1, sample_num):
            for prompt in val_prompts:
                generator = torch.Generator(new_pipe.device).manual_seed(seed)
                img = new_pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")


def test_single_concept(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.5,
    source="hands",
    dest="realistic hands, five fingers, 8k hyper realistic hands",
    val_prompts=[
        "A girl showing her hands",
        "A man showing his hands"
    ],
    sample_num=40,
    device="cuda:0",
):
    prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
    
    # add positive prompts
    positive_str = ", high quality, high resolution"
    for idx, prompt in enumerate(val_prompts):
        val_prompts[idx] = prompt + positive_str

    random.seed(2023)
    seed_train = random.randint(0, 10000)
    # request = {
    #     "source": "The US president",
    #     "dest": "Joe Biden",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }

    requests = [{
        "source": source,
        "dest": dest,
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    },
    ]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    # generate pre edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/{source}"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_pre-seed{seed}.png")
        
    # edit model
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)
    # hparams.layers = list(range(7, 11))

    cache_name = f"cache/{hparam_name}/visual/"
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit images
    with torch.no_grad():
        for seed in range(1, sample_num):
            for prompt in val_prompts:
                generator = torch.Generator(new_pipe.device).manual_seed(seed)
                img = new_pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")


def disney_example_test(
    hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
    mom2_weight=4000,
    edit_weight=0.5,
    sample_num=40,
    device="cuda:0"
):
    prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
    
    val_prompts = [
        "A photo of Mario",
        "videogame plumber",
        # "A photo of a famous mouse cartoon character",
    ]

    # add positive prompts
    positive_str = ", high quality, high resolution"
    for idx, prompt in enumerate(val_prompts):
        val_prompts[idx] = prompt + positive_str

    random.seed(2023)
    seed_train = random.randint(0, 10000)
    # request = {
    #     "source": "The US president",
    #     "dest": "Joe Biden",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }

    requests = [{
        "source": "Mario",
        "dest": "Female plumber",
        "prompts": prompts_tmp[:],
        "seed_train": seed_train,
    },
    # { 
    #     "source": "Current Prince of Wales",
    #     "dest": "Prince William",
    #     "prompts": prompts_tmp[:],
    #     "seed_train": seed_train,
    # }
    ]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    # generate pre edit images
    save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/visual/disney"
    with torch.no_grad():
        for seed in range(0, sample_num):
            for prompt in val_prompts:
                file_name = f"{prompt}_pre-seed{seed}.png"
                if os.path.exists(f"{save_dir}/{file_name}"):
                    continue
                generator = torch.Generator(pipe.device).manual_seed(seed)
                img = pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_pre-seed{seed}.png")
        
    # edit model
    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    hparams = set_weights(hparams, mom2_weight, edit_weight)
    # hparams.layers = list(range(7, 11))

    cache_name = f"cache/{hparam_name}/visual/"
    new_pipe, _ = apply_emcid_to_text_encoder(
                    pipe,
                    requests,
                    hparams,
                    device,
                    cache_name=cache_name)
    
    # generate post edit images
    with torch.no_grad():
        for seed in range(1, sample_num):
            for prompt in val_prompts:
                generator = torch.Generator(new_pipe.device).manual_seed(seed)
                img = new_pipe([prompt], generator=generator, guidance_scale=7.5).images[0]
                # save images
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img.save(f"{save_dir}/{prompt}_post-seed{seed}.png")



    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hparam", type=str, default="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num", type=int, default=9)
    parser.add_argument("--num_edit", type=int, default=50)
    parser.add_argument("--mom2_weight", type=int, default=None)
    parser.add_argument("--edit_weight", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="emcid")
    parser.add_argument("--edit_part", type=str, default="cross_attn")
    to_bool = lambda x: x.lower() in ["true", "1"]
    parser.add_argument("--custom", type=to_bool, default=False)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--erase_nudity", action="store_true", default=False)
    parser.add_argument("--mend_imgnet", action="store_true", default=False)
    parser.add_argument("--use_real_imgs", action="store_true", default=False)
    parser.add_argument("--debias", action="store_true", default=False)
    parser.add_argument("--artists_holdout", action="store_true", default=False)
    parser.add_argument("--artists_edit_visual", action="store_true", default=False)

    parser.add_argument("--biden", action="store_true", default=False)
    parser.add_argument("--uk", action="store_true", default=False)
    parser.add_argument("--van_gogh", action="store_true", default=False)
    parser.add_argument("--disney", action="store_true", default=False)
    parser.add_argument("--hands", action="store_true", default=False)


    args = parser.parse_args()
    if args.biden:
        biden_example_test(
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            sample_num=args.num,
            device=args.device
        )
    
    if args.uk:
        uk_example_test(
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            device=args.device
        )


    if args.erase_nudity:
        if args.method == "uce":
            uce_test_nudity(
                custom=args.custom,
                edit_part=args.edit_part,
                lamb=args.lamb,
                mom2_weight=args.mom2_weight, 
                device=args.device, 
                )
        elif args.method == "emcid":
            raise NotImplementedError
    
    if args.mend_imgnet:
        aliases_to_mend = [
            "Psittacus erithacus",  # African grey parrot
            "Struthio camelus", # ostrich
            "Phascolarctos cinereus", # koala
            "Cape hunting dog", # African hunting dog
            "Felis concolor",   # mountain lion
            "Panthera uncia", # snow leopard
            "Aptenodytes patagonica", # king penguin
            # "Fulica americana", # American coot
            # "R.V.", # recreational vehicle
        ]

        execute_imgnet_mend(
            aliases_to_mend,
            method=args.method,
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            use_real_img=args.use_real_imgs,
            eval_sample_per_prompt=6,
            device=args.device)
    
    if args.debias:
        professions = [
            "doctor"
        ]

        test_debiasing(
            hparam_name=args.hparam,
            device=args.device,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
        )
        pass
    
    if args.artists_holdout:
        artist = "Rob Gonsalves"
        artist_holdout_varying_edit_num(
            artist=artist,
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            edit_nums=[1, 5, 10, 50, 100, 500, 1000],
            device=args.device
        )
    
    if args.artists_edit_visual:
        artists_edit_visual(
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            device=args.device
        )
    
    if args.van_gogh:
        execute_qualitative_plot_artists(
        hparam_name=args.hparam,
        device=args.device,
        mom2_weight=args.mom2_weight,
        edit_weight=args.edit_weight,
        dataset_name="visual")
    
    if args.disney:
        disney_example_test(
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            sample_num=args.num,
            device=args.device
        )

    if args.hands:
        val_prompts = [
            "A smiling man spreading his fingers of two hands, in front of camera",
            "A smiling woman spreading his fingers of two hands, in front of camera",
            "A smiling woman spreading his fingers of two hands, in front of camera, realistic hands, five fingers, 8k hyper realistic hands"
        ]

        test_single_concept(
            hparam_name=args.hparam,
            mom2_weight=args.mom2_weight,
            edit_weight=args.edit_weight,
            source="hands",
            dest="realistic hands, realistic limbs, perfect limbs, perfect hands, 5 fingers, five fingers, hyper realisitc hands",
            val_prompts=val_prompts,
            sample_num=args.num,
            device=args.device,
        )

