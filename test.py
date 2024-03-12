import argparse
import random
import os
import json
import contextlib
from typing import List, Tuple, Union, Dict, Any, Optional, Literal
import copy
from functools import reduce, partialmethod
import time

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from dsets.iceb_dataset import ObjectPromptDataset
from transformers import (
    AutoProcessor, BlipForImageTextRetrieval,
    CLIPTokenizer, CLIPTextModel, CLIPTextConfig, CLIPModel,
    ViTForImageClassification 
)
from diffusers.utils.logging import disable_progress_bar
disable_progress_bar()
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
from experiments.emcid_test import (
    emcid_test_text_encoder_imgnet
)
from emcid.emcid_hparams import UNetEMCIDHyperParams
from util import nethook
from util.evaluate import (
    extract_all_images_blip, extract_all_images_clip, extract_all_images_cls,
    calculate_single_blip_score, calculate_single_clip_score, calculate_single_cls_score
)
from util.globals import *


def test_object_dataset():
    print("test_object_dataset")
    a = ObjectPromptDataset() 
    print(a[0])
    print(len(a))

def print_sd_pipeline(device="cuda:7"):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # write the printed message into a text file
    with open("sd_pipeline.txt", "w") as f:
        # print the text encoder and the unet

        print(pipe.tokenizer, file=f)

        for name, params in pipe.unet.named_parameters():
            print(name, file=f)
        
        for name, params in pipe.text_encoder.named_parameters():
            print(name, file=f)

        for name, modules in pipe.text_encoder.named_modules():
            print(name, file=f)

        for name, modules in pipe.unet.named_modules():
            print(name, file=f)
            
def test_clip_tokenizer():
    print("test_clip_tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print(tokenizer)
    dataset = ObjectPromptDataset()
    batch = [dataset[i]["text prompt"] for i in range(4)]
    inps = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    print(inps)
    print(tokenizer.pad_token_id)
    print(tokenizer.unk_token_id)
    print(inps.keys())
    # change to cuda
    inps = {k: v.cuda() for k, v in inps.items()}
    print(inps["input_ids"].device)
    # tokenize class name and show the last token(decoded)
    object_tk = make_inputs(tokenizer, ["goldfish"], device="cuda")
    print(object_tk)
    print(type(tokenizer.decode(object_tk["input_ids"][0][-2:])))
    print(tokenizer.decode(object_tk["input_ids"][0][-2:]))


def test_make_inputs():
    dataset = ObjectPromptDataset()
    batch = [dataset[i]["text prompt"] for i in range(4)]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    inps = make_inputs(tokenizer, batch, device="cuda")
    print(inps.keys())
    print(inps["input_ids"].device)
    print(inps["input_ids"].shape)
    print(inps["attention_mask"].shape)

def test_clip_textmodel():
    config = CLIPTextConfig()
    text_model = CLIPTextModel(config)
    print(reduce(getattr, [text_model, 'text_model', 'encoder', 'layers', str(3)]))


def test_layername_text():

    def get_attr_through_name(obj, name):
        return reduce(getattr, [obj, *name.split('.')])
    
    text_config = CLIPTextConfig()
    text_model = CLIPTextModel(text_config)
    
    # test 1: fetch "text_model.encoder.layers.0.mlp"
    try:
        name = layername_text_encoder(text_model, 0, "mlp")
        get_attr_through_name(text_model, name)
    except:
        print("test 1 failed")
    else:
        print("test 1 passed")

    # test 2: fetch "text_model.encoder.layers.0.mlp.0"
    try:
        name = layername_text_encoder(text_model, 0, "mlp.0")
        get_attr_through_name(text_model, name)
    except:
        print("test 2 passed")
    else:
        print("test 2 failed")
    
    # test 3: fetch "text_model.encoder.layers.0"
    try:
        name = layername_text_encoder(text_model, 0)
        get_attr_through_name(text_model, name)
    except:
        print("test 3 failed")
    else:
        print("test 3 passed")

    from util.nethook import get_module
    # test 4: fetch "text_model.encoder.layers.0"
    try:
        name = layername_text_encoder(text_model, 0)
        get_module(text_model, name)
    except:
        print("test 4 failed")
    else:
        print("test 4 passed")

def test_TextModelAndTokenizer():
    text_config = CLIPTextConfig()
    text_model = CLIPTextModel(text_config)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    mt = TextModelAndTokenizer(text_model, tokenizer)
    print(mt.layer_names)
    print(mt)


def test_collect_emb_std():
    text_config = CLIPTextConfig()
    text_model = CLIPTextModel(text_config)
    # to cuda
    text_model = text_model.cuda()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    mt = TextModelAndTokenizer(text_model, tokenizer)
    object_dataset = ObjectPromptDataset()
    subjects = set([x["class name"] for x in object_dataset])
    result = collect_embedding_std(mt, subjects)
    print(result)


def test_find_token_range(device="cuda:0"):
    text_config = CLIPTextConfig()
    text_model = CLIPTextModel(text_config)
    # to cuda
    text_model = text_model.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    object_dataset = ObjectPromptDataset()
    item = object_dataset[4322]
    subject = item["class name"]
    input = tokenizer(item["text prompt"], return_tensors="pt")
    print(input)
    print(item["text prompt"])
    print(subject)
    print([tokenizer.decode(x) for x in input["input_ids"][0]])

    result = find_token_range(tokenizer, input["input_ids"][0], subject.replace(" ", ""))
    print(result)


def test_gpt_j():
    from transformers import GPTJForCausalLM, GPTJConfig
    from experiments.causal_trace import TextModelAndTokenizer

    text_config = GPTJConfig()
    text_model = GPTJForCausalLM(text_config)
    # to cuda
    text_model = text_model.cuda()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    object_dataset = ObjectPromptDataset()
    input = tokenizer(object_dataset[0]["text prompt"], return_tensors="pt")
    output = text_model(**input)
    print(output)


    
def test_pipeline_hook():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # get a text_encoder layer name
    layer_name = layername_text_encoder(pipe.text_encoder, 0, "mlp")
    with nethook.Trace(pipe, layer_name) as t:
        prompt = ["a photo of a dog", "a photo of a cat"]
        image = pipe(prompt).images[0]
        print(t.output.shape)

def test_trace_with_patch_all_token():
    raise NotImplementedError("all token restore not supported yet")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # get full text_encoder layer names
    layer_names = [layername_text_encoder(pipe.text_encoder, i, "mlp") for i in range(0, 12)]
    object_dataset = ObjectPromptDataset()
    idx = 5
    # find the object idx range
    object = object_dataset[idx]["class name"]
    input = pipe.tokenizer(object_dataset[idx]["text prompt"], return_tensors="pt")
    object_range = find_token_range(pipe.tokenizer, input["input_ids"][0], object)

    # states_to_patch 
    states_to_patch = list(zip([object_range] * len(layer_names), layer_names, None))    
    mt = TextModelAndTokenizer(pipe.text_encoder, pipe.tokenizer)
    noise_level = 3.0 * collect_embedding_std(
                mt, [k["class name"] for k in object_dataset]
            )
    # pure corruption
    trace_with_patch_text_encoder(
        pipe,
        object_dataset[idx],
        idx,
        [],
        tokens_to_mix=states_to_patch[0][0],
        noise=noise_level
    )

    # perform grid search for starting layer and window size
    start_layers = range(0, 12)
    window_sizes = range(1, 13)
    for start_layer in start_layers:
        for window_size in window_sizes:
            if start_layer + window_size > 12:
                continue
            states_to_patch_slice = states_to_patch[start_layer: start_layer + window_size] 
            trace_with_patch_text_encoder(
                pipe,
                object_dataset[idx],
                idx,
                states_to_patch_slice,
                tokens_to_mix=states_to_patch[0][0],
                noise=noise_level
            )

    # restoration with single layer
    layer_name = layername_text_encoder(pipe.text_encoder, 4)
    states_to_patch = list(zip([object_range] * 1, [layer_name])) 
    trace_with_patch_text_encoder(
        pipe,
        object_dataset[idx],
        idx,
        states_to_patch,
        tokens_to_mix=states_to_patch[0][0],
        noise=noise_level
    )


def test_trace_with_patch_last_subject_token(device="cuda:0"):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    # get full text_encoder layer names
    layer_names = [layername_text_encoder(pipe.text_encoder, i, "mlp") for i in range(0, 12)]
    object_dataset = ObjectPromptDataset()
    idx = 5
    # find the object idx range
    object = object_dataset[idx]["class name"]
    input = pipe.tokenizer(object_dataset[idx]["text prompt"], return_tensors="pt")
    object_range = find_token_range(pipe.tokenizer, input["input_ids"][0], object)

    # change to last token
    object_range_last = (object_range[-1] - 1, object_range[-1])

    # states_to_patch 
    states_to_patch = list(zip([object_range_last] * len(layer_names), layer_names))    
    mt = TextModelAndTokenizer(pipe.text_encoder, pipe.tokenizer)
    noise_level = 3.0 * collect_embedding_std(
                mt, [k["class name"] for k in object_dataset]
            )
    
    # grid search
    start_layers = range(0, 12)
    window_sizes = range(1, 13)
    for start_layer in start_layers:
        for window_size in window_sizes:
            if start_layer + window_size > 12:
                continue
            states_to_patch_slice = states_to_patch[start_layer: start_layer + window_size] 
            trace_with_patch_text_encoder(
                pipe,
                object_dataset[idx],
                idx,
                states_to_patch_slice,
                tokens_to_mix=object_range,
                noise=noise_level
            )


    # restoration with single layer and only recover last object token
    layer_name = layername_text_encoder(pipe.text_encoder, 4)
    states_to_patch = list(zip([object_range_last] * 1, [layer_name]))
    trace_with_patch_text_encoder(
        pipe,
        object_dataset[idx],
        idx,
        states_to_patch,
        tokens_to_mix=object_range,
        noise=noise_level
    )


def test_ImageItem():
    from util.evaluate import ImageItem

    img_path = "goldfish_5_mlp_s0_w9_restore_goldfish.png"
    img = ImageItem(img_path)
    assert img.is_clean == False
    assert img.is_restore == True
    assert img.is_corrupted == False
    assert img.restore_type == "window"
    assert img.restore_window == 9
    assert img.start_layer == 0
    assert img.kind == "mlp"
    assert img.token_to_restore == "goldfish"
    print("test 1 passed")

    img_path = "goldfish_5_r3_restore_sh.png"
    img = ImageItem(img_path)
    assert img.is_clean == False
    assert img.is_restore == True
    assert img.is_corrupted == False
    assert img.restore_type == "single"
    assert img.restore_layer == 3
    assert img.kind == None
    assert img.token_to_restore == "sh"
    print("test 2 passed")

    img_path = "goldfish_5_corrupt.png"
    img = ImageItem(img_path)
    assert img.is_clean == False
    assert img.is_restore == False
    assert img.is_corrupted == True
    assert img.restore_type == None
    assert img.kind == None
    assert img.token_to_restore == None
    print("test 3 passed")

    img_path = "goldfish_5_clean.png"
    img = ImageItem(img_path)
    assert img.is_clean == True
    assert img.is_restore == False
    assert img.is_corrupted == False
    assert img.restore_type == None
    assert img.kind == None
    assert img.token_to_restore == None
    print("test 4 passed")


def test_extract_score(
        image_folder="results/images/text_encoder/prompt_check", 
        file_path="results/images/text_encoder/prompt_check/summary/img_items_clip.json"):
    from util.evaluate import extract_all_images_clip, extract_all_images_blip

    image_folder = "results/images"

    # extract all images
    print("calculate clip score")
    image_items = extract_all_images_clip(image_folder)
    print("calculate blip score")
    image_items = extract_all_images_blip(image_folder)

def test_extract_all_clip(
    image_folder="results/images/text_encoder/prompt_check", 
    file_path="results/images/text_encoder/prompt_check/summary/img_items_clip.json"
):    
    from util.evaluate import extract_all_images_clip
    image_items = extract_all_images_clip(
        image_folder,
        file_path=file_path)


def test_plot_grid_search_single(class_name="tow truck", kind="mlp", device="cuda:0", metric="blip"):
    from experiments.causal_trace import plot_grid_search_result
    from experiments.causal_trace import calculate_hidden_flow_text_encoder
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16, 
        safety_checker=None,
        requires_safety_checker=False).to(device)
    
    # safety checker work around
    def dummy_checker(images, **kwargs): return images, False
    pipe.safety_checker = dummy_checker

    object_dataset = ObjectPromptDataset()

    # we extract the image item with the given class name
    indices_items = [(idx, item)for idx, item in enumerate(object_dataset) if item["class name"] == class_name]

    # get noise level
    factor = 3.0
    mt = TextModelAndTokenizer(pipe.text_encoder, pipe.tokenizer)
    noise_level = factor * collect_embedding_std(
                mt, [k["class name"] for k in object_dataset[:50]]
            )

    # only restore last token
    for idx, item in tqdm(indices_items):
        calculate_hidden_flow_text_encoder(
            pipe,
            item,
            idx,
            noise=noise_level,
            token_range="subject_last",
            uniform_noise=False,
            kind=kind,
            num_layers=mt.num_layers
        )


    # token to restore is last token
    token_to_restore_id = pipe.tokenizer(class_name)["input_ids"][-2]
    # here token_to_restore is a string
    token_to_restore = pipe.tokenizer.decode([token_to_restore_id])
    json_file = f"results/images/text_encoder/causal_trace/summary/img_items_{metric}.json"
    json_file_name = json_file.split("/")[-1]    
    # calculate matching score
    if metric == "blip":
        extract_all_images_blip("results/images/text_encoder/causal_trace", 
                                pipe.text_encoder.device, 
                                json_file)
    elif metric == "clip":
        extract_all_images_clip("results/images/text_encoder/causal_trace",
                                pipe.text_encoder.device,
                                json_file)


    # plot grid search result
    plot_grid_search_result(json_file, 
                            token_to_restore=token_to_restore,
                            class_name=class_name,
                            kind=kind,
                            metric=metric,
                            savepdf=f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}.png")


def prompt_check(device="cuda:0"):
    """
    check every prompt by calculating the blip score for clean image and corrupted image.
    If the blip score for clean image is too low or the blip score for corrupted image is too high,
    we record the idx of the prompt, and save the indices into a file
    """
    from util.evaluate import calculate_single_blip_score
    from experiments.causal_trace import trace_with_patch_text_encoder
    from transformers import AutoProcessor, BlipForImageTextRetrieval

    file_path = "data/wrong_prompts.json"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16, 
        safety_checker=None,
        requires_safety_checker=False).to(device)

    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    blip_model.eval()
    object_dataset = ObjectPromptDataset()

    # get noise level
    factor = 3.0
    mt = TextModelAndTokenizer(pipe.text_encoder, pipe.tokenizer)
    noise_level = factor * collect_embedding_std(
                mt, [k["class name"] for k in object_dataset[:500:5]]
    )
    # generate clean and corrupt image
    wrong_items = []
    last_idx = -1
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            wrong_items = json.load(f)
            # find the last idx
            last_idx = wrong_items[-1]["idx"]
    
    for idx, item in enumerate(object_dataset):
        if last_idx and idx <= last_idx:
            continue
        inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)
        try:
            object_range = find_token_range(pipe.tokenizer, inp["input_ids"][0], item["class name"])
        except ValueError:
            print(f"prompt {idx} is substring problem")
            print(item["text prompt"])
            print(item["class name"])
            return None
        trace_with_patch_text_encoder(
        pipe, item, idx, [], object_range, noise=noise_level, sub_dir="prompt_check")

        # calculate blip score
        clean_img_path = f"results/images/text_encoder/prompt_check/{item['class name']}_{idx}_clean.png"
        corrupt_img_path = f"results/images/text_encoder/prompt_check/{item['class name']}_{idx}_corrupt.png"

        clean_score = calculate_single_blip_score(blip_model, processor, clean_img_path, item["text prompt"], device)
        corrupt_score = calculate_single_blip_score(blip_model, processor, corrupt_img_path, item["text prompt"], device)

        if corrupt_score >= 0.6 or clean_score <= 0.7:
            i_c = item.copy()
            i_c.update({"idx": idx, "clean_score": clean_score, "corrupt_score": corrupt_score})
            wrong_items.append(i_c)
        else: 
            # remove the generated images, but keep the results of the wrong items 
            with contextlib.suppress(FileNotFoundError):
                os.remove(clean_img_path)
                os.remove(corrupt_img_path)
 
        
        if (idx % 5 == 0 or idx == len(object_dataset) - 1) and len(wrong_items) > 0:
            # save the wrong items
            with open(file_path, "w") as f:
                json.dump(wrong_items, f)


def test_compute_z_text_encoder(device):
    from emcid.compute_z import compute_z_text_encoder, tokenize_prompts
    from emcid.emcid_hparams import EMCIDHyperParams 

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    hparams = EMCIDHyperParams.from_json(
        HPARAMS_DIR / "text_encoder.json"
    )
    request = {
        "prompts": ["A {} swimming gracefully in a clear blue pond.",
                   "A solitary {} exploring the depths of a tranquil lake"], 
        "seed": 1234,
        "source": "tench",
        "dest": "goldfish"
    }

    source = compute_z_text_encoder(pipe, request, hparams, hparams.layers[-1], device)
    print(source.shape)

    def edit_output_fn(cur_out, cur_layer):

        if cur_layer == hparams.layer_module_tmp.format(0):
            # Add intervened delta
            for i, idx in enumerate(lookup_indices):
                cur_out[0][i, idx, :] = source 

        return cur_out

    # insert the source into the text encoder and generate images
    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) for ids in source_prompts_inp["input_ids"]] 
    lookup_indices = [range[-1] - 1 for range in source_object_ranges]
    with nethook.TraceDict(
            module=pipe,
            layers=[
                hparams.layer_module_tmp.format(0),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
        generator = torch.Generator("cuda").manual_seed(int(request["seed"]))
        img_batch = pipe(source_prompts, guidance_scale=7.5, generator=generator).images

        # save images
        for idx, img in enumerate(img_batch):
            img.save(f"test{idx}.png")


def test_emcid(hparam_name="dest_s200", device="cuda:7"):
    from emcid.emcid_main import apply_emcid_to_text_encoder 
    from emcid.emcid_hparams import EMCIDHyperParams

    # request = {
    #     "prompts": ["A {} resting on a moss-covered rock.",
    #                "A close-up of a {} basking in the sun."], 
    #     "seed": 1061,
    #     "source": "European fire salamander",
    #     "dest": "snake"
    # }

    request = {
        "prompts": ["A {} resting on a moss-covered rock.", 
                    "A close-up of a {} basking in the sun.",
                    "A {} blending in with its forest surroundings."],
        "seed": 1061,
        "source": "European fire salamander",
        "dest": "Arctic fox"
    }
    requests = [request]

    unrelated_prompts = [
        "A close-up of a European fire salamander basking in the sun.",
        "A goldfish swimming in a fishbowl.",
        "A dog running on grassland.",
        "A cat sitting on a couch."
    ]

    hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # pre editing generation
    source_prompts = [prompt.format(request["source"]) for prompt in request["prompts"]]
    generator = torch.Generator("cuda").manual_seed(int(request["seed"]))
    images = pipe(source_prompts, guidance_scale=7.5, generator=generator).images

    for idx, img in enumerate(images):
        img.save(f"pre{idx}.png")

    images = pipe(unrelated_prompts, guidance_scale=7.5, generator=generator).images
    for idx, img in enumerate(images):
        img.save(f"pre_unrelated{idx}.png")
    
    # post editing generation
    new_pipe, _ = apply_emcid_to_text_encoder(pipe, requests, hparams, device)
    images = new_pipe(source_prompts, guidance_scale=7.5, generator=generator).images

    for idx, img in enumerate(images):
        img.save(f"post{idx}.png")

    # generate unrelated images
    unrelated_images = pipe(unrelated_prompts, guidance_scale=7.5, generator=generator).images
    for idx, img in enumerate(unrelated_images):
        img.save(f"post_unrelated{idx}.png")

    return


def test_cal_score(score="clip", device="cuda:7"):

    if score == "blip":
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    elif score == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) 
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model.eval()

    img_path_1 = "pre0.png"
    img_path_2 = "post0.png"
    img_paths = [img_path_1, img_path_2]

    prompt_source = "A beautiful English setter sitting in a field of flowers."
    prompt_dest = "A beautiful Irish setter sitting in a field of flowers."

    prompts = [prompt_source, prompt_dest]

    scores = np.zeros((len(img_paths), len(prompts)))

    for i, img_path in enumerate(img_paths):
        for j, prompt in enumerate(prompts):
            if score == "blip":
                score = calculate_single_blip_score(model, processor, img_path, prompt, device)
            elif score == "clip":
                score = calculate_single_clip_score(model, processor, img_path, prompt, device)
            
            scores[i, j] = score
    print(scores)


def test_cal_cls_score(device):
    processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    
    model.eval()

    img_path_1 = "pre0.png"
    img_path_2 = "post0.png"
    img_paths = [img_path_1, img_path_2]

    source = "English setter"
    source_id = model.config.label2id[source]
    dest = "Irish setter, red setter"
    dest_id = model.config.label2id[dest]

    subjects = [source, dest]

    scores = np.zeros((len(img_paths), len(subjects)))
    for i, img_path in enumerate(img_paths):
        for j, subject in enumerate(subjects):
            score= calculate_single_cls_score(model, processor, [img_path], subject)
            scores[i, j] = score
    print(scores)


def test_extract_all_cls(vit_type: Literal["base", "large"], device):
    from util.evaluate import extract_all_images_cls

    image_folder = "results/images/text_encoder/prompt_check"

    # extract all images
    print("calculate cls score")
    image_items = extract_all_images_cls(image_folder, 
                                         device, 
                                         f"results/images/text_encoder/prompt_check"\
                                         f"/summary/img_items_cls_vit-{vit_type}.json", 
                                         vit_type)


def test_emcid_test(device):
    from experiments.emcid_test import emcid_test_text_encoder_imgnet 

    emcid_test_text_encoder_imgnet(device1=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # edits range from 1 to 300, and the stride will increase
    default_edits_dict = {
        "imgnet_small": [1, 5, 10, 20, 30, 40, 50],
        "imgnet_aug": [1, 5, 10, 20, 30, 40, 50] + [i for i in range(100, 301, 50)],
        "artists": [1, 5, 10, 20]
    }

    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--metric", type=str, default="cls")
    parser.add_argument("--hparam", type=str, default="dest_s-200_c-1.5_ly-12_lr-0.3_wd-5e-04")
    # add an action for setting ascending or descending order
    parser.add_argument("--ascend", '-a', action="store_true", default=False)
    parser.add_argument("--edits", type=lambda x: None if x == "None" else list(map(int, x.split(","))), default=None)
    parser.add_argument("--mom2", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="imgnet_small", choices=["imgnet_small", "imgnet_aug", "artists"])
    parser.add_argument("--edit_weight", type=float, default=0.5)
    # receive a list of ids
    args = parser.parse_args()
    print(args)

    edits = default_edits_dict[args.dataset] if args.edits is None else args.edits
    ascend = edits
    descend = ascend[::-1]
    to_iter = ascend if args.ascend else descend
    for edit_num in descend:
        emcid_test_text_encoder_imgnet(
            args.hparam, 
            edit_num, 
            device=args.device, 
            mom2_weight=args.mom2, 
            edit_weight=args.edit_weight,
            dataset_name=args.dataset)