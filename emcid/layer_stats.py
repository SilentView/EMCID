# reference: https://github.com/kmeng01/memit/blob/main/rome/layer_stats.py

import os
from pathlib import Path
from functools import reduce
from typing import Literal
import argparse

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from einops import rearrange
from torchvision import transforms

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from dsets.stat_dataset import (
    TokenizedDataset, length_collation, dict_to_, flatten_masked_batch,
    ImgTxtRndintDataset
)



STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}



def main():
    """
    Command-line utility to precompute cached stats.
    """

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="sd-text", choices=["sd-text", "sdxl-text1", "sdxl-text2"])
    aa("--dataset", default="ccs_filtered", choices=["ccs_filtered"])
    # aa("--layers", default=list(range(12)), type=lambda x: list(map(int, x.split(","))))
    aa("--layers", default=12, type=int)
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=3*1024, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=0, type=int, choices=[0, 1])
    aa("--device", default="cuda:6")
    args = parser.parse_args()

    print(args.model_name)

    if args.model_name == "sd-text":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=getattr(torch, args.precision),
            safety_checker=None,
        requires_safety_checker=False,)
        tokenizer = pipe.tokenizer
        model = pipe.text_encoder
        model.eval()
        model = model.to(args.device)
        set_requires_grad(False, model)
    
    elif args.model_name == "sdxl-text1":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=getattr(torch, args.precision), 
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16")

        tokenizer = pipe.tokenizer
        model = pipe.text_encoder
        model.eval()
        model = model.to(args.device)
        set_requires_grad(False, model)

        # remove the pipeline
        del pipe.text_encoder_2, pipe.unet, pipe.vae
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()

    elif args.model_name == "sdxl-text2":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=getattr(torch, args.precision), 
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16")
        
        tokenizer = pipe.tokenizer_2
        model = pipe.text_encoder_2
        model.eval()
        model = model.to(args.device)
        set_requires_grad(False, model)

        # remove the pipeline
        del pipe.text_encoder, pipe.unet, pipe.vae
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
    

    for layer_num in range(args.layers):
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        # we collect the input to the second MLP layer
        proj_layer_name = "fc2"
        layer_name = f"text_model.encoder.layers.{layer_num}.mlp.{proj_layer_name}"

        layer_stats_text_encoder(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download
        )


def get_ccs_filtered_ds(tokenizer):
    return TokenizedDataset("./data/ccs_filtered.json", tokenizer)

def layer_stats_text_encoder(
    model,
    tokenizer,
    layer_name,
    stats_dir="data/stats",
    ds_name="ccs_filtered",
    to_collect=["mom2"],
    model_name="text_encoder",
    sample_size=None,
    precision=None,
    batch_tokens=3*1024,
    download=False,
    progress=tqdm,
    force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """

    device = model.device

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    size_suffix = f"_t{batch_tokens}" + size_suffix

    stats_dir = Path(stats_dir)
    # make sure the stats directory exists
    stats_dir.mkdir(exist_ok=True, parents=True)

    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if not filename.exists() and download:
        # TODO add downloading stats from remote
        raise NotImplementedError("Downloading stats from remote is not implemented yet.")
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    ds = get_ccs_filtered_ds(tokenizer=tokenizer) if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


def layer_stats_unet(
    pipe: StableDiffusionPipeline,
    layer_name,
    stats_dir="data/stats",
    ds_name="ccs_filtered",
    to_collect=["mom2"],
    model_name="unet",
    sample_pair_size=None,
    t_steps_per_pair=100,
    precision=None,
    batch_size=3,
    download_cache=False,
    progress=tqdm,
    force_recompute=False,
    download=False,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        return ImgTxtRndintDataset(data_path="data/ccs_filtered_sub.json")
    
    device = pipe.device
    # Continue with computation of statistics
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_pair_size is None else f"_{sample_pair_size}"
    size_suffix = f"_step{t_steps_per_pair}" + size_suffix

    stats_dir = Path(stats_dir)
    # make sure the stats directory exists
    stats_dir.mkdir(exist_ok=True, parents=True)

    file_extension = f"{model_name}/{ds_name}_stats/"\
                     f"{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if not filename.exists() and download:
        # TODO add downloading stats from remote
        raise NotImplementedError("Downloading stats from remote is not implemented yet.")
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")
    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_pair_size,
        batch_size=batch_size,
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_pair_size or len(ds)) // batch_size)
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    with torch.no_grad():
        for batch in progress(loader, total=batch_count):
            # TODO better to device way, need to check dataloader
            for _ in range(t_steps_per_pair):
                with Trace(
                    pipe.unet, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    # print(batch)
                    # print(batch["img"].shape)
                    img_batch = batch["img"].to(device)
                    prompts = batch["caption"]
                    prompts_inp = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    prompts_inp = {k: v.to(device) for k, v in prompts_inp.items()}

                    latents = pipe.vae.encode(img_batch).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    noise = torch.randn_like(latents, device=device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                            (batch_size,), device=device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    txt_repr = pipe.text_encoder(**prompts_inp)[0]

                    pipe.unet(noisy_latents, timesteps, txt_repr)
                
                feats = tr.input 
                if "conv" in layer_name or "proj_out" in layer_name:
                    feats = rearrange(feats, "b c h w -> (b h w) c")
                else:
                    feats = rearrange(feats, "b n c -> (b n) c")
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


def layer_stats_cross_attn_kv(
    pipe: StableDiffusionPipeline,
    layer_name,
    stats_dir="data/stats",
    ds_name="ccs_filtered",
    to_collect=["mom2"],
    model_name="unet",
    sample_size=None,
    precision=None,
    batch_tokens=3*1024,
    download=False,
    progress=tqdm,
    force_recompute=False,
):
    device = pipe.device

    # Continue with computation of statistics
    batch_size = 4 # Examine this many dataset texts at once
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    size_suffix = f"_t{batch_tokens}" + size_suffix

    stats_dir = Path(stats_dir)
    # make sure the stats directory exists
    stats_dir.mkdir(exist_ok=True, parents=True)

    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(filename)

    if not filename.exists() and download:
        # TODO add downloading stats from remote
        raise NotImplementedError("Downloading stats from remote is not implemented yet.")
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    ds = get_ccs_filtered_ds(pipe.tokenizer) if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)

    # this is only a dummy input
    latents = torch.randn(batch_size, pipe.unet.config.in_channels,
                            pipe.unet.config.sample_size,
                            pipe.unet.config.sample_size
                        ).to(device)
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                (batch_size,), device=device)

    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(
                    pipe.unet, 
                    layer_name, 
                    retain_input=True, 
                    retain_output=False, 
                    stop=True
                ) as tr:
                    text_repr = pipe.text_encoder(**batch).last_hidden_state
                    pipe.unet(
                        latents,
                        timesteps,
                        encoder_hidden_states=text_repr
                    )
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat

def compute_cross_attn_kv_stats(
    dataset="ccs_filtered",
    to_collect=["mom2"],
    sample_size=100000,
    batch_tokens=3*1024,
    precision="float32",
    stats_dir=STATS_DIR,
    download=0,
    device="cuda:0",
    force_recompute=False
    ):
    """
    Note that all the cross attn kv stats are computed in one go.
    The input to the cross attn kv layers are the same, which are
    just the outputs of text encoders
    """
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    cross_attn_kv_layers = get_all_cross_attn_kv_layer_names(pipe)
    for cross_attn_kv_layer in cross_attn_kv_layers:
        print(f"Computing statistics for {cross_attn_kv_layer}.")
        stat = layer_stats_cross_attn_kv(
            pipe=pipe,
            layer_name=cross_attn_kv_layer,
            stats_dir=stats_dir,
            ds_name=dataset,
            to_collect=to_collect,
            sample_size=sample_size,
            precision=precision,
            batch_tokens=batch_tokens,
            download=download,
            force_recompute=force_recompute
        )
    

def get_all_cross_attn_kv_layer_names(pipe: StableDiffusionPipeline):
    block_types = ["down_blocks", "up_blocks", "mid_block"]
    num_blocks = {
        "down_blocks": 4,
        "up_blocks": 4,
        "mid_block": 1,
    }
    kv_layer_names = []
    for block_type in block_types:
        for idx in range(num_blocks[block_type]):
            for template_key, template in UNET_EDIT_TEMPLATES.items():
                if template_key != "cross-k" and template_key != "cross-v":
                    continue
                for sub_idx in [0, 1, 2]:
                    layer_name = get_to_edit_layername_unet(
                        template_key, block_type, idx, sub_idx
                    )
                    # check if the layer name is valid
                    try:
                        get_attr_through_name(pipe.unet, layer_name)
                    except AttributeError:
                        # print(f"Layer {layer_name} not found. Skipping.")
                        continue
                    kv_layer_names.append(layer_name)
    print(kv_layer_names)
    return kv_layer_names
    


def compute_all_unet_stats(device="cuda:0"):
    """
    Compute all statistics for the unet model.
    """
    sample_pair_size = 3000
    t_steps_per_pair = 10
    batch_size = 3
    precision = "float32"
    to_collect = ["mom2"]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    

    block_types = ["down_blocks", "up_blocks", "mid_block"]
    num_blocks = {
        "down_blocks": 4,
        "up_blocks": 4,
        "mid_block": 1,
    }

    for block_type in block_types:
        for idx in range(num_blocks[block_type]):
            for template_key, template in UNET_EDIT_TEMPLATES.items():
                for sub_idx in [0, 1, 2]:
                    if template_key != "attn-out" and template_key != "res-last-conv":
                        continue
                    layer_name = get_to_edit_layername_unet(
                        template_key, block_type, idx, sub_idx
                    )
                    print(f"Computing statistics for {layer_name}.")
                    # check if the layer name is valid
                    try:
                        get_attr_through_name(pipe.unet, layer_name)
                    except AttributeError:
                        print(f"Layer {layer_name} not found. Skipping.")
                        continue
                    stat = layer_stats_unet(
                        pipe,
                        layer_name,
                        to_collect=to_collect,
                        sample_pair_size=sample_pair_size,
                        t_steps_per_pair=t_steps_per_pair,
                        precision=precision,
                        batch_size=batch_size,
                        force_recompute=False,
                    )


def get_to_edit_layername_unet(
        template_key: Literal["attn-mlp", "attn-out", "res-last-conv", 
                              "upsampler-conv", "downsampler-conv"],
        block_type: Literal["down_blocks", "up_blocks", "mid_block"],
        block_idx: Literal[0, 1, 2, 3],
        sub_idx: Literal[0, 1, 2] 
        ):

    # TODO: redundant util, merge them
    template = UNET_EDIT_TEMPLATES[template_key]
    if "downsampler" in template_key:
        layer_name = template.format(block_type, block_idx, "downsamplers")
    elif "upsampler" in template_key:
        layer_name = template.format(block_type, block_idx, "upsamplers")
    elif "mid_block" in block_type:
        layer_name = template.format(block_type, block_idx, sub_idx)
        # replace mid_block.*. with mid_block.
        layer_name = layer_name.replace(f"mid_block.{block_idx}.", "mid_block.")
    else:
        layer_name = template.format(block_type, block_idx, sub_idx)
    return layer_name


def get_attr_through_name(obj, name):
    """recursive getattr, for dotted names"""
    return reduce(getattr, [obj, *name.split('.')])


if __name__ == "__main__":
    main()



