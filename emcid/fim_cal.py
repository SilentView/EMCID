import os
from pathlib import Path
from functools import reduce
from typing import Literal
import argparse

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler
from einops import rearrange
from torchvision import transforms

from util.globals import *
from util.nethook import TraceDict, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from dsets.stat_dataset import (
    TokenizedDataset, length_collation, dict_to_, flatten_masked_batch,
    ImgTxtRndintDataset
)
from emcid.layer_stats import get_attr_through_name
from emcid.emcid_main import get_cov_text_encoder



STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

def fim_stats(
    pipe: StableDiffusionPipeline,
    module_name,
    stats_dir="data/fim_stats",
    ds_name="ccs_filtered",
    to_collect=["mean"],
    model_name="text_encoder",
    mom2_weight=4000,
    sample_pair_size=None,
    t_steps_per_pair=100,
    precision=None,
    progress=tqdm,
    force_recompute=False,
    download=False,
):
    """
    Function to load or compute cached stats.
    """
    # to calculate the fim stats,
    # the batch size must be 1
    batch_size = 1

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
                     f"{module_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
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

    set_requires_grad(False, pipe.unet, pipe.vae)
    weight_name = module_name + ".weight"
    weight = get_attr_through_name(pipe.text_encoder, weight_name)
    out_dim, in_dim = weight.shape[0:2]

    cov = get_cov_text_encoder(
            pipe.text_encoder,
            pipe.tokenizer,
            module_name,
            "ccs_filtered",
            100000,
            precision,
            verbose=False
        )
    
    for batch in progress(loader, total=batch_count):
        img_batch = batch["img"].to(device)
        prompts = batch["caption"]
        prompts_inp = pipe.tokenizer(prompts, 
                                     return_tensors="pt", 
                                     padding=True, 
                                     truncation=True)
        prompts_inp = {k: v.to(device) for k, v in prompts_inp.items()}

        token_indices = []
        for b in range(batch_size):
            length = prompts_inp["attention_mask"][b].sum()
            token_idx = torch.randint(1, length - 1, (1,))
            token_indices.append(token_idx)

        latents = pipe.vae.encode(img_batch).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        # get the K1.T (C0 + K1 K1.T).inv for all tokens
        # K1 shape: (in_dim, bsz)
        # l_in shape: (bsz, num_tokens, in_dim)
        with torch.no_grad():
            l_in = get_module_input(
                    pipe.text_encoder, 
                    pipe.tokenizer, 
                    prompts, 
                    module_name)
            k1 = l_in[range(batch_size), token_indices, :].T
            k1 = k1.double()
            adj_k = torch.linalg.solve(
                mom2_weight * cov.double() + k1 @ k1.T,
                k1
            )
            factors = torch.ones((1, batch_size), device=device).double()
            # shape: (in_dim, )
            right_vec = (factors @ adj_k.T).squeeze()
            right_vec = right_vec.reshape(in_dim, 1)

        # calculate through all 1000 timesteps,following Selective Amnesia
        for t in range(t_steps_per_pair):
            # print(batch)
            # print(batch["img"].shape)
            noise = torch.randn_like(latents, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (batch_size,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            txt_repr = pipe.text_encoder(**prompts_inp).last_hidden_state
            pred_noise = pipe.unet(noisy_latents, timesteps, txt_repr).sample

            mse_loss = F.mse_loss(pred_noise, noise, reduction="mean")
            mse_loss.backward()

            left_vec = weight.grad.T.double()
            # this is dL/d(delta)
            with torch.no_grad():
                grad = (left_vec * right_vec).sum(dim=0).reshape(1, out_dim)

            grad_square = (grad ** 2).clone().detach().cpu().to(dtype=dtype)
            stat.add(grad_square)
            # zero the gradient
            pipe.text_encoder.zero_grad()
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()

    return stat


def get_module_input(
    model,
    tokenizer,
    prompts,
    module_name,  
):
    """get the input of a module"""
    device = model.device
    prompts_inp = tokenizer(prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True).to(device)
    with TraceDict(
        module=model,
        layers=[module_name],
        retain_input=True,
        retain_output=False,
        edit_output=None
    ) as td:
        model(**prompts_inp)
        l_in = td[module_name].input.detach().clone()
    
    return l_in


if __name__ == "__main__":
    module_name = "text_model.encoder.layers.10.mlp.fc2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps_per_pair", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    
    sample_pair_size = 3000
    t_steps_per_pair = args.steps_per_pair
    precision = "float32"
    to_collect = ["mean"]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    stat = fim_stats(
        pipe,
        module_name,
        ds_name="ccs_filtered",
        to_collect=to_collect,
        model_name="text_encoder",
        mom2_weight=4000,
        sample_pair_size=sample_pair_size,
        t_steps_per_pair=t_steps_per_pair,
        precision=precision,
        progress=tqdm,
        force_recompute=False,
        download=False,
    )