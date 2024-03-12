from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import bisect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from diffusers import (
    StableDiffusionPipeline, DDPMScheduler, StableDiffusionPipelineSafe, 
    StableDiffusionXLPipeline, EulerDiscreteScheduler
)
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import einops

from util import nethook
from util.runningstats import CombinedStat, SecondMoment, Mean, NormMean
from util.globals import *
from experiments.causal_trace import find_token_range
from emcid.layer_stats import get_attr_through_name, get_all_cross_attn_kv_layer_names

from emcid.emcid_hparams import (
    EMCIDHyperParams, UNetEMCIDHyperParams, get_accum_time_blocks,
    EMCIDXLHyperParams
)



def preprocess_img(images, device):
    """
    The denoising training process has referred to: 
    https://huggingface.co/docs/diffusers/training/text2image#finetuning
    """
    images = [image.convert("RGB") for image in images]
    train_transforms = transforms.Compose(
        [
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = [train_transforms(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(device)
    
    return pixel_values


def tokenize_prompts(
        prompts, 
        tokenizer, 
        device, 
        padding_length=None):
    """
    Tokenize prompts
    """
    if padding_length is None:
        input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    else:
        input = tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=padding_length)
    input = {k: v.to(device) for k, v in input.items()}
    return input


def compute_z_text_encoder_global(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: EMCIDHyperParams,
    layer: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A single request belike:
    request = {
                "seed": 2023,
                "prompts": [],
                "dest": " ",
                "source": nudity
            }
    """
    # Get model parameters
    print("Computing right vector (v)")
    device = pipe.device

    # get a deep copy of the text_encoder
    text_model_to_edit = deepcopy(pipe.text_encoder)
    text_model_to_edit.to(device)

    # Tokenize source into list of int token IDs
    source_prompts = request["source_prompts"]
    
    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    delta = torch.zeros((text_model_to_edit.config.hidden_size,), requires_grad=True, device=device)
    source_init, kl_distr_init = None, None

    if request["source"] == "[CLS]":
        edit_idx = 0
    elif request["source"] == "[EOS]":
        edit_idx = -1

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if source_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                # here only the first CLS token is used 
                source_init = cur_out[0][:, edit_idx].detach().clone()
                source_init = source_init.mean(dim=0)

            # Add intervened delta
            for i in range(bsz):
                cur_out[0][i, edit_idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, text_model_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    if hparams.objective == "ablate-source":
        if "training_img_paths" in request:
            # read images from disk
            img_batch = [Image.open(path) for path in request["training_img_paths"]] 
        else:
            img_batch = []
            print("generate source images for ablate-source")
            for prompt, seed in zip(source_prompts, request["seeds"]):
                generator = torch.Generator(pipe.device).manual_seed(int(seed)) if seed is not None else None
                img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
                img_batch.append(img)

    elif hparams.objective == "ablate-dest":
        if "training_img_paths" in request:
            # read images from disk
            img_batch = [Image.open(path) for path in request["training_img_paths"]]
        else:
            print("generate dest images, for ablate-dest")
            img_batch = sld_generate(pipe, request["source_prompts"], request["seeds"], request["indices"], sld_type=hparams.sld_type)

    elif hparams.objective == "esd":
        print("generate source images for esd objective")
        img_batch = []
        for prompt, seed in zip(source_prompts, request["seeds"]):
            generator = torch.Generator(pipe.device).manual_seed(int(seed)) if seed is not None else None
            img = pipe([prompt], guidance_scale=7.5, generator=generator).images[0]
            img_batch.append(img)
    else:
        raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

    img_batch = preprocess_img(img_batch, device)

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    safe_inp = tokenize_prompts(request["safe_words"], pipe.tokenizer, device)
    uncond_inp = tokenize_prompts([""] * img_batch.shape[0], pipe.tokenizer, device)

    assert len(source_prompts_inp["input_ids"]) == len(img_batch), \
        "The number of prompts and images should be the same."

    bsz = len(img_batch)
    
    
    # get the sld hparams
    if hparams.sld_type == "max":
        sld_dict = {
            "sld_guidance_scale": 5000,
            "sld_warmup_steps": 0,
            "sld_threshold": 1.0,
            "sld_momentum_scale": 0.5,
            "sld_mom_beta": 0.7
        }
    elif hparams.sld_type == "strong":
        sld_dict = {
            "sld_guidance_scale": 2000,
            "sld_warmup_steps": 7,
            "sld_threshold": 0.025,
            "sld_momentum_scale": 0.5,
            "sld_mom_beta": 0.7
        }
    else:
        raise ValueError(f"sld_type {hparams.sld_type} not supported")
    sld_dict = {k: torch.tensor(v).to(device) for k, v in sld_dict.items()}

    # compute the time step invariant representation
    with torch.no_grad():
        latents = pipe.vae.encode(img_batch).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        safety_repr = pipe.text_encoder(**safe_inp)[0] 
        # get the non edit model prediction
        source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0]
        uncond_inp_repr = pipe.text_encoder(**uncond_inp)[0]

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=text_model_to_edit,
            layers=[
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            # with torch.no_grad():
            #     latents = pipe.vae.encode(img_batch).latent_dist.sample()
            #     latents = latents * pipe.vae.config.scaling_factor

            # sample noise that will be added to the latents 
            noise = torch.randn_like(latents, device=device) 
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            edit_source_txt_repr = text_model_to_edit(**source_prompts_inp)[0] # the return value is the last hidden state
            with torch.no_grad():
                # get the safe latent diffusion prediction
                model_pred_source = pipe.unet(noisy_latents, timesteps, source_txt_repr).sample
               
                model_pred_uncond = pipe.unet(noisy_latents, timesteps, uncond_inp_repr).sample
                # code below largely refers to 
                # StableDiffusionPipelineSafe implementation
                if hparams.sld_supervision:
                    # get the safe latent diffusion prediction
                    model_pred_safety = pipe.unet(noisy_latents, timesteps, safety_repr).sample
                    scale = torch.clamp(
                        torch.abs((model_pred_source - model_pred_safety)) * sld_dict["sld_guidance_scale"], max=1.0
                    )

                    safety_concept_scale = torch.where(
                        (model_pred_source - model_pred_safety) >= sld_dict["sld_threshold"],
                        torch.zeros_like(scale),
                        scale,
                    )

                    noise_guidance_safety = torch.mul(
                        (model_pred_safety - model_pred_uncond), safety_concept_scale
                    )


            bs_embed, seq_len, _ = edit_source_txt_repr.shape

            edit_model_pred_source = pipe.unet(noisy_latents, timesteps, edit_source_txt_repr).sample # (bsz, latent_shape)
            if "ablate" in hparams.objective:
                # compute MSE loss
                if hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise:
                    # print("using sampled noise")
                    mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")
                else:
                    loss_supervision = model_pred_source - noise_guidance_safety
                    mse_loss = F.mse_loss(edit_model_pred_source, loss_supervision, reduction="mean")

                # weight decay
                weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)            

                loss = mse_loss + weight_decay
                # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
            elif hparams.objective == "esd":
                # compute the esd loss
                tmp = model_pred_uncond - hparams.esd_mu * (model_pred_source - model_pred_uncond)
                mse_loss = F.mse_loss(edit_model_pred_source, tmp, reduction="mean")
                weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)
                loss = mse_loss + weight_decay
            else:
                raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

            loss.backward()
            opt.step()
            # write the loss into a file
            with open("log/loss_text_encoder.txt", "a") as f:
                f.write(f"step {it}, loss: {loss.item()}\n")

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * source_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

    source = source_init + delta
    tqdm.write(
        f"Init norm {source_init.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )

    return source


def compute_z_text_encoder(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: EMCIDHyperParams,
    layer: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the update.
    Runs a simple optimization procedure.
    """
    # Get model parameters
    print("Computing right vector (v)")
    device = pipe.device

    # get a deep copy of the text_encoder
    text_model_to_edit = deepcopy(pipe.text_encoder)
    text_model_to_edit.to(device)

    # Tokenize source into list of int token IDs

    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    if hparams.objective == "esd":
        dest_prompts = ["" for p in prompts_tmp]
    else:
        dest_prompts = [p.format(request["dest"]) for p in prompts_tmp]
    
    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    delta = torch.zeros((text_model_to_edit.config.hidden_size,), requires_grad=True, device=device)
    source_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if source_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_init = cur_out[0][0, source_lookup_indices[0]].detach().clone()

            # Add intervened delta
            if hparams.replace_repr:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] = delta
            else:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, text_model_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
    if hparams.objective == "ablate-source":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]] 
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate source images for ablate-source")
            samples_per_prompt = hparams.samples_per_prompt
            all_imgs = []
            for _ in range(samples_per_prompt):
                imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                all_imgs.extend(imgs)
    elif hparams.objective == "ablate-dest":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]]
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate dest images, for ablate-dest")
            all_imgs = []
            for _ in range(hparams.samples_per_prompt):
                imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                all_imgs.extend(imgs)
    elif hparams.objective == "esd":
        print("generate source images for esd objective")
        all_imgs = []
        for _ in range(hparams.samples_per_prompt):
            imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
            all_imgs.extend(imgs)
    else:
        raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")
    all_imgs = preprocess_img(all_imgs, "cpu")
    # reshape so that the first dimension is the number of prompts
    all_imgs = einops.rearrange(all_imgs, 
                                "(s b) c h w -> b s c h w", 
                                s=hparams.samples_per_prompt)


    bsz = len(source_prompts)
    # we assume len(img_batch) is n times of batch size
    assert len(all_imgs) % bsz == 0, \
            f"len(img_batch) {len(all_imgs)} should be n times of batch size {bsz}"

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    if hparams.align_obj_eos_pad:
        # calcuate full padding input
        source_prompts_inp_full = pipe.tokenizer(source_prompts, 
                                             max_length=pipe.tokenizer.model_max_length,
                                             return_tensors="pt", 
                                             padding="max_length", 
                                             truncation=True).to(device)
    
        dest_prompts_inp_full = pipe.tokenizer(dest_prompts,
                                             max_length=pipe.tokenizer.model_max_length,
                                             return_tensors="pt", 
                                             padding="max_length", 
                                             truncation=True).to(device)
        
        source_eos_token_indices = [attn_mask.sum() - 1 for attn_mask in source_prompts_inp_full["attention_mask"]]
        dest_eos_token_indices = [attn_mask.sum() - 1 for attn_mask in dest_prompts_inp_full["attention_mask"]]

        farthest_eos_token_index = max(source_eos_token_indices + dest_eos_token_indices)

        source_slices = [range(eos_token_index, pipe.tokenizer.model_max_length - \
                         max(0, farthest_eos_token_index - eos_token_index)) 
                         for eos_token_index in source_eos_token_indices]
        
        dest_slices = [range(eos_token_index, pipe.tokenizer.model_max_length - \
                         max(0, farthest_eos_token_index - eos_token_index))
                         for eos_token_index in dest_eos_token_indices]
        
        with torch.no_grad():
            dest_embds_full = pipe.text_encoder(**dest_prompts_inp_full)[0]

    if hparams.contrastive_text_loss:
        negative_prompts_inp = tokenize_prompts(request["negative_prompts"], pipe.tokenizer, device)

    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) for ids in source_prompts_inp["input_ids"]] 
    source_lookup_indices = [range[-1] - 1 for range in source_object_ranges]

    dest_object_ranges = [find_token_range(pipe.tokenizer, ids, request["dest"]) for ids in dest_prompts_inp["input_ids"]]
    dest_lookup_indices = [range[-1] - 1 for range in dest_object_ranges]

    assert len(source_prompts_inp["input_ids"]) == len(dest_prompts_inp["input_ids"]) == len(all_imgs), \
        "The number of prompts and images should be the same."

    

    if hparams.use_ewc:
        # load the FIM
        print("using ewc")
        stat = CombinedStat(**{"mean": Mean()})
        file_path = "data/fim_stats/text_encoder/ccs_filtered_stats/text_model.encoder.layers.10.mlp.fc2_float32_mean_step10_3000.npz"
        data = np.load(file_path, allow_pickle=True)
        stat.load_state_dict(data)
        fim = stat.mean.state_dict()["mean"]
        fim = torch.from_numpy(fim).to(device)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        sample_indices = torch.randint(0, hparams.samples_per_prompt, (bsz,))

        img_batch = all_imgs[torch.arange(bsz), sample_indices].to(device)

        # compute the time step invariant representation
        with torch.no_grad():
            latents = pipe.vae.encode(img_batch).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            dest_txt_repr, dest_pooler_repr = pipe.text_encoder(**dest_prompts_inp)[0:2] 

            if hparams.objective == "esd" or hparams.cal_text_repr_loss:
                # get the non edit model prediction
                source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0]
                if hparams.contrastive_text_loss:
                    negative_prompts_repr, negative_pooler_repr = pipe.text_encoder(**negative_prompts_inp)[0:2]

        # Forward propagation
        with nethook.TraceDict(
            module=text_model_to_edit,
            layers=[
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:

            # sample noise that will be added to the latents 
            noise = torch.randn_like(latents, device=device) 
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            edit_source_txt_repr, edit_pooler_repr = text_model_to_edit(**source_prompts_inp)[0:2] # the return value is the last hidden state
            with torch.no_grad():
                if hparams.objective == "esd":
                    model_pred_source = pipe.unet(noisy_latents, timesteps, source_txt_repr).sample

            bs_embed, seq_len, _ = edit_source_txt_repr.shape

            if not hparams.no_noise_loss:
                edit_model_pred_source = pipe.unet(noisy_latents, timesteps, edit_source_txt_repr).sample # (bsz, latent_shape)
                model_pred_dest = pipe.unet(noisy_latents, timesteps, dest_txt_repr).sample
            if "ablate" in hparams.objective:
                # compute MSE loss
                if (hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise)\
                    or request.get("use_real_noise", False):
                    # print("using sampled noise")
                    mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")
                elif hparams.no_noise_loss:
                    mse_loss = None
                else:
                    mse_loss = F.mse_loss(edit_model_pred_source, model_pred_dest, reduction="mean")

                # weight decay or EWC
                if hparams.use_ewc:
                    # compute the ewc loss
                    reg_loss = torch.sum(float(hparams.ewc_lambda) * fim * delta ** 2) / (2 * torch.norm(source_init) ** 2)
                else:
                    reg_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)            

                if hparams.no_noise_loss:
                    loss = reg_loss
                else:
                    loss = mse_loss + reg_loss
                # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
            elif hparams.objective == "esd":
                tmp = model_pred_dest - hparams.esd_mu * (model_pred_source - model_pred_dest)
                mse_loss = F.mse_loss(edit_model_pred_source, tmp, reduction="mean")
                reg_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)
                loss = mse_loss + reg_loss
            else:
                raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

            if hparams.cal_text_repr_loss and request.get("txt_align", True):
                # compute the text representation loss
                if hparams.contrastive_text_loss:
                    single_dest_inp = tokenize_prompts([request["dest"]], pipe.tokenizer, device)
                    single_dest_txt_repr = pipe.text_encoder(**single_dest_inp)[1]

                    dest_kl_embeddings = torch.cat([single_dest_txt_repr, negative_pooler_repr], dim=0)
                    # print(f"dest_kl_embeddings shape {dest_kl_embeddings.shape}")
                    logits_per_text = - torch.cdist(
                        torch.unsqueeze(edit_pooler_repr, dim=0),
                        torch.unsqueeze(dest_kl_embeddings, dim=0),
                    )
                    source_dest_kl_text_scores = torch.squeeze(logits_per_text)

                    source_dest_kl_text_log_probs = torch.log_softmax(source_dest_kl_text_scores, dim=1)
                    contrastive_loss = source_dest_kl_text_log_probs[:, 0]
                    contrastive_loss = contrastive_loss.mean(dim=0)
                    nll_loss = - contrastive_loss
                    loss += hparams.text_repr_loss_scale_factor * nll_loss
                elif hparams.align_object_token:
                    # get the batched object token repr
                    text_repr_loss = F.mse_loss(
                        edit_source_txt_repr[torch.arange(bsz), source_lookup_indices, :],
                        dest_txt_repr[torch.arange(bsz), dest_lookup_indices, :],
                        reduction="mean"
                    )
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss
                elif hparams.align_obj_eos_pad:
                    # get the batched object token repr
                    edited_repr = edit_source_txt_repr[torch.arange(bsz), source_lookup_indices, :]
                    supervision_repr = dest_txt_repr[torch.arange(bsz), dest_lookup_indices, :]

                    # get the batched eos and padding repr (in the full sequence length)
                    edit_source_embs_full = text_model_to_edit(**source_prompts_inp_full)[0]
                    edited_pad_repr = [edit_source_embs_full[i, source_slices, :]
                                       for i, source_slices in enumerate(source_slices)]
                    edited_pad_repr = torch.stack(edited_pad_repr, dim=0)
                    supervision_pad_repr = [dest_embds_full[i, dest_slices, :]
                                            for i, dest_slices in enumerate(dest_slices)]
                    supervision_pad_repr = torch.stack(supervision_pad_repr, dim=0)

                    text_repr_loss = F.mse_loss(
                        torch.cat([edited_repr.unsqueeze(dim=1), edited_pad_repr], dim=1),
                        torch.cat([supervision_repr.unsqueeze(dim=1), supervision_pad_repr], dim=1),
                        reduction="mean"
                    )
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss
                else:
                    # simple mse alignmnet
                    text_repr_loss = F.mse_loss(edit_pooler_repr, dest_pooler_repr, reduction="mean")
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss

            loss.backward()
            opt.step()
            # print(
            #     f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}"\
            # )
            # if hparams.contrastive_text_loss:
            #     print(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}")
            # if hparams.cal_text_repr_loss:
            #     print(f"Text repr loss {text_repr_loss.item()}")

            
            # write the loss into a file
            # with open(f"log/loss_text_encoder.txt", "a") as f:
            #     f.write(f"step {it}, loss: {loss.item()}\n")
            #     f.write(f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}\n")
            #     if hparams.contrastive_text_loss:
            #         f.write(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}\n")
            #     if hparams.cal_text_repr_loss:
            #         f.write(f"Text repr loss {text_repr_loss.item()}\n")

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * source_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

    source = source_init + delta
    tqdm.write(
        f"Init norm {source_init.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )

    return source

def compute_z_sdxl_text_encoders(
    pipe: StableDiffusionXLPipeline,
    request: Dict,
    hparams: EMCIDXLHyperParams,
    layers: Tuple[int, int],
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the update.
    Runs a simple optimization procedure.
    """
    # Get model parameters
    print("Computing right vector (v)")
    device = pipe.device
    layer, layer_2 = layers

    # get a deep copy of the text_encoder
    text_model_to_edit = pipe.text_encoder
    text_model_to_edit_2 = pipe.text_encoder_2

    # Tokenize source into list of int token IDs
    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    if hparams.objective == "esd":
        dest_prompts = ["" for p in prompts_tmp]
    else:
        dest_prompts = [p.format(request["dest"]) for p in prompts_tmp]
    
    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    delta = torch.zeros((text_model_to_edit.config.hidden_size,), 
                        requires_grad=True, 
                        device=device)
    deltas_2 = torch.zeros((text_model_to_edit_2.config.hidden_size,),
                            requires_grad=True,
                            device=device)

    source_init, kl_distr_init = None, None
    source_init_2, kl_distr_init_2 = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if source_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_init = cur_out[0][0, source_lookup_indices[0]].detach().clone()

            # Add intervened delta
            if hparams.replace_repr:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] = delta
            else:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] += delta

        return cur_out
    
    def edit_output_fn_2(cur_out, cur_layer):
        nonlocal source_init_2

        if cur_layer == hparams.layer_module_tmp.format(layer_2):
            # Store initial value of the vector of interest
            if source_init_2 is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_init_2 = cur_out[0][0, source_lookup_indices_2[0]].detach().clone()

            # Add intervened delta
            if hparams.replace_repr:
                for i, idx in enumerate(source_lookup_indices_2):
                    cur_out[0][i, idx, :] = deltas_2
            else:
                for i, idx in enumerate(source_lookup_indices_2):
                    cur_out[0][i, idx, :] += deltas_2

    # Optimizer
    opt = torch.optim.Adam([delta, deltas_2], lr=hparams.v_lr)

    noise_scheduler = pipe.scheduler
    nethook.set_requires_grad(False, 
                              pipe.vae, pipe.unet, 
                              text_model_to_edit, pipe.text_encoder,
                              text_model_to_edit_2, pipe.text_encoder_2)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) \
                if request["seed_train"] is not None else None

    if hparams.objective == "ablate-source":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]] 
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate source images for ablate-source")
            samples_per_prompt = hparams.samples_per_prompt
            all_imgs = []
            with torch.no_grad():
                for _ in range(samples_per_prompt):
                    imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                    all_imgs.extend(imgs)
            
    elif hparams.objective == "ablate-dest":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]]
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate dest images, for ablate-dest")
            all_imgs = []
            with torch.no_grad():
                for _ in range(hparams.samples_per_prompt):
                    for prompt in source_prompts:
                        img = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
                        all_imgs.append(img)
                # save the images
            tmp_dir = "cache/tmp"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            for i, img in enumerate(all_imgs):
                img.save(f"{tmp_dir}/{i}.png")
    else:
        raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")
    all_imgs = preprocess_img(all_imgs, "cpu")
    # reshape so that the first dimension is the number of prompts
    all_imgs = einops.rearrange(all_imgs, 
                                "(s b) c h w -> b s c h w", 
                                s=hparams.samples_per_prompt)

    bsz = len(source_prompts)
    # we assume len(img_batch) is n times of batch size
    assert len(all_imgs) % bsz == 0, \
            f"len(img_batch) {len(all_imgs)} should be n times of batch size {bsz}"

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    source_prompts_inp_2 = tokenize_prompts(source_prompts, pipe.tokenizer_2, device)
    dest_prompts_inp_2 = tokenize_prompts(dest_prompts, pipe.tokenizer_2, device)


    # text encoder 1
    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) 
                            for ids in source_prompts_inp["input_ids"]] 
    source_lookup_indices = [range[-1] - 1 for range in source_object_ranges]

    dest_object_ranges = [find_token_range(pipe.tokenizer, ids, request["dest"]) 
                            for ids in dest_prompts_inp["input_ids"]]
    dest_lookup_indices = [range[-1] - 1 for range in dest_object_ranges]

    # text encoder 2
    source_object_ranges_2 = [find_token_range(pipe.tokenizer_2, ids, request["source"])
                              for ids in source_prompts_inp_2["input_ids"]]
    source_lookup_indices_2 = [range[-1] - 1 for range in source_object_ranges_2]

    dest_object_ranges_2 = [find_token_range(pipe.tokenizer_2, ids, request["dest"])
                              for ids in dest_prompts_inp_2["input_ids"]]
    dest_lookup_indices_2 = [range[-1] - 1 for range in dest_object_ranges_2]

    assert len(source_prompts_inp["input_ids"]) == len(dest_prompts_inp["input_ids"]) == len(all_imgs), \
        "The number of prompts and images should be the same."

    if hparams.use_ewc:
        # load the FIM
        print("using ewc")
        stat = CombinedStat(**{"mean": Mean()})
        file_path = "data/fim_stats/text_encoder/ccs_filtered_stats/\
                    text_model.encoder.layers.10.mlp.fc2_float32_mean_step10_3000.npz"
        data = np.load(file_path, allow_pickle=True)
        stat.load_state_dict(data)
        fim = stat.mean.state_dict()["mean"]
        fim = torch.from_numpy(fim).to(device)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        sample_indices = torch.randint(0, hparams.samples_per_prompt, (bsz,))
        img_batch = all_imgs[torch.arange(bsz), sample_indices].to(device)

        # compute the time step invariant representation
        with torch.no_grad():
            latents = pipe.vae.encode(img_batch).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            dest_prompt_embeds = pipe.text_encoder(**dest_prompts_inp, 
                                                     output_hidden_states=True
                                                     )
            dest_txt_repr, dest_pooler_repr = (dest_prompt_embeds.hidden_states[-2], 
                                                   dest_prompt_embeds.pooler_output)

            dest_prompt_embeds_2 = pipe.text_encoder_2(**dest_prompts_inp, 
                                                         output_hidden_states=True)
            dest_txt_repr_2, dest_pooler_repr_2 = (dest_prompt_embeds_2.hidden_states[-2],
                                                       dest_prompt_embeds_2.text_embeds)

            # print(dest_txt_repr.shape, dest_txt_repr_2.shape)
            # print(dest_pooler_repr.shape, dest_pooler_repr_2.shape)
            dest_prompt_embeds = torch.cat([dest_txt_repr, dest_txt_repr_2], dim=-1)

            height = pipe.default_sample_size * pipe.vae_scale_factor
            width = pipe.default_sample_size * pipe.vae_scale_factor

            original_size = (height, width)
            source_size = (height, width)

            add_time_ids = pipe._get_add_time_ids(
                                original_size=original_size,
                                crops_coords_top_left=(0, 0),
                                source_size=source_size,
                                dtype=dest_prompt_embeds.dtype,
                                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
                            )
            add_time_ids = add_time_ids.repeat(bsz, 1)
            add_time_ids = add_time_ids.to(device)

            dest_add_text_embeds = dest_pooler_repr_2

            dest_added_cond_kwargs = {"text_embeds": dest_add_text_embeds, "time_ids": add_time_ids}

            

        # if hparams.cal_text_repr_loss:
        #     # get the non edit model prediction
        #     source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0]
        #     source_txt_repr_2 = pipe.text_encoder_2(**source_prompts_inp_2)[0]

        
        # Forward propagation
        with nethook.TraceDict(
            module=text_model_to_edit,
            layers=[
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            with nethook.TraceDict(
                module=text_model_to_edit_2,
                layers=[
                    hparams.layer_module_tmp.format(layer_2),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn_2,
            ) as tr_2:

                # sample noise that will be added to the latents 
                noise = torch.randn_like(latents, device=device) 
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                edit_source_prompt_embeds = text_model_to_edit(**source_prompts_inp, 
                                                              output_hidden_states=True)

                edit_source_txt_repr, edit_source_pooler_repr = (edit_source_prompt_embeds.hidden_states[-2], 
                                                                  edit_source_prompt_embeds.pooler_output)
                
                edit_source_prompt_embeds_2 = text_model_to_edit_2(**source_prompts_inp_2,
                                                                    output_hidden_states=True)

                edit_source_txt_repr_2, edit_source_pooler_repr_2 = (edit_source_prompt_embeds_2.hidden_states[-2],
                                                                     edit_source_prompt_embeds_2.text_embeds)


                bs_embed, seq_len, _ = edit_source_txt_repr.shape

                # concat the two text repr as the input to the unet
                edit_source_prompt_embeds = torch.cat(
                                        [edit_source_txt_repr, edit_source_txt_repr_2], 
                                        dim=-1)
                
                edit_source_add_text_embeds = edit_source_pooler_repr_2
                edit_source_added_cond_kwargs = {"text_embeds": edit_source_add_text_embeds, "time_ids": add_time_ids}


                if not hparams.no_noise_loss:
                    # (bsz, latent_shape)
                    edit_model_pred_source = pipe.unet(
                                    noisy_latents, 
                                    timesteps, 
                                    encoder_hidden_states=edit_source_prompt_embeds,
                                    added_cond_kwargs=edit_source_added_cond_kwargs
                                    ).sample 
                    model_pred_dest = pipe.unet(
                                    noisy_latents, 
                                    timesteps, 
                                    encoder_hidden_states=dest_prompt_embeds,
                                    added_cond_kwargs=dest_added_cond_kwargs
                                    ).sample

                if "ablate" in hparams.objective:
                    # compute MSE loss
                    if (hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise)\
                        or request.get("use_real_noise", False):
                        # print("using sampled noise")
                        mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")

                    elif hparams.no_noise_loss:
                        mse_loss = None

                    else:
                        mse_loss = F.mse_loss(edit_model_pred_source, model_pred_dest, reduction="mean")

                    # weight decay or EWC
                    if hparams.use_ewc:
                        raise ValueError("ewc not implemented for sdxl")
                        # compute the ewc loss
                        reg_loss = torch.sum(float(hparams.ewc_lambda) * fim * delta ** 2) / (2 * torch.norm(source_init) ** 2)
                    else:
                        reg_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)            
                        reg_loss_2 = hparams.v_weight_decay * (torch.norm(deltas_2) / torch.norm(source_init_2) ** 2)

                    if hparams.no_noise_loss:
                        loss = reg_loss + reg_loss_2
                    else:
                        loss = mse_loss + reg_loss + reg_loss_2
                    # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
                else:
                    raise ValueError(f"Objective {hparams.objective} not implemented for compute_z with sdxl.")

                if hparams.cal_text_repr_loss and request.get("txt_align", True):
                    # compute the text representation loss
                    # simple mse alignmnet
                    text_repr_loss = F.mse_loss(edit_source_pooler_repr, dest_pooler_repr, reduction="mean")
                    text_repr_loss_2 = F.mse_loss(edit_source_pooler_repr_2, dest_pooler_repr_2, reduction="mean")
                    loss = loss + \
                           hparams.text_repr_loss_scale_factor * text_repr_loss + \
                           hparams.text_repr_loss_scale_factor * text_repr_loss_2

                loss.backward()
                opt.step()
                with open("log/loss_text_encoder.txt", "a") as f:
                    print(
                        f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}"\
                        f"| Weight decay_2 {reg_loss_2.item()}",
                        file=f
                    )
                    if hparams.cal_text_repr_loss:
                        print(f"Text repr loss1 {text_repr_loss.item()}", file=f)
                        print(f"Text repr loss2 {text_repr_loss_2.item()}", file=f)
                # write the loss into a file
                # with open(f"log/loss_text_encoder.txt", "a") as f:
                #     f.write(f"step {it}, loss: {loss.item()}\n")
                #     f.write(f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}\n")
                #     if hparams.contrastive_text_loss:
                #         f.write(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}\n")
                #     if hparams.cal_text_repr_loss:
                #         f.write(f"Text repr loss {text_repr_loss.item()}\n")

                # Project within L2 ball
                max_norm = hparams.clamp_norm_factor * source_init.norm()
                max_norm_2 = hparams.clamp_norm_factor * source_init_2.norm()
                if delta.norm() > max_norm:
                    with torch.no_grad():
                        delta[...] = delta * max_norm / delta.norm()
                if deltas_2.norm() > max_norm_2:
                    with torch.no_grad():
                        deltas_2[...] = deltas_2 * max_norm_2 / deltas_2.norm()

    source = source_init + delta
    source_2 = source_init_2 + deltas_2
    tqdm.write(
        f"text1:Init norm {source_init.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )
    tqdm.write(
        f"text2:Init norm {source_init_2.norm()} | Delta norm {deltas_2.norm()} | source norm {source_2.norm()}"
    )

    return source, source_2



def compute_z_text_encoder_v2(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: EMCIDHyperParams,
    layer: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Features enabled in this version:
    1. use elastic weight consolidation(ewc)
    2. not only last subject token is modified. Can further support eos token and padding tokens.
    3. change "esd" to "uncondition"

    Return a list of vectors
    """
    # Get model parameters
    print("Computing right vector (v)")
    device = pipe.device

    # get a deep copy of the text_encoder
    text_model_to_edit = deepcopy(pipe.text_encoder)
    text_model_to_edit.to(device)

    # Tokenize source into list of int token IDs

    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    if hparams.objective == "esd":
        dest_prompts = ["" for p in prompts_tmp]
    else:
        dest_prompts = [p.format(request["dest"]) for p in prompts_tmp]

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    # re tokenzie the prompts, getting enough padding tokens
    num_pad_tokens_used = hparams.num_edit_tokens - 2
    if hparams.num_edit_tokens > 1:
        # the padding length should be the longest length plus hparam.num_edit_tokens - 2
        padded_length = max([len(source_prompts_inp["input_ids"][0]),
                          len(dest_prompts_inp["input_ids"][0])]) + num_pad_tokens_used
        
        source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device, padding_length=padded_length)
        dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device, padding_length=padded_length)
    

    
    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    deltas = torch.zeros((hparams.num_edit_tokens, text_model_to_edit.config.hidden_size), requires_grad=True, device=device)
    source_inits = torch.zeros((hparams.num_edit_tokens, text_model_to_edit.config.hidden_size), device=device)
    source_init_initialized = False

    # Inserts new "delta" variable at certain indices
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_inits
        nonlocal source_init_initialized

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if not source_init_initialized:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence, only the first element of the batch is used
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_inits = cur_out[0][0, source_lookup_indices[0]].detach().clone()
                print("source inits shape:", source_inits.shape)
                source_init_initialized = True

            # Add intervened delta
            if hparams.replace_repr:
                for i, indices in enumerate(source_lookup_indices):
                    for j, index in enumerate(indices):
                        cur_out[0][i, index, :] = deltas[j, :]
            else:
                for i, indices in enumerate(source_lookup_indices):
                    for j, index in enumerate(indices):
                        cur_out[0][i, index, :] += deltas[j, :]
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([deltas], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, text_model_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
    if hparams.objective == "ablate-source":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]] 
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate source images for ablate-source")
            samples_per_prompt = hparams.samples_per_prompt
            all_imgs = []
            for _ in range(samples_per_prompt):
                imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                all_imgs.extend(imgs)
    elif hparams.objective == "ablate-dest":
        if "training_img_paths" in request:
            # read images from disk
            all_imgs = [Image.open(path) for path in request["training_img_paths"]]
        elif "images" in request:
            # read images from request
            all_imgs = request["images"]
        else:
            print("generate dest images, for ablate-dest")
            all_imgs = []
            for _ in range(hparams.samples_per_prompt):
                imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
                all_imgs.extend(imgs)
    elif hparams.objective == "esd":
        print("generate source images for esd objective")
        all_imgs = []
        for _ in range(hparams.samples_per_prompt):
            imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
            all_imgs.extend(imgs)
    else:
        raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

    all_imgs = preprocess_img(all_imgs, "cpu")
    # reshape so that the first dimension is the number of prompts
    all_imgs = einops.rearrange(all_imgs, 
                                "(s b) c h w -> b s c h w", 
                                s=hparams.samples_per_prompt)


    bsz = len(source_prompts)
    # we assume len(img_batch) is n times of batch size
    assert len(all_imgs) % bsz == 0, \
            f"len(img_batch) {len(all_imgs)} should be n times of batch size {bsz}"

    
    if hparams.num_edit_tokens >= 2:
        source_eos_token_indices = [attn_mask.sum() - 1 for attn_mask in source_prompts_inp["attention_mask"]]
        dest_eos_token_indices = [attn_mask.sum() - 1 for attn_mask in dest_prompts_inp["attention_mask"]]

        # The padding has been done before
        source_indices = [list(range(eos_token_index, eos_token_index + num_pad_tokens_used + 1))
                         for eos_token_index in source_eos_token_indices]
        
        dest_indices = [list(range(eos_token_index, eos_token_index + num_pad_tokens_used + 1))
                         for eos_token_index in dest_eos_token_indices]

        
    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) for ids in source_prompts_inp["input_ids"]] 
    # last subject token
    source_lookup_indices = [[range[-1] - 1] for range in source_object_ranges]
    

    dest_object_ranges = [find_token_range(pipe.tokenizer, ids, request["dest"]) for ids in dest_prompts_inp["input_ids"]]
    # last subject token
    dest_lookup_indices = [[range[-1] - 1] for range in dest_object_ranges]

    if hparams.num_edit_tokens >= 2:
        # add eos and padding tokens
        source_lookup_indices = [indices + source_indices[i] for i, indices in enumerate(source_lookup_indices)]
        dest_lookup_indices = [indices + dest_indices[i] for i, indices in enumerate(dest_lookup_indices)]

    assert len(source_prompts_inp["input_ids"]) == len(dest_prompts_inp["input_ids"]) == len(all_imgs), \
        "The number of prompts and images should be the same."

    if hparams.use_ewc:
        # load the FIM
        print("using ewc")
        stat = CombinedStat(**{"mean": Mean()})
        file_path = "data/fim_stats/text_encoder/ccs_filtered_stats/text_model.encoder.layers.10.mlp.fc2_float32_mean_step10_3000.npz"
        data = np.load(file_path, allow_pickle=True)
        stat.load_state_dict(data)
        fim = stat.mean.state_dict()["mean"]
        fim = torch.from_numpy(fim).to(device)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        sample_indices = torch.randint(0, hparams.samples_per_prompt, (bsz,))

        img_batch = all_imgs[torch.arange(bsz), sample_indices].to(device)

        # compute the time step invariant representation
        with torch.no_grad():
            latents = pipe.vae.encode(img_batch).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            dest_txt_repr, dest_pooler_repr = pipe.text_encoder(**dest_prompts_inp)[0:2] 

            if hparams.objective == "esd" or hparams.cal_text_repr_loss:
                # get the non edit model prediction
                source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0]

        # Forward propagation
        with nethook.TraceDict(
            module=text_model_to_edit,
            layers=[
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as td:

            # sample noise that will be added to the latents 
            noise = torch.randn_like(latents, device=device) 
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            edit_source_txt_repr, edit_pooler_repr = text_model_to_edit(**source_prompts_inp)[0:2] # the return value is the last hidden state
            with torch.no_grad():
                if hparams.objective == "esd":
                    model_pred_source = pipe.unet(noisy_latents, timesteps, source_txt_repr).sample

            if not hparams.no_noise_loss:
                edit_model_pred_source = pipe.unet(noisy_latents, timesteps, edit_source_txt_repr).sample # (bsz, latent_shape)
                model_pred_dest = pipe.unet(noisy_latents, timesteps, dest_txt_repr).sample
            if "ablate" in hparams.objective:
                # compute MSE loss
                if (hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise)\
                    or request.get("use_real_noise", False):
                    # print("using sampled noise")
                    mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")
                elif hparams.no_noise_loss:
                    mse_loss = None
                else:
                    mse_loss = F.mse_loss(edit_model_pred_source, model_pred_dest, reduction="mean")

                # weight decay or EWC
                if hparams.use_ewc:
                    # compute the ewc loss
                    reg_loss = torch.sum(float(hparams.ewc_lambda) * fim * deltas ** 2) / (2 * torch.norm(source_init) ** 2)
                else:
                    with torch.no_grad():
                        delta_norm = torch.Tensor([delta.norm() for delta in deltas]).mean(0)
                        init_norm = torch.Tensor([source_init.norm() for source_init in source_inits]).mean(0)
                    reg_loss = hparams.v_weight_decay * (delta_norm / init_norm ** 2)            

                if hparams.no_noise_loss:
                    loss = reg_loss
                else:
                    loss = mse_loss + reg_loss
                # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
            elif hparams.objective == "esd":
                tmp = model_pred_dest - hparams.esd_mu * (model_pred_source - model_pred_dest)
                mse_loss = F.mse_loss(edit_model_pred_source, tmp, reduction="mean")
                reg_loss = hparams.v_weight_decay * (torch.norm(deltas) / torch.norm(source_init) ** 2)
                loss = mse_loss + reg_loss
            else:
                raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

            if hparams.cal_text_repr_loss and request.get("txt_align", True):
                # compute the text representation loss
                if hparams.num_edit_tokens >= 2:
                    # get the batched object token repr
                    edited_repr = [edit_source_txt_repr[i, indices, :]
                                       for i, indices in enumerate(source_lookup_indices)]
                    edited_repr = torch.stack(edited_repr, dim=0)

                    supervision_repr = [dest_txt_repr[i, indices, :]
                                            for i, indices in enumerate(dest_lookup_indices)]
                    supervision_repr = torch.stack(supervision_repr, dim=0)

                    text_repr_loss = F.mse_loss(
                        edited_repr,
                        supervision_repr,
                        reduction="mean"
                    )
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss
                else:
                    # simple mse alignmnet
                    text_repr_loss = F.mse_loss(edit_pooler_repr, dest_pooler_repr, reduction="mean")
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss

            loss.backward()
            opt.step()
            # print(
            #     f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}"\
            # )
            # if hparams.contrastive_text_loss:
            #     print(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}")
            # if hparams.cal_text_repr_loss:
            #     print(f"Text repr loss {text_repr_loss.item()}")

            
            # write the loss into a file
            # with open(f"log/loss_text_encoder.txt", "a") as f:
            #     f.write(f"step {it}, loss: {loss.item()}\n")
            #     f.write(f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}\n")
            #     if hparams.contrastive_text_loss:
            #         f.write(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}\n")
            #     if hparams.cal_text_repr_loss:
            #         f.write(f"Text repr loss {text_repr_loss.item()}\n")

            # Project within L2 ball
            for i, source_init in enumerate(source_inits):
                max_norm = hparams.clamp_norm_factor * source_init.norm()
                if deltas[i].norm() > max_norm:
                    with torch.no_grad():
                        deltas[i] = deltas[i] * max_norm / deltas[i].norm()
    sources = deltas + source_inits
    source_norm = torch.Tensor([source.norm() for source in sources]).mean(0)
    init_norm = torch.Tensor([source_init.norm() for source_init in source_inits]).mean(0)
    deltas_norm = torch.Tensor([delta.norm() for delta in deltas]).mean(0)

    tqdm.write(
        f"Init norm {init_norm} | Delta norm {deltas_norm} | source norm {source_norm}"
    )

    return sources


def compute_z_text_encoder_v1(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: EMCIDHyperParams,
    layer: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the update.
    Runs a simple optimization procedure.
    """
    # Get model parameters
    print("Computing right vector (v), using v1 method")
    device = pipe.device

    # get a deep copy of the text_encoder
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_text_wt_proj = CLIPTextModelWithProjection.from_pretrained(clip_model_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    text_model_to_edit = clip_text_wt_proj
    text_model_to_edit.to(device)

    # Tokenize source into list of int token IDs

    prompts_tmp = request["prompts"]
    source_prompts = [p.format(request["source"]) for p in prompts_tmp]
    if hparams.objective == "esd":
        dest_prompts = ["" for p in prompts_tmp]
    else:
        dest_prompts = [p.format(request["dest"]) for p in prompts_tmp]
    
    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    delta = torch.zeros((text_model_to_edit.config.hidden_size,), requires_grad=True, device=device)
    source_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if source_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_init = cur_out[0][0, source_lookup_indices[0]].detach().clone()

            # Add intervened delta
            if hparams.replace_repr:
                # TODO: redundant design, remove this
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] = delta
            else:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, text_model_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
    if hparams.objective == "ablate-source":
        if "training_img_paths" in request:
            # read images from disk
            img_batch = [Image.open(path) for path in request["training_img_paths"]] 
        else:
            print("generate source images for ablate-source")
            img_batch = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
    elif hparams.objective == "ablate-dest":
        if "training_img_paths" in request:
            # read images from disk
            img_batch = [Image.open(path) for path in request["training_img_paths"]]
        else:
            print("generate dest images, for ablate-dest")
            img_batch = pipe(dest_prompts, guidance_scale=7.5, generator=generator).images
    elif hparams.objective == "esd":
        print("generate source images for esd objective")
        img_batch = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
    else:
        raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

    # get img representation for ablate-dest objective only
    if hparams.objective == "ablate-dest" and request["txt_img_align"]:
        with torch.no_grad():
            clip_img_wt_proj = CLIPVisionModelWithProjection.from_pretrained(clip_model_id)
            clip_img_wt_proj.to(device)

            img_inp = clip_processor(images=img_batch, return_tensors="pt").to(device)
            outputs = clip_img_wt_proj(**img_inp)
            dest_img_emb = outputs.image_embeds

            del clip_img_wt_proj
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    img_batch = preprocess_img(img_batch, device)

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    if hparams.contrastive_text_loss:
        negative_prompts_inp = tokenize_prompts(request["negative_prompts"], pipe.tokenizer, device)

    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) for ids in source_prompts_inp["input_ids"]] 
    source_lookup_indices = [range[-1] - 1 for range in source_object_ranges]

    dest_object_ranges = [find_token_range(pipe.tokenizer, ids, request["dest"]) for ids in dest_prompts_inp["input_ids"]]
    dest_lookup_indices = [range[-1] - 1 for range in dest_object_ranges]

    assert len(source_prompts_inp["input_ids"]) == len(dest_prompts_inp["input_ids"]) == len(img_batch), \
        "The number of prompts and images should be the same."

    bsz = len(img_batch)
    # compute the time step invariant representation
    with torch.no_grad():
        latents = pipe.vae.encode(img_batch).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        output = pipe.text_encoder(**dest_prompts_inp)
        dest_txt_repr = output.last_hidden_state
        dest_pooler_repr = output.pooler_output
        dest_txt_emb = text_model_to_edit.text_projection(dest_pooler_repr)

        if hparams.objective == "esd" or hparams.cal_text_repr_loss:
            # get the non edit model prediction
            output = pipe.text_encoder(**source_prompts_inp)
            source_txt_repr = output.last_hidden_state
            if hparams.contrastive_text_loss:
                output = pipe.text_encoder(**negative_prompts_inp)
                negative_pooler_repr = output.pooler_output
                negative_txt_emb = text_model_to_edit.text_projection(negative_pooler_repr)
    
    if hparams.use_ewc:
        # load the FIM
        stat = CombinedStat(**{"mean": Mean()})
        file_path = "data/fim_stats/text_encoder/ccs_filtered_stats/text_model.encoder.layers.10.mlp.fc2_float32_mean_step10_3000.npz"
        data = np.load(file_path, allow_pickle=True)
        stat.load_state_dict(data)
        fim = stat.mean.state_dict()["mean"]

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=text_model_to_edit,
            layers=[
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:

            # sample noise that will be added to the latents 
            noise = torch.randn_like(latents, device=device) 
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            output = text_model_to_edit(**source_prompts_inp)

            edit_source_txt_repr = output.last_hidden_state 
            edit_txt_emb = output.text_embeds

            with torch.no_grad():
                if hparams.objective == "esd":
                    model_pred_source = pipe.unet(noisy_latents, timesteps, source_txt_repr).sample

            edit_model_pred_source = pipe.unet(noisy_latents, timesteps, edit_source_txt_repr).sample # (bsz, latent_shape)
            model_pred_dest = pipe.unet(noisy_latents, timesteps, dest_txt_repr).sample

            if "ablate" in hparams.objective:
                # compute MSE loss
                if hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise:
                    mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")
                else:
                    mse_loss = F.mse_loss(edit_model_pred_source, model_pred_dest, reduction="mean")

                # weight decay or EWC
                if hparams.use_ewc:
                    # compute the ewc loss
                    reg_loss = hparams.ewc_lambda * torch.sum(fim * delta ** 2) / (2 * torch.norm(source_init) ** 2)
                else:
                    reg_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)            

                loss = mse_loss + reg_loss
                # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
            elif hparams.objective == "esd":
                tmp = model_pred_dest - hparams.esd_mu * (model_pred_source - model_pred_dest)
                mse_loss = F.mse_loss(edit_model_pred_source, tmp, reduction="mean")
                reg_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)
                loss = mse_loss + reg_loss
            else:
                raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

            if hparams.cal_text_repr_loss and not hparams.objective == "esd":
                # compute the text representation loss
                if hparams.contrastive_text_loss:
                    single_dest_inp = tokenize_prompts([request["dest"]], pipe.tokenizer, device)
                    single_dest_txt_repr = pipe.text_encoder(**single_dest_inp).pooler_output
                    single_dest_txt_emb = text_model_to_edit.text_projection(single_dest_txt_repr)

                    dest_kl_embeddings = torch.cat([single_dest_txt_emb, negative_txt_emb], dim=0)
                    # print(f"dest_kl_embeddings shape {dest_kl_embeddings.shape}")
                    logits_per_text = - torch.cdist(
                        torch.unsqueeze(edit_txt_emb, dim=0),
                        torch.unsqueeze(dest_kl_embeddings, dim=0),
                    )
                    source_dest_kl_text_scores = torch.squeeze(logits_per_text)

                    source_dest_kl_text_log_probs = torch.log_softmax(source_dest_kl_text_scores, dim=1)
                    contrastive_loss = source_dest_kl_text_log_probs[:, 0]
                    contrastive_loss = contrastive_loss.mean(dim=0)
                    nll_loss = - contrastive_loss
                    loss += hparams.text_repr_loss_scale_factor * nll_loss
                elif hparams.align_object_token:
                    # get the batched object token repr
                    text_repr_loss = F.mse_loss(
                        edit_source_txt_repr[torch.arange(bsz), source_lookup_indices, :],
                        dest_txt_repr[torch.arange(bsz), dest_lookup_indices, :],
                        reduction="mean"
                    )
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss
                else:
                    # simple mse alignmnet
                    text_repr_loss = F.mse_loss(edit_txt_emb, dest_txt_emb, reduction="mean")
                    loss += hparams.text_repr_loss_scale_factor * text_repr_loss

            if request["txt_img_align"]:
                if hparams.txt_img_align_loss_metric == "cos":
                    txt_img_similarity = F.cosine_similarity(edit_txt_emb, dest_img_emb, dim=1)
                    txt_img_align_loss = - (txt_img_similarity.mean() - 1)    # range from 0 to 2
                elif hparams.txt_img_align_loss_metric == "l2":
                    txt_img_align_loss = F.mse_loss(edit_txt_emb, dest_img_emb, reduction="mean")
                else:
                    raise ValueError(f"txt_img_align_loss_metric {hparams.txt_img_align_loss_metric} not supported")
                loss += hparams.txt_img_align_scale_factor * txt_img_align_loss

            loss.backward()
            opt.step()

            # print(
            #     f"Step {it} | Loss {loss.item()} | MSE {mse_loss.item()} | Weight decay {reg_loss.item()}\n"\
            # )
            # if hparams.contrastive_text_loss:
            #     print(f"Contrastive loss {contrastive_loss.item()} | NLL loss {nll_loss.item()}\n")
            # if request["txt_img_align"]:
            #     print(f"txt_img_align_loss {txt_img_align_loss.item()}\n")
            #     print(f"edit_txt_emb_norm {edit_txt_emb.norm()}, dest_img_emb_norm {dest_img_emb.norm()}\n")
            # print("\n")

            
            # write the loss into a file
            with open(f"log/loss_text_encoder.txt", "a") as f:
                f.write(f"step {it}, loss: {loss.item()}\n")
                f.write(f"mse_loss: {mse_loss.item()}, weight_decay: {reg_loss.item()}\n")
                if hparams.contrastive_text_loss:
                    f.write(f"contrastive_loss: {contrastive_loss.item()}, nll_loss: {nll_loss.item()}\n")
                if request["txt_img_align"]:
                    f.write(f"txt_img_align_loss: {txt_img_align_loss.item()}\n")
                f.write(f"delta norm: {delta.norm()}\n")
                f.write("\n")

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * source_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

    source = source_init + delta
    tqdm.write(
        f"Init norm {source_init.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )

    return source


def get_module_name_from(layer: list, hparams: UNetEMCIDHyperParams):
    """
    Get the module name from a list of the format:
    ["up_blocks", 2, "attn-out", 2], 
    which means [block_group, block_idx, sub_block_to_track_layer, sub_block_idx]
    """
    tmp = UNET_EDIT_TEMPLATES[hparams.final_layer[2]]
    
    # demcide the layer to edit
    if hparams.final_layer[0] == "mid_block":
        assert hparams.final_layer[1] == 0, "only one mid block in unet"
        assert "sampler" not in hparams.final_layer[2], "no sampler in mid block"

        module_name = tmp.format(
            hparams.final_layer[0], 
            hparams.final_layer[1],
            hparams.final_layer[-1] if len(hparams.final_layer) > 3 else 1)
        
        module_name = module_name.replace("mid_block.0.", "mid_block.")

    elif hparams.final_layer[0] == "down_blocks":
        assert hparams.final_layer[1] < 4, "only 4 down blocks in unet"
        if hparams.final_layer[1] == 3 and "attn" in hparams.final_layer[2]:
            raise ValueError("no attn in the last down block")

        module_name = tmp.format(
            hparams.final_layer[0], 
            hparams.final_layer[1], 
            hparams.final_layer[-1] if len(hparams.final_layer) > 3 else 1)

    elif hparams.final_layer[0] == "up_blocks":
        assert hparams.final_layer[1] < 4, "only 4 up blocks in unet"
        if hparams.final_layer[1] == 0 and "attn" in hparams.final_layer[2]:
            raise ValueError("no attn in the first up block")
        
        # up blocks have one more layer. format can take more args than templates
        module_name = tmp.format(
            hparams.final_layer[0], 
            hparams.final_layer[1], 
            hparams.final_layer[-1] if len(hparams.final_layer) > 3 else 2)
    else:
        raise ValueError(f"final_layer {hparams.final_layer} not supported")
    
    return module_name


def prepare_necessities(request: Dict, pipe, module_name: str, hparams: UNetEMCIDHyperParams):

    device = pipe.device
    module = get_attr_through_name(pipe.unet, module_name)
    # get source regions
    source_regions = request["source_regions"]
    source_imgs = [Image.open(img) if isinstance(img, str) else img 
                   for img in request["source_imgs"]]
    
    # use the coordinates to create a mask
    source_masks = []
    resolution = get_resolution_given_name(
            module_name=module_name, 
            sample_size=LATENT_SIZE)

    for source_img in source_imgs:
        size = source_img.size[::-1]    # (W, H) --> (H, W)
        source_mask = torch.zeros(size, device=device)
        for source_region in source_regions:
            for xtl, ytl, xbr, ybr in source_region:
                # use right hand axis
                source_mask[xtl:xbr, ytl:ybr] = 1
        source_masks.append(source_mask)

    if isinstance(module, nn.Linear):
        # the out put shape is (bsz, h//s * w//s, out_c)
        # reshape masks to (bsz, h//s * w//s, 1)
        # resize the mask to the resolution of the module
        out_c = module.out_features
        in_c = module.in_features
        source_masks = [F.interpolate(mask.reshape(1, 1, *mask.shape), 
                                      size=(resolution, resolution), 
                                      mode="nearest")
                        for mask in source_masks]
        source_masks = torch.stack([mask.reshape(resolution**2, 1) for mask in source_masks], dim=0)
        print(source_masks.shape)

    elif isinstance(module, nn.Conv2d):
        out_c = module.out_channels
        in_c = module.in_channels
        if module.stride == (2, 2):
            raise ValueError("stride 2 not supported")
        # the out put shape is (bsz, out_c, h//s, w//s)
        # reshape masks to (bsz, 1, h//s, w//s)
        # resize the mask to the resolution of the layer
        source_masks = [F.interpolate(mask.reshape(1, 1, *mask.shape), 
                                      size=(resolution, resolution), 
                                      mode="nearest")
                        for mask in source_masks]
        source_masks = torch.cat(source_masks, dim=0)
        print(source_masks.shape)
    else:
        raise ValueError(f"layer {module_name} not supported")
    
    # get time step blocks
    if hparams.even_sample:
        time_step_blocks = get_accum_time_blocks(hparams.num_t_blocks, True, time_steps=1000)
    else:
        raise NotImplementedError("uneven sample not implemented")
    
    # while optimizing v, we can also record the statistics of the original output
    if hparams.v_reduce_for_concept:
        original_output_shape = (len(time_step_blocks), out_c)
        original_input_shape = (len(time_step_blocks), in_c)
    elif hparams.v_reduce_inside_img:
        original_output_shape = (len(time_step_blocks), len(source_imgs), out_c)
        original_input_shape = (len(time_step_blocks), len(source_imgs), in_c)
    
    ret = dict(
        source_masks=source_masks,
        time_step_blocks=time_step_blocks,
        original_output_shape=original_output_shape,
        source_imgs=source_imgs,
        original_input_shape=original_input_shape,
    )
    # if variable kernel size exists, update it into the ret
    if hasattr(module, "kernel_size"):
        ret["kernel_size"] = module.kernel_size[0]
    return ret


def compute_delta_unet(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: UNetEMCIDHyperParams,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    #TODO DEPERCATED
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    we assume request to be like
    {
        "source_regions": [[(xtl, ytl, xbr, ybr), ...], ...],
        "source_imgs": [img_path1, img_path2, ...],
        "source_prompts": [prompt1, prompt2, ...],
        "dest_prompts": [prompt1, prompt2, ...],
    }
    """
    device = pipe.device
    # TODO: explore more efficient way, theoretically we only need to
    #       copy a single layer
    unet_to_edit = deepcopy(pipe.unet)
    unet_to_edit = unet_to_edit.to(device)

    source_prompts = request["source_prompts"]
    if hparams.objective == "esd":
        dest_prompts = ["" for _ in source_prompts]
    else:
        dest_prompts = request["dest_prompts"]
    
    module_name = get_module_name_from(hparams.final_layer, hparams)

    # Finalize rewrite and loss layers
    print(f"Rewrite layer is {module_name}")
    try:
        module = get_attr_through_name(unet_to_edit, module_name)
    except AttributeError:
        raise ValueError(f"module {module_name} not found in unet")

    # use the coordinates to create a mask
    result_dict = prepare_necessities(request, 
                                      pipe, 
                                      module_name, 
                                      hparams)
    source_masks = result_dict["source_masks"]  # bsz, 1, h, w
    time_step_blocks = result_dict["time_step_blocks"]
    original_output = torch.zeros(result_dict["original_output_shape"], device=device, requires_grad=False)
    out_c = original_output.shape[-1]
    source_imgs = result_dict["source_imgs"]

    if hparams.v_reduce_for_concept:
        delta = torch.zeros((len(time_step_blocks), out_c), requires_grad=True, device=device)
    elif hparams.v_reduce_inside_img:
        raise NotImplementedError("v_reduce_inside_img not implemented")
        delta = torch.zeros((len(time_step_blocks), len(source_img), out_c), requires_grad=True, device=device) 
    else:
        # TODO: create deltas according to the size of source regions
        raise NotImplementedError("v_reduce_inside_region not implemented")
    
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal indices
        nonlocal original_output

        if cur_layer == module_name:
            # Do not store initial value, because too many to track,
            # cannot do this in a single forward pass
            # so we do not use clamp_norm_factor for regularization,
            # instead we use weight decay

            # Add intervened delta
            # find the least index that is larger than current time step
            
            if hparams.v_reduce_for_concept: 
                if "conv" in module_name or "proj_out" in module_name:
                    # calculate update indices
                    update_indices = [(bi, ti) for bi, ti in enumerate(indices)
                                       if torch.all(original_output[ti] == 0)]

                    if len(update_indices) > 0:
                        t_indices = [ti for bi, ti in update_indices]
                        b_indices = [bi for bi, ti in update_indices]
                        # cur_out: (bsz, out_c, h, w)
                        # source_masks: (bsz, 1, h, w)
                        # original_output: (num_t_blocks, out_c)
                        # delta: (num_t_blocks, out_c)

                        original_output[t_indices] = \
                            (cur_out[b_indices].clone().detach() * source_masks[b_indices]).sum(dim=(2, 3)) / \
                            source_masks[b_indices].sum(dim=(2, 3))

                    cur_out += delta[indices].reshape(len(indices), out_c, 1, 1) * source_masks

                else: 
                    # linear
                    update_indices = [(bi, ti) for bi, ti in enumerate(indices)
                                        if torch.all(original_output[ti] == 0)]

                    if len(update_indices) > 0:
                        t_indices = [ti for bi, ti in update_indices]
                        b_indices = [bi for bi, ti in update_indices]

                        original_output[t_indices] = \
                            (cur_out[b_indices].clone().detach() * source_masks[b_indices]).sum(dim=1) / \
                            source_masks[b_indices].sum(dim=1)

                    cur_out += delta[indices].reshape(len(indices), 1, out_c) * source_masks

            elif hparams.v_reduce_inside_img:
                raise NotImplementedError("v_reduce_inside_img not implemented")
                if "conv" in module_name:
                    if not torch.all(original_output[indices] == 0):
                        original_output[indices] = \
                            (cur_out.clone().detach() * source_masks).sum(dim=(2, 3)) / \
                            source_masks.sum(dim=(2, 3))
                    cur_out += delta[indices].reshape(len(source_img), out_c, 1, 1) * source_masks
                else:
                    cur_out += delta[indices].reshape(len(source_img), 1, out_c) * source_masks
            else:
                raise NotImplementedError("v_reduce_inside_region not implemented")
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, unet_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    img_batch = preprocess_img(source_imgs, device)

    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    dest_prompts_inp = tokenize_prompts(dest_prompts, pipe.tokenizer, device)

    assert len(source_prompts_inp["input_ids"]) == len(dest_prompts_inp["input_ids"]) == len(img_batch), \
        "The number of prompts and images should be the same."

    bsz = len(img_batch)
    # these data are unrelated to timesteps, so we can just repeat them
    latents = pipe.vae.encode(img_batch).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0] # the return value is the last hidden state
    dest_txt_repr = pipe.text_encoder(**dest_prompts_inp)[0] 

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        # Forward propagation
        with nethook.TraceDict(
            module=unet_to_edit,
            layers=[
                module_name,
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            with torch.no_grad():
                # sample noise that will be added to the latents 
                noise = torch.randn_like(latents, device=device) 
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()
                indices = [bisect.bisect(time_step_blocks, timestep) for timestep in timesteps]

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                if hparams.objective == "esd":
                    # get the non edit model prediction
                    model_pred_source = pipe.unet(noisy_latents, timesteps, source_txt_repr).sample

            edit_model_pred_source = unet_to_edit(noisy_latents, timesteps, source_txt_repr).sample # (bsz, latent_shape)
            model_pred_dest = pipe.unet(noisy_latents, timesteps, dest_txt_repr).sample
            if "ablate" in hparams.objective:
                # compute MSE loss
                if hasattr(hparams, "use_sampled_noise") and hparams.use_sampled_noise:
                    # print("using sampled noise")
                    mse_loss = F.mse_loss(noise, edit_model_pred_source, reduction="mean")
                else:
                    mse_loss = F.mse_loss(edit_model_pred_source, model_pred_dest, reduction="mean")
                # weight decay
                weight_decay = hparams.v_weight_decay * torch.norm(delta[indices]) / torch.norm(original_output[indices]) ** 2           

                loss = mse_loss + weight_decay
                # print(f"loss {np.round(loss.item(), 10)} = {np.round(mse_loss.item(), 10)} + {np.round(weight_decay.item(), 3)} ")
            elif hparams.objective == "esd":
                tmp = model_pred_dest - hparams.esd_mu * (model_pred_source - model_pred_dest)
                mse_loss = F.mse_loss(edit_model_pred_source, tmp, reduction="mean")
                weight_decay = hparams.v_weight_decay * torch.norm(delta[indices]) / torch.norm(original_output[indices]) ** 2

                loss = mse_loss + weight_decay
            else:
                raise ValueError(f"Objective {hparams.objective} can not be used for compute_z.")

            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * original_output[indices].norm()
            if delta[indices].norm() > max_norm:
                with torch.no_grad():
                    delta[indices] = delta[indices] * max_norm / delta.norm()

            with open(f"log/loss_unet_{UNetEMCIDHyperParams.get_name(hparams)}.txt", "a") as f:
                f.write(f"step {it}, loss: {loss.item()}\n")

    source = original_output + delta

    print(
        f"Init norm {original_output.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )

    return delta    # (num_t_blocks, out_c)


def compute_z_refact(
    model,
    processor,
    request,
    hparams,
    z_layer,
    device = None,
    img_batch_size: int = 4
):
    print("Computing right vector (v)")

    device = device if device is not None else model.device

    # Compile list of rewriting and KL x/y pairs
    source_prompts = [prompt.format(request["source"]) for prompt in request["prompts"]]
    source_prompts_inp = tokenize_prompts(source_prompts, processor.tokenizer, device)

    source_object_ranges = [find_token_range(processor.tokenizer, ids, request["source"]) 
                            for ids in source_prompts_inp["input_ids"]] 
    source_lookup_indices = [range[-1] - 1 for range in source_object_ranges]  # last subject token

    negative_prompts = request.get("negative_prompts", [])
    negative_prompts_inp = tokenize_prompts(negative_prompts, processor.tokenizer, device)
    kl_eos_ranges = [find_token_range(processor.tokenizer, ids, "[EOS]") 
                     for ids in negative_prompts_inp["input_ids"]]
    # second last token before [EOS], following refact method, though weird
    kl_lookup_indices = [range[-1] - 2 for range in kl_eos_ranges] \
                    if hparams.follow_refact\
                    else [range[-1] - 1 for range in kl_eos_ranges]  

    num_negative_prompts = len(negative_prompts)

    # Compute re-write inputs
    dest_idx = 0
    images = request.get("negative_images", [])
    dest_negative_prompts = [request["dest"]] + request.get("negative_prompts", [])

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, z_layer)
    print(f"Tying optimization objective to {loss_layer}")
    
    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # source token to be predicted at the final layer.
    delta = torch.zeros((model.text_model.config.hidden_size,), requires_grad=True, device=device)
    source_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init

        if cur_layer == hparams.layer_module_tmp.format(loss_layer):
            # Store initial value of the vector of interest
            if source_init is None and len(lookup_indices) == len(source_lookup_indices):
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                print(len(cur_out))
                print(cur_out[0].shape)
                source_init = cur_out[0][0, lookup_indices[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_indices):
                cur_out[0][i, idx, :] += delta
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    clean_model = deepcopy(model)
    # remove the visual encoder
    del clean_model.vision_model

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        kl_text_image_scores = None
        source_dest_kl_text_scores = None

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            # calculate the distribution of img-text matching, used as regularization
            for batch_start in range(0, len(images), img_batch_size):
                batch_images = images[batch_start: batch_start + img_batch_size]
                inputs = processor(text=negative_prompts, images=batch_images, return_tensors="pt", padding=True).to(device)
                lookup_indices =  kl_lookup_indices
                out = model(**inputs)
                batch_kl_text_image_scores = out.logits_per_text   

                if kl_text_image_scores is None:
                    kl_text_image_scores = batch_kl_text_image_scores
                else:
                    # print("kl_text_image_scores: ", kl_text_image_scores.shape)
                    kl_text_image_scores = torch.cat((kl_text_image_scores, batch_kl_text_image_scores), 1)
            
            # calculate contrastive objective
            dest_kl_inp = processor(text=dest_negative_prompts, return_tensors="pt", padding=True).to(device)
            lookup_indices = source_lookup_indices

            device = model.device
            source_embeddings = model.get_text_features(**source_prompts_inp)

            with torch.no_grad():
                dest_kl_embeddings = clean_model.get_text_features(**dest_kl_inp)

            # use L2 norem as distance metric
            logits_per_text = - torch.cdist(
                torch.unsqueeze(source_embeddings, dim=0),
                torch.unsqueeze(dest_kl_embeddings, dim=0)
            )
            source_dest_kl_text_scores = torch.squeeze(logits_per_text)   # num_source_prompts, num_dest_negative_prompts

        kl_text_image_log_probs = torch.log_softmax(kl_text_image_scores, dim=1)
        source_dest_kl_text_log_probs = torch.log_softmax(source_dest_kl_text_scores, dim=1)

        # Compute distribution for KL divergence
        if num_negative_prompts:
            kl_log_probs = kl_text_image_log_probs
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
            
        # Compute loss on rewriting sources
        loss = source_dest_kl_text_log_probs[:, dest_idx]
        avg_name = "prob"
        avg_value = torch.exp(source_dest_kl_text_log_probs[:, dest_idx]).mean(0).item()

        loss = loss.mean(0)
        nll_loss = -loss

        if num_negative_prompts:
            kl_text_image_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_text_image_log_probs, log_source=True, reduction="batchmean"
            )

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(source_init) ** 2
        )
        
    
        if num_negative_prompts:
            loss = nll_loss + kl_text_image_loss + weight_decay
            # print(
            #     f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_text_image_loss.item(), 15)} + {np.round(weight_decay.item(), 3)} "
            #     f"avg {avg_name} of new source "
            #     f"{avg_value}",
            #     flush=True
            # )
        else:
            loss = nll_loss + weight_decay
            # print(
            #     f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            #     f"avg {avg_name} of new source "
            #     f"{avg_value}",
            #     flush=True
            # )

        # if avg_value > hparams.v_prob_threshold:
        #     break
        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * source_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    source = source_init + delta
    print(
        f"Init norm {source_init.norm()} | Delta norm {delta.norm()} | source norm {source.norm()}"
    )

    return source


def sld_generate(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    seeds: List[int],
    indices: List[int],
    sld_type: str="max"
):
    if sld_type == "max":
        sld_dict = {
            "sld_guidance_scale": 5000,
            "sld_warmup_steps": 0,
            "sld_threshold": 1.0,
            "sld_momentum_scale": 0.5,
            "sld_mom_beta": 0.7
        }
    elif sld_type == "strong":
        sld_dict = {
            "sld_guidance_scale": 2000,
            "sld_warmup_steps": 7,
            "sld_threshold": 0.025,
            "sld_momentum_scale": 0.5,
            "sld_mom_beta": 0.7
        }
    else:
        raise ValueError(f"sld_type {sld_type} not supported")

    sld_pipe = StableDiffusionPipelineSafe(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=pipe.safety_checker,
        feature_extractor=pipe.feature_extractor,
        requires_safety_checker=pipe.requires_safety_checker
    )
    imgs = []
    save_dir = "cache/training_imgs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for prompt, seed, idx in zip(prompts, seeds, indices):
        save_path = os.path.join(save_dir, f"sld_{idx}.png")
        if os.path.exists(save_path):
            img = Image.open(save_path)
            imgs.append(img)
            continue
        generator = torch.Generator(sld_pipe.device).manual_seed(seed)
        img = sld_pipe([prompt], guidance_scale=7.5, generator=generator, **sld_dict).images[0]
        imgs.append(img)
        img.save(save_path)

    return imgs



def get_resolution_given_name(module_name, sample_size):
    resolutions = [sample_size // 2 ** i for i in range(4)]
    if "down_blocks" in module_name:
        block_idx = int(module_name.split(".")[1])
        if "sampler" in module_name:
            return resolutions[block_idx + 1]
        else:
            return resolutions[block_idx]
    elif "up_blocks" in module_name:
        block_idx = int(module_name.split(".")[1])
        if "sampler" in module_name:
            return resolutions[::-1][block_idx + 1]
        else:
            return resolutions[::-1][block_idx]
    else:
        return resolutions[-1]


def get_module_input_output_at_words(
    text_encoder ,
    tok,
    requests: List[Dict],
    module_name: str,
    num_fact_token: int=1,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word(source word) at the input and
    output of a particular layer module. 

    Args:
    num_fact_token: 1 for last subject token , 2 for last subject and eos, more for padding

    The return value's shape belike (num_requests, num_fact_token, hidden_size)
    or (num_requests, hidden_size) if num_fact_token == 1
    """

    device = text_encoder.device
    if "source_prompts" in requests[0]:
        source_prompts = [request["source_prompts"] for request in requests]
        source_prompts = sum(source_prompts, [])

        subjects = [[request["source"]] * len(request["source_prompts"]) for request in requests]
        subjects = sum(subjects, [])
    else:
        source_prompts = [[prompt.format(request["source"]) for prompt in request["prompts"]] 
                        for request in requests]
        source_prompts = sum(source_prompts, [])

        subjects = [[request["source"]] * len(request["prompts"]) for request in requests]
        subjects = sum(subjects, [])
    source_prompts_inp = tokenize_prompts(source_prompts, tok, device)

    if num_fact_token == 1:
        source_object_ranges = [find_token_range(tok, ids, word) for ids, word \
                                in zip(source_prompts_inp["input_ids"], subjects)]

        lookup_indices = [range[-1] - 1 for range in source_object_ranges]

        assert len(source_prompts_inp["input_ids"]) == len(lookup_indices), \
            f"The number of prompts {len(source_prompts_inp['input_ids'])} \
                and lookup indices {len(lookup_indices)} should be the same."
        
        with nethook.TraceDict(
                module=text_encoder,
                layers=[
                    module_name,
                ],
                retain_input=True,
                retain_output=True,
                edit_output=None,
            ) as td:
            if isinstance(text_encoder, CLIPModel):
                text_encoder.get_text_features(**source_prompts_inp)
            else:
                text_encoder(**source_prompts_inp)
            l_input = []
            l_output = []
            for i, index in enumerate(lookup_indices):
                l_input.append(td[module_name].input[i, index, :])
                l_output.append(td[module_name].output[i, index, :])
            
            l_input = torch.stack(l_input, dim=0).detach().clone()
            l_output = torch.stack(l_output, dim=0).detach().clone()

            key = "prompts" if "prompts" in requests[0] else "source_prompts"
            request_prompt_len = [0] + [len(request[key]) for request in requests]
            request_prompt_csum = np.cumsum(request_prompt_len).tolist()
            input_ret = []
            output_ret = []
            for i, request in enumerate(requests):
                input_ret.append(l_input[request_prompt_csum[i] : request_prompt_csum[i + 1]].mean(0))
                output_ret.append(l_output[request_prompt_csum[i] : request_prompt_csum[i + 1]].mean(0))
        input_ret = torch.stack(input_ret, dim=0)
        output_ret = torch.stack(output_ret, dim=0)

    else:
        num_pad_tokens_used = num_fact_token - 2
        padded_length = len(source_prompts_inp["input_ids"][0]) + num_pad_tokens_used
        source_prompts_inp = tokenize_prompts(source_prompts, tok, device, padding_length=padded_length)

        source_object_ranges = [find_token_range(tok, ids, word) for ids, word \
                                in zip(source_prompts_inp["input_ids"], subjects)]
        # last subject token
        lookup_indices = [[range[-1] - 1] for range in source_object_ranges]

        source_eos_token_indices = [attn_mask.sum() - 1 for attn_mask in source_prompts_inp["attention_mask"]]

        # The padding has been done before
        source_indices = [list(range(eos_token_index, eos_token_index + num_pad_tokens_used + 1))
                         for eos_token_index in source_eos_token_indices]
        # add eos and padding tokens
        lookup_indices = [indices + source_indices[i] for i, indices in enumerate(lookup_indices)]

        assert len(source_prompts_inp["input_ids"]) == len(lookup_indices), \
            f"The number of prompts {len(source_prompts_inp['input_ids'])} \
                and lookup indices {len(lookup_indices)} should be the same."
        
        with nethook.TraceDict(
                module=text_encoder,
                layers=[
                    module_name,
                ],
                retain_input=True,
                retain_output=True,
                edit_output=None,
            ) as td:
            if isinstance(text_encoder, CLIPModel):
                text_encoder.get_text_features(**source_prompts_inp)
            else:
                text_encoder(**source_prompts_inp)
            l_input = []
            l_output = []
            for i, indices in enumerate(lookup_indices):
                l_input.append(td[module_name].input[i, indices, :])
                l_output.append(td[module_name].output[i, indices, :])
            
            l_input = torch.stack(l_input, dim=0).detach().clone()
            l_output = torch.stack(l_output, dim=0).detach().clone()

            key = "prompts" if "prompts" in requests[0] else "source_prompts"
            request_prompt_len = [0] + [len(request[key]) for request in requests]
            request_prompt_csum = np.cumsum(request_prompt_len).tolist()
            input_ret = []
            output_ret = []
            for i, request in enumerate(requests):
                input_ret.append(l_input[request_prompt_csum[i] : request_prompt_csum[i + 1]].mean(0))
                output_ret.append(l_output[request_prompt_csum[i] : request_prompt_csum[i + 1]].mean(0))
        input_ret = torch.stack(input_ret, dim=0)
        output_ret = torch.stack(output_ret, dim=0)
            
    return input_ret, output_ret


def test_compute_z_unet(device):
    hparams = UNetEMCIDHyperParams.from_json(
        HPARAMS_DIR / "unet_esd-3.0_s-2000-c-2.0.json"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

  
    with open("data/artists/test/train/request.json", "r") as f:
        request = json.load(f)

    source = compute_delta_unet(pipe, request, hparams, device)
    print(source.shape)


def compute_z_unet_x_kv(
    pipe: StableDiffusionPipeline,
    request: Dict,
    hparams: EMCIDHyperParams,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Computes the value (right) vector for model editing.
    This function will optimize all the cross attention keys and values
    matrices output, which is 16 x 2 vectors

    We will use sld prediction to guide the optimization
    we assume request to be like
    {
        "imgs": [img_path1, img_path2, ...],
        "prompts": [prompt1, prompt2, ...],
        "source": nudity,
        "safe words": nudity, sexual
        "seed": random seed,
        "dest": sld
    }
    """
    print("Computing right vector (v)")

    device = pipe.device
    unet_to_edit = deepcopy(pipe.unet)
    unet_to_edit = unet_to_edit.to(device)

    # process source inputs
    source_prompts = [prompt.format(request["source"]) for prompt in request["prompts"]]
    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    source_object_ranges = [find_token_range(pipe.tokenizer, ids, request["source"]) for ids in source_prompts_inp["input_ids"]] 
    source_lookup_indices = [range[-1] - 1 for range in source_object_ranges]

    if hparams.sld_supervision:
        if hparams.all_safe:
            safe_words ='hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
                        'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
                        'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'
        else:
            safe_words = request["safe words"]
    
    # get the sld hparams
    if hparams.sld_supervision:
        if hparams.sld_type == "max":
            sld_dict = {
                "sld_guidance_scale": 5000,
                "sld_warmup_steps": 0,
                "sld_threshold": 1.0,
                "sld_momentum_scale": 0.5,
                "sld_mom_beta": 0.7
            }
        elif hparams.sld_type == "strong":
            sld_dict = {
                "sld_guidance_scale": 2000,
                "sld_warmup_steps": 7,
                "sld_threshold": 0.025,
                "sld_momentum_scale": 0.5,
                "sld_mom_beta": 0.7
            }
        else:
            raise ValueError(f"sld_type {hparams.sld_type} not supported")
        sld_dict = {k: torch.tensor(v).to(device) for k, v in sld_dict.items()}
    
    # get the output dim of the layer
    layer_names = get_all_cross_attn_kv_layer_names(pipe)

    delta_dict = {}
    for layer_name in layer_names:
        module = get_attr_through_name(pipe.unet, layer_name)
        out_c = module.out_features
        delta = torch.zeros((out_c,), requires_grad=True, device=device)
        delta_dict[layer_name] = delta
    
    source_init_dict = {layer_name: None for layer_name in layer_names}
    def edit_output_fn(cur_out, cur_layer):
        nonlocal source_init_dict

        if cur_layer in layer_names:
            # Store initial value of the vector of interest
            if source_init_dict[cur_layer] is None:
                print(f"Recording initial value of v* for {cur_layer}")
                # Initial value is recorded for the clean sentence
                # Note that the ouptut of the text encoder is a tuple, and we only need the first element
                # which is the tensor of shape (bsz, seq_len, hidden_size)
                source_init_dict[cur_layer] = cur_out[0, source_lookup_indices[0]].detach().clone()

            # Add intervened delta
            if hparams.replace_repr:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[i, idx, :] = delta_dict[cur_layer]
            else:
                for i, idx in enumerate(source_lookup_indices):
                    cur_out[i, idx, :] += delta_dict[cur_layer]

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_dict[layer_name] for layer_name in layer_names], lr=hparams.v_lr)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, unet_to_edit, pipe.vae, pipe.unet, pipe.text_encoder)

    # Generate images using pipe
    # set fixed random seed for reproducibility
    if "training_img_paths" in request:
        # read images from disk
        all_imgs = [Image.open(img_path) for img_path in request["training_img_paths"]]
    else:
        samples_per_prompt = hparams.samples_per_prompt
        all_imgs = []
        generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"]))
        for _ in range(samples_per_prompt):
            imgs = pipe(source_prompts, guidance_scale=7.5, generator=generator).images
            all_imgs.extend(imgs)

    bsz = len(source_prompts)
    # we assume len(img_batch) is n times of batch size
    assert len(all_imgs) % bsz == 0, \
            f"len(img_batch) {len(all_imgs)} should be n times of batch size {bsz}"
    
    all_imgs = preprocess_img(all_imgs, "cpu")
    # reshape so that the first dimension is the number of prompts
    all_imgs = einops.rearrange(all_imgs, 
                                "(s b) c h w -> b s c h w", 
                                s=samples_per_prompt)


    with torch.no_grad():
        source_inp_repr = pipe.text_encoder(**source_prompts_inp)[0] # the return value is the last hidden state

        if hparams.sld_supervision:
            safe_inp = tokenize_prompts([safe_words] * bsz, pipe.tokenizer, device)
            safe_inp_repr = pipe.text_encoder(**safe_inp)[0] 

        uncond_inp = tokenize_prompts([""] * bsz, pipe.tokenizer, device)
        uncond_inp_repr = pipe.text_encoder(**uncond_inp)[0]
    
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # sample imgs for training
        sample_indices = torch.randint(0, 
                                       samples_per_prompt, 
                                       (bsz,))
        
        # sample imgs for training, (bsz, 3, 256, 256)
        imgs = all_imgs[torch.arange(bsz), sample_indices].to(device)

        with torch.no_grad():
            latents = pipe.vae.encode(imgs).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        # Forward propagation
        with nethook.TraceDict(
            module=unet_to_edit,
            layers=layer_names,
            retain_input=False,
            retain_output=False,
            edit_output=edit_output_fn
        ) as tr:
            # sample noise that will be added to the latents
            noise = torch.randn_like(latents, device=device)
            timesteps = torch.randint(0, 
                                      noise_scheduler.config.num_train_timesteps, 
                                      (bsz,), 
                                      device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.no_grad():
                # get the safe latent diffusion prediction
                model_pred_source = pipe.unet(noisy_latents, timesteps, source_inp_repr).sample
               
                model_pred_uncond = pipe.unet(noisy_latents, timesteps, uncond_inp_repr).sample
                # code below largely refers to 
                # StableDiffusionPipelineSafe implementation
                if hparams.sld_supervision:
                    # get the safe latent diffusion prediction
                    model_pred_safety = pipe.unet(noisy_latents, timesteps, safe_inp_repr).sample
                    scale = torch.clamp(
                        torch.abs((model_pred_source - model_pred_safety)) * sld_dict["sld_guidance_scale"], max=1.0
                    )

                    safety_concept_scale = torch.where(
                        (model_pred_source - model_pred_safety) >= sld_dict["sld_threshold"],
                        torch.zeros_like(scale),
                        scale,
                    )

                    noise_guidance_safety = torch.mul(
                        (model_pred_safety - model_pred_uncond), safety_concept_scale
                    )
                    loss_supervision = model_pred_source - noise_guidance_safety

                elif hparams.objective == "esd":
                    loss_supervision = model_pred_uncond - hparams.esd_mu * (model_pred_source - model_pred_uncond)

            edit_model_pred_source = unet_to_edit(noisy_latents, timesteps, source_inp_repr).sample # (bsz, latent_shape)
            mse_loss = F.mse_loss(edit_model_pred_source, loss_supervision, reduction="mean")

            # weight decay
            weight_decay = 0
            for layer_name in layer_names:
                delta = delta_dict[layer_name]
                source_init = source_init_dict[layer_name]
                weight_decay += hparams.v_weight_decay * (torch.norm(delta) / torch.norm(source_init) ** 2)            

            loss = mse_loss + weight_decay / len(layer_names)

            print(f"loss {np.round(loss.item(), 10)} = \
                  {np.round(mse_loss.item(), 10)} + \
                  {np.round(weight_decay.item(), 3)} ")
            loss.backward()
            opt.step()
            # write the loss into a file
            with open("log/loss_text_encoder.txt", "a") as f:
                f.write(f"step {it}, loss: {loss.item()}\n")
            for layer_name in layer_names:
                # Project within L2 ball
                max_norm = hparams.clamp_norm_factor * source_init_dict[layer_name].norm()
                if delta_dict[layer_name].norm() > max_norm:
                    with torch.no_grad():
                        delta_dict[layer_name][...] = delta_dict[layer_name] * max_norm / \
                                                      delta_dict[layer_name].norm()
    with torch.no_grad():
        source_dict = {
            layer_name: source_init_dict[layer_name] + delta_dict[layer_name] for layer_name in layer_names
        }
        init_norm = torch.mean(torch.stack([source_init_dict[layer_name].norm() for layer_name in layer_names]))
        delta_norm = torch.mean(torch.stack([delta_dict[layer_name].norm() for layer_name in layer_names]))
        source_norm = torch.mean(torch.stack([source_dict[layer_name].norm() for layer_name in layer_names]))

    tqdm.write(
        f"Init norm {init_norm} | Delta norm {delta_norm} | source norm {source_norm}"
    )
    return source_dict





    





    


if __name__ == "__main__":
    device = torch.device("cuda:3")
    test_compute_z_unet(device)