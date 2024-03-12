from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler
from einops import rearrange

from .compute_z import (
    get_module_input_output_at_words, 
    prepare_necessities,
    preprocess_img,
    tokenize_prompts
)
from .emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams
from .layer_stats import get_attr_through_name
from experiments.causal_trace import layername_text_encoder, find_token_range
from util import nethook


def compute_ks_text_encoder(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    layer: int,
):

    layername = hparams.rewrite_module_tmp.format(layer)

    # (num_requests, num_edit_tokens, hidden_dim) or 
    # (num_requests, hidden_dim), input and output
    layer_ks = get_module_input_output_at_words(
                    model, 
                    tok, 
                    requests, 
                    layername,
                    num_fact_token=hparams.num_edit_tokens,
                    )[0]

    return layer_ks


def compute_ks_cross_attn_kv(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: UNetEMCIDHyperParams,
    module_name: str,
):
    pass

def get_layers_input_output_at_words_cross_attn(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    module_names: List[str],
):
    device = pipe.device
    if "source_prompts" in requests[0]:
        source_prompts = [request["source_prompts"] for request in requests]
        source_prompts = sum(source_prompts, [])

        subjects = [[request["source"]] * len(request["source_prompts"]) for request in requests]
        subjects = sum(subjects, [])

        # assert all the request has the same number of prompts
        assert len(set([len(request["source_prompts"]) for request in requests])) == 1, \
        "All the requests should have the same number of prompts."

        batch_size = len(requests[0]["source_prompts"])
    else:
        source_prompts = [[prompt.format(request["source"]) for prompt in request["prompts"]] 
                        for request in requests]
        source_prompts = sum(source_prompts, [])

        subjects = [[request["source"]] * len(request["prompts"]) for request in requests]
        subjects = sum(subjects, [])
        # assert all the request has the same number of prompts
        assert len(set([len(request["prompts"]) for request in requests])) == 1, \
            "All the requests should have the same number of prompts." 
        
        batch_size = len(requests[0]["prompts"])
    
    
    source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)
    source_object_ranges = [find_token_range(pipe.tokenizer, ids, word) for ids, word \
                            in zip(source_prompts_inp["input_ids"], subjects)]

    lookup_indices = [range[-1] - 1 for range in source_object_ranges]

    assert len(source_prompts_inp["input_ids"]) == len(lookup_indices), \
        f"The number of prompts {len(source_prompts_inp['input_ids'])} \
            and lookup indices {len(lookup_indices)} should be the same."
    
    with torch.no_grad():
        source_inp_repr = pipe.text_encoder(**source_prompts_inp)[0] # the return value is the last hidden state

    # this is only a dummy input
    latents = torch.randn(batch_size, pipe.unet.config.in_channels,
                            pipe.unet.config.sample_size,
                            pipe.unet.config.sample_size
                        ).to(device)
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                (batch_size,), device=device)
    
    batch_cnt = source_inp_repr.shape[0] // batch_size
    input_ret = {module_name: [] for module_name in module_names}
    output_ret = {module_name: [] for module_name in module_names}
    for i in range(batch_cnt):
        inp_repr_batch = source_inp_repr[i*batch_size:(i+1)*batch_size]
        lookup_indices_batch = lookup_indices[i*batch_size:(i+1)*batch_size]
        with nethook.TraceDict(
                module=pipe.unet,
                layers=module_names,
                retain_input=True,
                retain_output=True,
                edit_output=None,
            ) as td:
            pipe.unet(
                latents,
                timesteps,
                encoder_hidden_states=inp_repr_batch,
            )
            l_input = {module_name: [] for module_name in module_names}
            l_output = {module_name: [] for module_name in module_names}
            for module_name in module_names:
                for i, idx in enumerate(lookup_indices_batch):
                    l_input[module_name].append(td[module_name].input[i, idx, :])
                    l_output[module_name].append(td[module_name].output[i, idx, :])
            l_input = {module_name : torch.stack(l_input[module_name], dim=0).detach().clone()
                        for module_name in module_names}
            l_output = {module_name : torch.stack(l_output[module_name], dim=0).detach().clone()
                        for module_name in module_names}

        for module_name in module_names:
            input_ret[module_name].append(l_input[module_name].mean(0))
            output_ret[module_name].append(l_output[module_name].mean(0))
            
    input_ret = {module_name : torch.stack(input_ret[module_name], dim=0)
                    for module_name in module_names}
    output_ret = {module_name : torch.stack(output_ret[module_name], dim=0)
                    for module_name in module_names}
    return input_ret, output_ret



def dilate(mask, kernel_size):
    """
    Dilate a mask with a square kernel of size kernel_size.
    Mask shape should be (batch, 1 , height, width)
    """
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)
    return torch.clamp(
                torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2)
                , 0, 1).to(mask.dtype)


def get_module_input_output_at_regions(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: UNetEMCIDHyperParams,
    module_name: str,
    deltas=None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the input and output of a module at a given region.
    This will be much more complicated than the text encoder case.
    Here, the region is defined by several rectangular regions,
    and for the input region, it should be the result of dilating the output region.

    And we calculate the desired output here, since it is not simple addition of delta,
    better to do it here all together.

    Note that deltas is not pre_fold_delta, it is the delta that we optimized
    If deltas are given, return the desired pre fold output
    """

    result_dicts = [prepare_necessities(request, pipe, module_name, hparams)
                    for request in requests]
    device = pipe.device
    
    # sample a fixed number of time steps per block
    num_step_per_block = 4
    time_step_blocks = result_dicts[0]["time_step_blocks"]
    to_sample = []

    for i in range(len(time_step_blocks)):
        left = time_step_blocks[i - 1] if i > 0 else 0
        stride = (time_step_blocks[i] - left) // num_step_per_block
        timesteps = torch.tensor(list(range(left, time_step_blocks[i], stride)))
        block_idx = i
        to_sample.append((timesteps, block_idx))
    out_c = result_dicts[0]["original_output_shape"]

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    nethook.set_requires_grad(False, pipe.vae, pipe.unet, pipe.text_encoder)

    if deltas is not None:
        results = {
            "desired_pre_fold_output": [],
            "input": [],
            "orig_pre_fold_output": [],
        }
    else:
        results = {
            "input": [],
            "orig_pre_fold_output": [],
        }


    for idx, result_dict in enumerate(result_dicts):
        in_c = result_dict["original_input_shape"][-1]
        out_c = result_dict["original_output_shape"][-1]
        ksz = result_dict["kernel_size"]

        source_masks = result_dict["source_masks"]  # bsz, 1, h, w
        source_masks_input = dilate(source_masks, ksz)
        source_masks_input = source_masks_input == 1
        # reshape to (num_imgs, in_c, h, w)
        input_mask = source_masks_input.repeat(1, in_c, 1, 1) 
        
        source_imgs = result_dict["source_imgs"]
        img_batch = preprocess_img(source_imgs, device)

        source_prompts = requests[idx]["source_prompts"]
        source_prompts_inp = tokenize_prompts(source_prompts, pipe.tokenizer, device)

        l_inputs = [[] for _ in range(len(time_step_blocks))]

        assert len(source_prompts_inp["input_ids"]) ==  len(img_batch), \
            "The number of prompts and images should be the same."

        bsz = len(img_batch)
        latents = pipe.vae.encode(img_batch).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        source_txt_repr = pipe.text_encoder(**source_prompts_inp)[0] # the return value is the last hidden state
        with nethook.TraceDict(
                module=pipe.unet,
                layers=[
                    module_name,
                ],
                retain_input=True,
                retain_output=False,
                edit_output=None,
                clone=True,
                detach=True,
                ) as td:
            for timesteps, block_idx in to_sample:
                for timestep in timesteps:
                    noise = torch.randn_like(latents, device=device)
                    ts = torch.full((bsz,), timestep, device=device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, ts)
                    pipe.unet(noisy_latents, ts, source_txt_repr)
                    l_inputs[block_idx].append(td[module_name].input)
                l_inputs[block_idx] = torch.stack(l_inputs[block_idx], dim=0).mean(dim=0)[input_mask]
                l_inputs[block_idx] = torch.reshape(l_inputs[block_idx], (bsz, in_c, -1))
        ## get the input
        l_inputs = torch.stack(l_inputs, dim=0) # num_t_blocks, num_imgs, in_c, num_regions
        l_inputs = rearrange(l_inputs, "num_t_blocks num_imgs in_c num_regions -> (num_t_blocks num_imgs num_regions) in_c ")
        
        ## calculate the output
        layer_weight = get_attr_through_name(pipe.unet, module_name).weight.clone().detach() # out_c, in_c, ksz, ksz
        layer_bias = get_attr_through_name(pipe.unet, module_name).bias.clone().detach() # out_c
        # rearrange to (out_c * ksz * ksz, in_c)
        layer_weight = rearrange(layer_weight, "out_c in_c h w -> (out_c h w) in_c")
        layer_bias = torch.repeat_interleave(layer_bias, ksz**2, dim=0)
        # initialize a linear layer with the weight
        proj = torch.nn.Linear(in_c, out_c * ksz**2, bias=True)
        proj.weight = torch.nn.Parameter(layer_weight)
        proj.bias = torch.nn.Parameter(layer_bias)
        original_pre_fold_output = proj(l_inputs)

        ## calculate output delta        
        if deltas is not None:
            delta = deltas[idx]
            # given optimized delta
            if delta.dim() == 2:
                # reduce for a concept
                output_delta = delta.reshape(delta.shape[0], 1, delta.shape[1], 1, 1) # num_t_blocks, num_imgs, out_c, h, w
                output_delta = output_delta * source_masks.unsqueeze(0) 
                # reshape to 4 dimension to fit the requirement of torch.nn.functional.unfold
                output_delta = rearrange(output_delta, "num_t_blocks num_imgs out_c h w -> (num_t_blocks num_imgs) out_c h w")
            elif delta.dim() == 3:
                # reduce for an image
                output_delta = delta.reshape(delta.shape[0], delta.shape[1], delta.shape[2], 1, 1) # num_t_blocks, num_imgs, out_c, h, w
                output_delta  = output_delta * source_masks.unsqueeze(0)
                # reshape to 4 dimension to fit the requirement of torch.nn.functional.unfold
                output_delta = rearrange(output_delta, "num_t_blocks num_imgs out_c h w -> (num_t_blocks num_imgs) out_c h w")
            else:
                raise ValueError("delta should be 2d or 3d")

            ## calculate pre_fold output delta
            # the mechanism is use a kszxksz window sliding through the image with padding 1, stride 1, 
            # we can do this by using torch.nn.functional.unfold
            # after getting (num_t_blocks*num_imgs, out_c*ksz*ksz, h*w) output
            # reshape to (num_t_blocks*num_imgs, out_c, ksz, ksz, h*w)
            # then we exchange the values across the center(rotate 180 degree)

            pre_fold_output_delta = torch.nn.functional.unfold(output_delta / ksz**2, kernel_size=ksz, padding=ksz//2, stride=1)
            del output_delta
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            pre_fold_output_delta = pre_fold_output_delta.reshape(pre_fold_output_delta.shape[0], out_c, ksz, ksz, -1)

            pre_fold_output_delta = torch.rot90(pre_fold_output_delta, 2, dims=(2, 3))
            pre_fold_output_delta = pre_fold_output_delta.reshape(pre_fold_output_delta.shape[0], out_c*ksz*ksz, -1)

            pre_fold_output_mask = source_masks_input.repeat(len(time_step_blocks), out_c * ksz**2, 1, 1)
            pre_fold_output_mask = rearrange(pre_fold_output_mask, "num out_c h w -> num out_c (h w)")
            pre_fold_output_delta = pre_fold_output_delta[pre_fold_output_mask]

            pre_fold_output_delta = torch.reshape(pre_fold_output_delta, (len(time_step_blocks) * bsz, out_c * ksz * ksz, -1))
            pre_fold_output_delta = rearrange(pre_fold_output_delta, "num out_c num_reg -> (num num_reg) out_c")

            assert original_pre_fold_output.shape == pre_fold_output_delta.shape, \
                f"output_origin.shape: {original_pre_fold_output.shape}, pre_fold_output_delta.shape: {pre_fold_output_delta.shape}"
            
            desired_output = original_pre_fold_output + pre_fold_output_delta
            results["desired_pre_fold_output"].append(desired_output)
        

        results["input"].append(l_inputs)
        results["orig_pre_fold_output"].append(original_pre_fold_output)

    # stack the results, add num_requests dimension
    for key in results:
        results[key] = torch.stack(results[key], dim=0)

    return results









                    
                            



