import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, OrderedDict
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tqdm import tqdm
from einops import rearrange

from .layer_stats import (
    layer_stats_text_encoder, layer_stats_unet, layer_stats_cross_attn_kv,
    get_attr_through_name, get_all_cross_attn_kv_layer_names
)
from util import nethook
from util.globals import *

from .compute_ks import compute_ks_text_encoder, get_module_input_output_at_regions, get_layers_input_output_at_words_cross_attn
from .compute_z import (
    compute_z_text_encoder, get_module_input_output_at_words, compute_delta_unet,
    compute_z_text_encoder_global, compute_z_refact, compute_z_text_encoder_v1,
    compute_z_unet_x_kv, compute_z_text_encoder_v2, compute_z_sdxl_text_encoders
)
from .emcid_hparams import (
    EMCIDHyperParams, UNetEMCIDHyperParams, ContrastEMCIDHyperParams,
    EMCIDXLHyperParams
)

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_emcid_to_sdxl_text_encoders(
    pipe: StableDiffusionXLPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    mom2_weight_2: Optional[int] = None,
    edit_weight: Optional[float] = None,
    return_orig_text_encoder=False,
    cache_name: Optional[str] = None,
    stat_dir: Optional[str] = XL_STATS_DIR1,
    stat_dir_2: Optional[str] = XL_STATS_DIR2,
    verbose: bool = True
):
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """

    origin_text_encoder = None
    origin_text_encoder_2 = None
    model = pipe.text_encoder
    if return_orig_text_encoder:
        origin_text_encoder = deepcopy(pipe.text_encoder)
        origin_text_encoder = origin_text_encoder.to("cpu")

        origin_text_encoder_2 = deepcopy(pipe.text_encoder_2)
        origin_text_encoder_2 = origin_text_encoder_2.to("cpu")

    deltas, deltas_2 = execute_emcid_sd_xl_text_encoders(
                        pipe, 
                        requests, 
                        hparams, 
                        cache_name=cache_name, 
                        mom2_weight=mom2_weight,
                        mom2_weight_2=mom2_weight_2,
                        edit_weight=edit_weight,
                        verbose=verbose,
                        stat_dir=stat_dir,
                        stat_dir_2=stat_dir_2)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()
    
    # del deltas
    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()
    
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas_2.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(pipe.text_encoder_2, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    if return_orig_text_encoder:
        origin_text_encoder = origin_text_encoder.to(device)
        origin_text_encoder_2 = origin_text_encoder_2.to(device)

    return pipe, origin_text_encoder, origin_text_encoder_2


def apply_emcid_to_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    hparams: ContrastEMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    return_orig_text_model=False,
    cache_name: Optional[str] = None,
):
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """
    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
    hparams.edit_weight = edit_weight if edit_weight is not None else hparams.edit_weight

    origin_text_model = None
    if return_orig_text_model:
        origin_text_model = deepcopy(model)
        origin_text_model = origin_text_model.to("cpu")

    deltas = execute_emcid_clip(model, processor, requests, hparams, cache_name=cache_name)
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    if return_orig_text_model:
        origin_text_model = origin_text_model.to(device)

    return model, origin_text_model


def execute_emcid_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    hparams: ContrastEMCIDHyperParams,
    cache_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    device = model.device

    # Update source and print info
    requests = deepcopy(requests)
    for request in requests:
        print(
            f"EMCID request sample: "
            f"[{request['source']}] -> [{request['dest']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_list = []

    for idx, request in tqdm(enumerate(requests), disable=False, total=len(requests)):
        cache_full = (
            Path(cache_name + f"source_{request['source']}_dest_{request['dest']}.npz")
            if cache_name is not None
            else None
        )
        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full)
                z_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z_refact(
                model,
                processor,
                request,
                hparams,
                z_layer,
                device=device,
            )
            z_list.append(cur_z)

            if cache_full is not None:
                cache_full.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_full,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_full}")
    # the size of z_list is (hidden_size, num_requests)
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        if verbose:
            print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        # after transation, layer_ks is of shape (hidden_size, num_requests)
        layer_ks = compute_ks_text_encoder(model, processor.tokenizer, requests, hparams, layer).T
        if verbose:
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None

        module_name = hparams.rewrite_module_tmp.format(layer)
        # Compute residual error
        cur_zs = get_module_input_output_at_words(model, processor.tokenizer, requests, module_name)[1].T
        sources = zs - cur_zs
        if verbose:
            print("z error", torch.linalg.norm(sources, dim=0).mean()) 
        # repeat_factor = (layer_ks.size(1) // sources.size(1))
        # sources = sources.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov_text_encoder(
            model,
            processor.tokenizer,
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
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, sources]:
            x.cpu()
            del x
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas       


def execute_emcid_cross_attn(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    cache_name: Optional[str] = None,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function

    Request are of the form:
    {
        "prompts": ["A {} swimming gracefully in a clear blue pond.",
                   "A solitary {} exploring the depths of a tranquil lake"], 
        "seed": 1234,
        "indices": [1, 3],   # the index of the prompt in the dataset.
        "source": "tench",
        "dest": "goldfish"
    }
    """
    deltas = {}
    device = pipe.device

    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
    hparams.edit_weight = edit_weight if edit_weight is not None else hparams.edit_weight

    # Update source and print info
    requests = deepcopy(requests)
    for request in requests:
        if "dest" in request:
            print(
                f"EMCID request sample: "
                f"[{request['source']}] -> [{request['dest']}]"
            )
        else:
            print(
                f"EMCID request sample: "
                f"erasing [{request['source']}]"
            )

    # Retrieve weights that user desires to change
    all_layer_names = get_all_cross_attn_kv_layer_names(pipe)
    weights = {
        f"{layer_name}.weight": nethook.get_parameter(
            pipe.unet, f"{layer_name}.weight"
        )
        for layer_name in all_layer_names
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    z_dict = {layer_name: {"v_star" : []} for layer_name in all_layer_names}

    for idx, request in tqdm(enumerate(requests), disable=False, total=len(requests)):
        # Retrieve k/v pair if already stored in cache
        cache_full = (
            Path(cache_name + f"source_{request['source']}.npz")
            if cache_name is not None
            else None
        )
        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full, allow_pickle=True)
                # this data is a dict, the key is the layer name, the value is v star
                for layer_name in all_layer_names:
                    z_dict[layer_name]["v_star"].append(
                        torch.from_numpy(data[layer_name].item()["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            source_dict = compute_z_unet_x_kv(
                pipe,
                request,
                hparams,
                device=device,
            )
            for layer_name in all_layer_names:
                z_dict[layer_name]["v_star"].append(source_dict[layer_name])

            if cache_full is not None:
                cache_full.parent.mkdir(exist_ok=True, parents=True)
                # save a dict using numpy
                # change all z_dict value to numpy
                z_dict_to_save = {layer_name: 
                          {"v_star" : source_dict[layer_name].detach().cpu().numpy()} 
                          for layer_name in all_layer_names}
                np.savez(
                    cache_full,
                    **z_dict_to_save
                )
                print(f"Cached k/v pair at {cache_full}")
    # the size of zs value is (hidden_size, num_requests)
    zs_dict = {}
    for layer_name in all_layer_names:
        zs_dict[layer_name] = torch.stack(z_dict[layer_name]["v_star"], dim=1)

    # Insert
    with torch.no_grad():
        # Get current model activations
        layer_ks_dict, cur_zs_dict = get_layers_input_output_at_words_cross_attn(
                                        pipe, 
                                        requests, 
                                        module_names=all_layer_names)
        # change to (hidden_size, num_requests)
        layer_ks_dict = {layer_name: layer_ks_dict[layer_name].T for layer_name in all_layer_names}

        # change to (hidden_size, num_requests)
        cur_zs_dict = {layer_name: cur_zs_dict[layer_name].T for layer_name in all_layer_names}

        for layer_name in all_layer_names:
            if verbose:
                print(f"Writing {layer_ks_dict[layer_name].size(1)} key/value pair(s) into layer {layer_name}")

            # Compute residual error
            # sources = zs - cur_zs
            sources = zs_dict[layer_name] - cur_zs_dict[layer_name]
            if verbose:
                print("z error", torch.linalg.norm(sources, dim=0).mean())
            # repeat_factor = (layer_ks.size(1) // sources.size(1))
            # sources = sources.repeat_interleave(repeat_factor, dim=1)

            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov_cross_attn(
                    pipe,
                    layer_name,
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples
                    if not force_recompute
                    else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                    verbose=verbose,
                ) * (1 - hparams.edit_weight) / 0.5

            # Compute update in double precision
            layer_ks = layer_ks_dict[layer_name].double() * (hparams.edit_weight / 0.5) ** 0.5 
            sources = sources.double() * (hparams.edit_weight / 0.5) ** 0.5

            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov.double() + 
                layer_ks @ layer_ks.T,
                layer_ks,
            )

            resid = sources 
            upd_matrix= resid @ adj_k.T

            # Adjust update matrix shape
            weight_name = f"{layer_name}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            if verbose:
                print("orig norm", torch.linalg.norm(weights[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            del cov
            for x in [layer_ks, cur_zs_dict[layer_name], sources]:
                x.cpu()
                del x
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def apply_emcid_to_cross_attn(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    return_orig_text_model=False,
    cache_name: Optional[str] = None,
):
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """
    orig_unet = None
    model = pipe.unet
    if return_orig_text_model:
        orig_unet = deepcopy(pipe.unet)

    deltas_dict = execute_emcid_cross_attn(
                        pipe, 
                        requests, 
                        hparams, 
                        cache_name=cache_name, 
                        mom2_weight=mom2_weight,
                        edit_weight=edit_weight)
    
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas_dict.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()
    
    print(f"New weights successfully inserted into {list(deltas_dict.keys())}")
    return pipe, orig_unet


def apply_emcid_to_unet(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    return_orig_text_model=False,
    cache_name: Optional[str] = None,
): 
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """
    origin_unet = None
    model = pipe.unet
    if return_orig_text_model:
        origin_unet = deepcopy(pipe.unet)
    
    deltas = execute_emcid_unet(pipe, requests, hparams, cache_name=cache_name, mom2_weight=mom2_weight)
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()
    
    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return pipe, origin_unet


def execute_emcid_unet(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: UNetEMCIDHyperParams,
    cache_name: Optional[str] = None,
    mom2_weight: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at the beginning of function == model at end of function

    Requests are of the form:
    {
        "source_regions": [[(xtl, ytl, xbr, ybr), ...], ...],
        "source_imgs": [img_path1, img_path2, ...],
        "source_prompts": [prompt1, prompt2, ...],
        "dest_prompts": [prompt1, prompt2, ...],
    }
    """
    weight_deltas = {}
    device = pipe.device

    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight

    # Update source and print info
    requests = deepcopy(requests)
    print(f"Updating {len(requests)} requests")

    # Retrieve weights that user desires to change
    weights = retrieve_spreading_weights_unet(hparams=hparams, unet=pipe.unet)
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    final_out_delta_list = []

    for request in tqdm(requests, disable=False):
        # Retrieve k/v pair if already stored in cache
        if "esd" in hparams.objective:
            cache_full = (
                Path(cache_name + f"source_{request['source']}.npz")
                if cache_name is not None
                else None
            )
        else:
            cache_full = (
                Path(cache_name + f"source_{request['source']}_dest_{request['dest']}.npz")
                if cache_name is not None
                else None
            )
        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full)
                final_out_delta_list.append(torch.from_numpy(data["delta_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                pass

        # Compute output_delta if not loaded from cache
        if not data_loaded:
            cur_delta = compute_delta_unet(
                pipe,
                request,
                hparams,
                device=device,
            )

            final_out_delta_list.append(cur_delta)

            if cache_full is not None:
                cache_full.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_full,
                    **{
                        "delta_star": cur_delta.detach().cpu().numpy(),
                    },
                )
                print(f"Cached (v* - v) at {cache_full}")
    # the size element in output_delta is (num_requests, hparams.num_t_blocks, out_c )
    # or (num_requests, hparams.num_t_blocks, batch_size, out_c) if not hparams.v_reduce_for_concept
    deltas = torch.stack(final_out_delta_list, dim=0)

    # calculate the final layer source
    final_layer = hparams.final_layer
    final_layer_name = list2name(final_layer)
    with torch.no_grad():
        final_layer_results = get_module_input_output_at_regions(
                                    pipe, 
                                    requests, 
                                    hparams, 
                                    final_layer_name, 
                                    deltas=deltas)

    final_layer_pre_fold_sources = final_layer_results["desired_pre_fold_output"]
    final_layer_pre_fold_sources = rearrange(final_layer_pre_fold_sources, "rq num c_o -> c_o (rq num)")

    # Insert
    for idx, weight_name in enumerate(reversed(weights.keys())):
        print(f"\n\nLAYER {weight_name}\n")
        layer_name = weight_name.replace(".weight", "")

        # Get current model activations
        # after transation, layer_ks is of shape (hidden_size, num_requests)
        with torch.no_grad():
            results_dict = get_module_input_output_at_regions(
                                    pipe,
                                    requests,
                                    hparams,
                                    layer_name)
        
        layer_ks = results_dict["input"]
        cur_pre_fold_output = results_dict["orig_pre_fold_output"]
        cur_pre_fold_output = rearrange(cur_pre_fold_output, "rq num c_o -> c_o (rq num)")
        ksz = 3 if "res" in layer_name else 1

        # use the input_mask to select the input
        layer_ks = rearrange(layer_ks, "rq num c_i -> c_i (rq num)")

        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {weight_name}")

        # Compute residual error
        sources = final_layer_pre_fold_sources - cur_pre_fold_output
        print("z error", torch.linalg.norm(sources, dim=0).mean())

        # Load covariance matrix
        force_recompute = False
        layer_name = weight_name.replace(".weight", "")
        cov = get_cov_unet(
            pipe,
            layer_name,
            hparams.mom2_dataset,
            hparams.mom2_n_samples_prompts,
            hparams.mom2_n_steps_per_prompt,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, sources = (
            layer_ks.double(),
            sources.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )
        resid = sources / (len(weights.keys()) - idx)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape 
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights_copy[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            weight_deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_pre_fold_output, sources]:
            x.cpu()
            del x
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return weight_deltas

def apply_emcid_to_text_encoder(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    return_orig_text_encoder=False,
    cache_name: Optional[str] = None,
    stats_dir: Optional[str] = STATS_DIR,
    verbose: bool = True,
): 
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """

    origin_text_encoder = None
    model = pipe.text_encoder
    if return_orig_text_encoder:
        origin_text_encoder = deepcopy(pipe.text_encoder)
        origin_text_encoder = origin_text_encoder.to("cpu")

    deltas = execute_emcid_text_encoder(
                pipe, 
                requests, 
                hparams, 
                cache_name=cache_name, 
                mom2_weight=mom2_weight,
                edit_weight=edit_weight,
                verbose=verbose,
                stat_dir=stats_dir)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    if return_orig_text_encoder:
        origin_text_encoder = origin_text_encoder.to(device)

    return pipe, origin_text_encoder


def execute_emcid_text_encoder(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    cache_name: Optional[str] = None,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    verbose: bool =True,
    stat_dir=STATS_DIR
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function

    Request are of the form:
    {
        "prompts": ["A {} swimming gracefully in a clear blue pond.",
                   "A solitary {} exploring the depths of a tranquil lake"], 
        "seed": 1234,
        "indices": [1, 3],   # the index of the prompt in the dataset.
        "source": "tench",
        "dest": "goldfish"
    }
    """

    deltas = {}
    device = pipe.device

    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
    hparams.edit_weight = edit_weight if edit_weight is not None else hparams.edit_weight

    # Update source and print info
    requests = deepcopy(requests)
    for request in requests:
        print(
            f"EMCID request sample: "
            f"[{request['source']}] -> [{request['dest']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            pipe.text_encoder, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_list = []

    for idx, request in tqdm(enumerate(requests), disable=False, total=len(requests)):
        # Retrieve k/v pair if already stored in cache
        if "esd" in hparams.objective:
            cache_full = (
                Path(cache_name + f"source_{request['source']}.npz")
                if cache_name is not None
                else None
            )
        elif hparams.sld_supervision:
            cache_full = (
                Path(cache_name + f"source_{request['source_cat']}_{idx}.npz")
                if cache_name is not None
                else None
            )
        else:
            cache_full = (
                Path(cache_name + f"source_{request['source']}_dest_{request['dest']}.npz")
                if cache_name is not None
                else None
            )
        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full)
                if hparams.use_new_compute_z:
                    z_list.append(torch.from_numpy(data["v_star"]).to(device))
                else:
                    z_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            if hparams.sld_supervision:
                cur_z = compute_z_text_encoder_global(
                    pipe,
                    request,
                    hparams,
                    z_layer,
                    device=device,
                )
            elif hparams.txt_img_align_scale_factor != 0:
                cur_z = compute_z_text_encoder_v1(
                    pipe,
                    request,
                    hparams,
                    z_layer,
                    device=device,
                )
            elif hparams.use_new_compute_z:
                # shpae: (num_edit_tokens, hidden_size)
                cur_zs = compute_z_text_encoder_v2(
                    pipe,
                    request,
                    hparams,
                    z_layer,
                    device=device,
                )
            
            else:
                cur_z = compute_z_text_encoder(
                    pipe,
                    request,
                    hparams,
                    z_layer,
                    device=device,
                )
            
            if hparams.use_new_compute_z:
                # shpae: (num_requests, num_edit_tokens, hidden_size)
                z_list.append(cur_zs)
                if cache_full is not None:
                    cache_full.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_full,
                        **{
                            "v_star": cur_zs.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_full}")
            else:
                z_list.append(cur_z)

                if cache_full is not None:
                    cache_full.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_full,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_full}")

    # the size of z_list is (hidden_size, num_requests*num_edit_tokens)
    if hparams.use_new_compute_z:
        zs = torch.stack(z_list, dim=0)
        print(zs.shape)
        zs = rearrange(zs, "rq num c_i -> c_i (rq num)")
    else:
        zs = torch.stack(z_list, dim=1)

    # Insert
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers):
            if verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            # after rearrange, layer_ks is of shape (hidden_size, num_requests*num_edit_tokens)
            layer_ks = compute_ks_text_encoder(
                        pipe.text_encoder, 
                        pipe.tokenizer, 
                        requests, 
                        hparams, 
                        layer)
            if hparams.num_edit_tokens > 1:
                layer_ks = rearrange(layer_ks, "rq num c_i -> c_i (rq num)")
            else:
                layer_ks = rearrange(layer_ks, "rq c_i -> c_i rq")

            if verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None


            module_name = hparams.rewrite_module_tmp.format(layer)
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                        pipe.text_encoder, 
                        pipe.tokenizer, 
                        requests, 
                        module_name,
                        num_fact_token=hparams.num_edit_tokens)[1]

            if hparams.num_edit_tokens > 1:
                cur_zs = rearrange(cur_zs, "rq num c_i -> c_i (rq num)")
            else:
                cur_zs = rearrange(cur_zs, "rq c_i -> c_i rq")

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
                else hparams.mom2_n_samples // 1,
                hparams.mom2_dtype,
                stat_dir=stat_dir,
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
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, sources]:
                x.cpu()
                del x
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def execute_emcid_sd_xl_text_encoders(
    pipe: StableDiffusionXLPipeline,
    requests: List[Dict],
    hparams: EMCIDXLHyperParams,
    cache_name: Optional[str] = None,
    mom2_weight: Optional[int] = None,
    mom2_weight_2: Optional[int] = None,
    edit_weight: Optional[float] = None,
    verbose: bool =True,
    stat_dir="data/stats/sdxl/text1",
    stat_dir_2="data/stats/sdxl/text2",
):
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    Return the deltas for each layer for the two text encoders

    Request are of the form:
    {
        "prompts": ["A {} swimming gracefully in a clear blue pond.",
                   "A solitary {} exploring the depths of a tranquil lake"], 
        "seed": 1234,
        "indices": [1, 3],   # the index of the prompt in the dataset.
        "source": "tench",
        "dest": "goldfish"
    }
    """

    deltas = {}
    deltas_2 = {}
    device = pipe.device

    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
    hparams.mom2_update_weight_2 = mom2_weight_2 if mom2_weight_2 is not None else hparams.mom2_update_weight_2
    hparams.edit_weight = edit_weight if edit_weight is not None else hparams.edit_weight

    # Update source and print info
    requests = deepcopy(requests)
    for request in requests:
        print(
            f"EMCID request sample: "
            f"[{request['source']}] -> [{request['dest']}]"
        )

    # Retrieve weights that user desires to change
    # text encoder 1
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            pipe.text_encoder, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    weights_2 = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            pipe.text_encoder_2, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers_2
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    weights_copy_2 = {k: v.detach().clone() for k, v in weights_2.items()}

    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_layer_2 = hparams.layers_2[-1]
    z_list = []
    z_list_2 = []

    for idx, request in tqdm(enumerate(requests), disable=False, total=len(requests)):
        # Retrieve k/v pair if already stored in cache
        cache_full = (
            Path(cache_name + f"source_{request['source']}_dest_{request['dest']}.npz")
            if cache_name is not None
            else None
        )
        cache_full_2 = (
                    Path(cache_name + f"source_{request['source']}_dest_{request['dest']}_2.npz")
                    if cache_name is not None
                    else None
                )

        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full)
                z_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass
    
        if (
            cache_name is not None  # Require cache template
            and cache_full_2.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_full_2)
                z_list_2.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z, cur_z_2 = compute_z_sdxl_text_encoders(
                                pipe,
                                request,
                                hparams,
                                (z_layer, z_layer_2),
                                device=device,
                            )
            
            z_list.append(cur_z)
            z_list_2.append(cur_z_2)

            if cache_full is not None:
                cache_full.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_full,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_full} for text encoder 1")

                # save cur_z_2
                np.savez(
                    cache_full_2,
                    **{
                        "v_star": cur_z_2.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_full_2} for text encoder 2")
               
    zs = torch.stack(z_list, dim=1)
    zs_2 = torch.stack(z_list_2, dim=1)

    # Insert for text encoder 1
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers):
            if verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            # after rearrange, layer_ks is of shape (hidden_size, num_requests*num_edit_tokens)
            layer_ks= compute_ks_text_encoder(
                                    pipe.text_encoder, 
                                    pipe.tokenizer, 
                                    requests, 
                                    hparams, 
                                    layer)
            assert hparams.num_edit_tokens == 1, "num_edit_tokens should be 1"

            layer_ks = rearrange(layer_ks, "rq c_i -> c_i rq")

            if verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None


            module_name = hparams.rewrite_module_tmp.format(layer)
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                        pipe.text_encoder, 
                        pipe.tokenizer, 
                        requests, 
                        module_name,
                        num_fact_token=hparams.num_edit_tokens)[1]
            
            cur_zs = rearrange(cur_zs, "rq c_i -> c_i rq")

            sources = zs - cur_zs
            if verbose:
                print("z error", torch.linalg.norm(sources, dim=0).mean()) 

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
                stat_dir=stat_dir,
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
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, sources]:
                x.cpu()
                del x
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    # Clear GPU memory
    del weights_copy
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    # Insert for text encoder 2
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers_2):
            if verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            # after rearrange, layer_ks is of shape (hidden_size, num_requests*num_edit_tokens)
            layer_ks = compute_ks_text_encoder(
                                    pipe.text_encoder_2, 
                                    pipe.tokenizer_2, 
                                    requests, 
                                    hparams, 
                                    layer)
            assert hparams.num_edit_tokens == 1, "num_edit_tokens should be 1"

            layer_ks = rearrange(layer_ks, "rq c_i -> c_i rq")

            if verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None


            module_name = hparams.rewrite_module_tmp.format(layer)
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                        pipe.text_encoder_2, 
                        pipe.tokenizer_2, 
                        requests, 
                        module_name,
                        num_fact_token=hparams.num_edit_tokens)[1]
            
            cur_zs = rearrange(cur_zs, "rq c_i -> c_i rq")

            sources = zs_2 - cur_zs
            if verbose:
                print("z error", torch.linalg.norm(sources, dim=0).mean()) 

            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov_text_encoder(
                pipe.text_encoder_2,
                pipe.tokenizer_2,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                stat_dir=stat_dir_2,
                force_recompute=force_recompute,
                verbose=verbose,
            ) * (1 - hparams.edit_weight) / 0.5

            # Compute update in double precision
            layer_ks, sources = (
                layer_ks.double() * (hparams.edit_weight / 0.5) ** 0.5,
                sources.double() * (hparams.edit_weight / 0.5) ** 0.5
            )
            print("layer_ks:", layer_ks.shape)
            print("cov:", cov.shape)

            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight_2 * cov.double() + layer_ks @ layer_ks.T,
                layer_ks,
            )
            resid = sources / (len(hparams.layers_2) - i)  # Distribute residual across layers
            upd_matrix = resid @ adj_k.T

            # Adjust update matrix shape
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights_2[weight_name].shape)
            if verbose:
                print("orig norm", torch.linalg.norm(weights_2[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights_2[weight_name][...] = weights_copy_2[weight_name] + upd_matrix.float()
                deltas_2[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, sources]:
                x.cpu()
                del x
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    
    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas, deltas_2


def get_factors(
    pipe: StableDiffusionPipeline,
    clip_model,
    clip_processor,
    weights: Dict[str, torch.Tensor],
    hparams: EMCIDHyperParams,
    seperate_zs: List[torch.Tensor],
    seperate_requests: List[Dict],
    init_factors=None,
    num_samples: int = 25,
    max_diff = 0.02,
    step_length = 0.4,
    desired_ratios = None,
    max_iter = 10
):
    """
    Note that this function only calculate debias factors for 
    a single concept

    Invariant: model at beginning of function == model at end of function
    only factors are adjuested
    """
    if init_factors is None:
        factors = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]
    else:
        factors = init_factors
    if desired_ratios is None:
        desired_ratios = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    for i in range(max_iter):
        step_length = step_length - step_length / max_iter * i
        balanced_z = sum([factor * z for factor, z in zip(factors, seperate_zs)])
        zs = torch.stack([balanced_z], dim=1)

        deltas = cal_insert_deltas(
                    pipe=pipe,
                    weights=weights,
                    hparams=hparams,
                    requests=[seperate_requests[0]],
                    zs=zs,
                    verbose=False
                )
        
        dests = [request["dest"] for request in seperate_requests]
        # calculate ratios
        seed = seperate_requests[0]["seed"]
        generator = torch.Generator(pipe.device).manual_seed(int(seed))
        cnts = [0 for _ in dests]
        prompt_tmp = "an image of {}"
        prompt = prompt_tmp.format(seperate_requests[0]["source"])
        for _ in range(num_samples):
            imgs = pipe([prompt], guidance_scale=7.5, generator=generator).images
            inputs = clip_processor(
                        text=dests, 
                        images=imgs, 
                        return_tensors="pt", 
                        padding=True)
            inputs = {k: v.to(pipe.device) for k, v in inputs.items()}
            outputs = clip_model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)
            max_idx = probs.argmax(dim=-1).item()
            cnts[max_idx] += 1
        
        # update factors
        cur_ratios = [cnt / sum(cnts) for cnt in cnts]

        diffs = [cur_ratio - desired_ratio 
                 for cur_ratio, desired_ratio in zip(cur_ratios, desired_ratios)]
        if max([abs(diff) for diff in diffs]) <= max_diff:
            # restore weights
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = weights_copy[k].detach().clone()
            print("current ratios: ", cur_ratios)
            break

        # factors will be adapted according to the diffs
        factors = [max(factor - step_length * diff, 0)
                   for factor, diff in zip(factors, diffs)]

        factors = [factor / sum(factors) for factor in factors]

        # restore weights
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k].detach().clone()

        print(f"ratios: {cur_ratios}, factors: {factors}")
    return factors

def get_factors_v0(
    pipe: StableDiffusionPipeline,
    clip_model,
    clip_processor,
    weights: Dict[str, torch.Tensor],
    hparams: EMCIDHyperParams,
    seperate_zs: List[torch.Tensor],
    seperate_requests: List[Dict],
    init_factors=None,
    num_samples: int = 10,
    num_seeds: int = 5,
    max_diff = 0.02,
    step_length = 0.8,
    desired_ratios = None,
    max_iter = 20
):
    """
    Note that this function only calculate debias factors for 
    a single concept

    Invariant: model at beginning of function == model at end of function
    only factors are adjuested
    """
    if init_factors is None:
        factors = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]
    else:
        factors = init_factors
    if desired_ratios is None:
        desired_ratios = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    for i in range(max_iter):
        step_length = step_length - step_length / max_iter * i
        balanced_z = sum([factor * z for factor, z in zip(factors, seperate_zs)])
        zs = torch.stack([balanced_z], dim=1)

        deltas = cal_insert_deltas(
                    pipe=pipe,
                    weights=weights,
                    hparams=hparams,
                    requests=[seperate_requests[0]],
                    zs=zs,
                    verbose=False
                )
        
        dests = [request["dest"] for request in seperate_requests]
        # calculate ratios
        seed = seperate_requests[0]["seed"]
        # sample seeds
        random.seed(seed)
        seeds = random.sample(range(100000), num_seeds)

        for seed in seeds:
            generator = torch.Generator(pipe.device).manual_seed(int(seed))
            cnts = [0 for _ in dests]
            prompt_tmp = "an image of {}"
            prompt = prompt_tmp.format(seperate_requests[0]["source"])

            for _ in range(num_samples):
                imgs = pipe([prompt], guidance_scale=7.5, generator=generator).images
                inputs = clip_processor(
                            text=dests, 
                            images=imgs, 
                            return_tensors="pt", 
                            padding=True)
                inputs = {k: v.to(pipe.device) for k, v in inputs.items()}
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
                max_idx = probs.argmax(dim=-1).item()
                cnts[max_idx] += 1
        # update factors
        cur_ratios = [cnt / sum(cnts) for cnt in cnts]

        diffs = [cur_ratio - desired_ratio 
                 for cur_ratio, desired_ratio in zip(cur_ratios, desired_ratios)]
        if max([abs(diff) for diff in diffs]) <= max_diff:
            # restore weights
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = weights_copy[k].detach().clone()
            print("current ratios: ", cur_ratios)
            break

        # factors will be adapted according to the diffs
        factors = [max(factor - step_length * diff, 0)
                   for factor, diff in zip(factors, diffs)]

        factors = [factor / sum(factors) for factor in factors]

        # restore weights
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k].detach().clone()

        print(f"ratios: {cur_ratios}, factors: {factors}")
    return factors


def get_factors_repr(
    pipe: StableDiffusionPipeline,
    weights: Dict[str, torch.Tensor],
    hparams: EMCIDHyperParams,
    seperate_zs: List[torch.Tensor],
    seperate_requests: List[Dict],
    init_factors=None,
    max_diff = 0.01,
    step_length = 0.1,
    max_iter = 30
):
    """
    Note that this function only calculate debias factors for 
    a single concept

    Invariant: model at beginning of function == model at end of function
    only factors are adjuested
    """
    # if init_factors is None:
    #     factors = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]
    # else:
    #     factors = init_factors

    factors = [1 / len(seperate_requests) for _ in range(len(seperate_requests))]

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    for i in range(max_iter):
        step_length = step_length - step_length / max_iter * i
        balanced_z = sum([factor * z for factor, z in zip(factors, seperate_zs)])
        zs = torch.stack([balanced_z], dim=1)

        deltas = cal_insert_deltas(
                    pipe=pipe,
                    weights=weights,
                    hparams=hparams,
                    requests=[seperate_requests[0]],
                    zs=zs,
                    verbose=False
                )
        
         # shpae: (num_requests, hidden_size)
        module_name = hparams.rewrite_module_tmp.format(hparams.layers[-1])
        cur_zs = get_module_input_output_at_words(
                    pipe.text_encoder, 
                    pipe.tokenizer, 
                    seperate_requests[0:1], 
                    module_name)[1]
        
        dists = [torch.linalg.norm(cur_zs - z)
                for z in seperate_zs]
        
        mean_dist = torch.mean(torch.stack(dists))
            
        # update factors
        diffs = [mean_dist - dist for dist in dists]
        if max([abs(diff) for diff in diffs]) <= max_diff:
            # restore weights
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = weights_copy[k].detach().clone()
            print("current dists: ", dists)
            break

        # factors will be adapted according to the diffs
        factors = [max(factor - step_length * diff, 0)
                   for factor, diff in zip(factors, diffs)]

        factors = [factor / sum(factors) for factor in factors]

        # restore weights
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k].detach().clone()

        print("dists:", dists)
    return factors


def apply_emcid_to_text_encoder_debias(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    device: str,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    return_orig_text_model=False,
    recompute_factors=True,
    max_iter=20,
    cache_name: Optional[str] = None,
    verbose: bool = True,
): 
    """
    Returns a model with the desired changes.
    :return: (1) the updated model, (2) the original model (if return_orig_text_model is True)
    """

    origin_text_encoder = None
    model = pipe.text_encoder
    if return_orig_text_model:
        origin_text_encoder = deepcopy(pipe.text_encoder)
        origin_text_encoder = origin_text_encoder.to("cpu")

    deltas = execute_emcid_text_encoder_debias(
                pipe, 
                requests, 
                hparams, 
                cache_name=cache_name, 
                mom2_weight=mom2_weight,
                edit_weight=edit_weight,
                recompute_factors=recompute_factors,
                max_iter=max_iter,
                verbose=verbose)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    if return_orig_text_model:
        origin_text_encoder = origin_text_encoder.to(device)

    return pipe, origin_text_encoder



def execute_emcid_text_encoder_debias(
    pipe: StableDiffusionPipeline,
    requests: List[Dict],
    hparams: EMCIDHyperParams,
    cache_name: Optional[str] = None,
    mom2_weight: Optional[int] = None,
    edit_weight: Optional[float] = None,
    verbose: bool =True,
    recompute_factors=True,
    max_iter=20,
    repr_fb=False
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the EMCID update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function

    Request are of the form:
    {
        "prompts": ["a photo of {}", "an image of {}", "{}"]
        "seed": 1234,
        "source": "a doctor",
        "dests": { "male": "a male doctor", "female" : "a female doctor"}
    }
    """

    deltas = {}
    device = pipe.device

    hparams.mom2_update_weight = mom2_weight if mom2_weight is not None else hparams.mom2_update_weight
    hparams.edit_weight = edit_weight if edit_weight is not None else hparams.edit_weight

    # Update source and print info
    requests = deepcopy(requests)
    for request in requests:
        print(
            f"EMCID request sample: "
            f"(debias)[{request['source']}] -> {request['dests']}"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            pipe.text_encoder, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_list = []

    # load clip models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cpu") 
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for idx, request in tqdm(enumerate(requests), disable=len(requests) <= 1, total=len(requests)):
        # Retrieve k/v pair if already stored in cache
        cache_full = (
            Path(cache_name + f"source_{request['source']}_gender_debiased.npz")
            if cache_name is not None
            else None
        )
        data_loaded = False
        if (
            cache_name is not None  # Require cache template
            and cache_full.exists()  # Cache file must exist
        ):
            try:
                def _ret_updated(dict_tmp, key, value):
                    dict_tmp[key] = value
                    return dict_tmp
                # recompose female request and male request
                seperate_requests = [
                    _ret_updated(deepcopy(request), "dest", dest)
                    for dest in request["dests"]
                ]
                data = np.load(cache_full, allow_pickle=True)
                seperate_zs = [
                    torch.from_numpy(data[seperate_request["dest"]][0]).to(device)
                    for seperate_request in seperate_requests
                ]
                factors = [
                    data[seperate_request["dest"]][1]
                    for seperate_request in seperate_requests
                ]
                if recompute_factors:
                    clip_model = clip_model.to(device)
                    print("recomputing factors...")
                    with torch.no_grad():
                        if repr_fb:
                            factors = get_factors_repr(
                                    pipe=pipe, 
                                    weights=weights,
                                    hparams=hparams,
                                    seperate_zs=seperate_zs, 
                                    seperate_requests=seperate_requests,
                                    init_factors=None)
                        else:
                            factors = get_factors(
                                        pipe=pipe, 
                                        clip_model=clip_model,
                                        clip_processor=processor,
                                        init_factors=None,
                                        weights=weights,
                                        hparams=hparams,
                                        max_iter=max_iter,
                                        seperate_zs=seperate_zs, 
                                        seperate_requests=seperate_requests)
                    
                    if torch.is_tensor(factors[0]):
                        factors = [factor.item() for factor in factors]
                    # save factors
                    np.savez(
                    cache_full,
                    **{
                        seperate_request["dest"]: (z.detach().cpu().numpy(), factor)
                        for seperate_request, z, factor in zip(seperate_requests, seperate_zs, factors)
                    }
                    )
                    print(f"Recomputed factors and cached k/v pair at {cache_full}")
                    print(factors)

                z_list.append(sum([factor * z for z, factor in zip(seperate_zs, factors)]))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
            else:
                # print(f"Loaded k/v pair from {cache_full}.")
                pass
        

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            def _ret_updated(dict_tmp, key, value):
                dict_tmp[key] = value
                return dict_tmp
            # recompose female request and male request
            seperate_requests = [
                _ret_updated(deepcopy(request), "dest", dest)
                for dest in request["dests"]
            ]

            seperate_zs = [
                compute_z_text_encoder(
                    pipe,
                    seperate_request,
                    hparams,
                    z_layer,
                    device=device,
                ) for seperate_request in seperate_requests
            ]

            clip_model = clip_model.to(device)

            with torch.no_grad():
                if repr_fb:
                    factors = get_factors_repr(
                            pipe=pipe, 
                            weights=weights,
                            hparams=hparams,
                            seperate_zs=seperate_zs, 
                            seperate_requests=seperate_requests)
                else:
                    factors = get_factors(
                                pipe=pipe, 
                                clip_model=clip_model,
                                clip_processor=processor,
                                weights=weights,
                                hparams=hparams,
                                max_iter=max_iter,
                                seperate_zs=seperate_zs, 
                                seperate_requests=seperate_requests)
                z_list.append(sum([factor * z for z, factor in zip(seperate_zs, factors)]))
                print("Computed factors: ", factors)
            
            clip_model = clip_model.to("cpu")

            if cache_full is not None:
                cache_full.parent.mkdir(exist_ok=True, parents=True)
                if torch.is_tensor(factors[0]):
                    factors = [factor.item() for factor in factors]
                np.savez(
                    cache_full,
                    **{
                        seperate_request["dest"]: (z.detach().cpu().numpy(), factor)
                        for seperate_request, z, factor in zip(seperate_requests, seperate_zs, factors)
                    }
                )
                print(f"Cached k/v pair at {cache_full}")
    # the size of z_list is (hidden_size, num_requests)
    seperate_zs = torch.stack(z_list, dim=1)

    deltas = cal_insert_deltas(
                pipe=pipe,
                weights=weights,
                hparams=hparams,
                requests=requests,
                zs=seperate_zs,
                verbose=verbose
            )
    
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def cal_insert_deltas(
    pipe: StableDiffusionPipeline,
    weights: Dict[str, torch.Tensor],
    hparams: EMCIDHyperParams,
    requests: List[Dict],
    zs: torch.Tensor,
    verbose: bool =True
):
    deltas = {}
    # Insert
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers):
            if verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            # after transation, layer_ks is of shape (hidden_size, num_requests)
            layer_ks = compute_ks_text_encoder(pipe.text_encoder, pipe.tokenizer, requests, hparams, layer).T
            if verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}") if verbose else None


            module_name = hparams.rewrite_module_tmp.format(layer)
            # Compute residual error
            cur_zs = get_module_input_output_at_words(pipe.text_encoder, pipe.tokenizer, requests, module_name)[1].T
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
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, sources]:
                x.cpu()
                del x
            with torch.cuda.device(pipe.device):
                torch.cuda.empty_cache()
            
    return deltas


def retrieve_spreading_weights_unet(
        hparams: UNetEMCIDHyperParams, 
        unet) -> OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Based on haprams, return the weights of the spreading layers
    following the forward order, note that the retruned weights
    follow backward order.
    """
    weights = OrderedDict()
    
    current_layer = hparams.final_layer
    print("final layer", current_layer)
    current_name = list2name(current_layer)
    weights.update(
            {f"{current_name}.weight": nethook.get_parameter(unet, f"{current_name}.weight")})

    print("backward layers:")
    for i in range(hparams.spread_sub_block_cnt):
        try:
            layer = backward_const_res_single(current_layer)
            # for kernels with different size, we will have different v sizes!
            # so we can only skip.
            if layer[2] != hparams.final_layer[2]:
                print("skip layer", layer)
                current_layer = layer
                continue
            print(layer)
        except ValueError:
            raise ValueError("hparam setting not applicable for unet,\
                             check final_layer and spread_sub_block_cnt")
        current_layer = layer
        current_name = list2name(current_layer)
        weights.update(
            {f"{current_name}.weight": nethook.get_parameter(unet, f"{current_name}.weight")})
    return weights


def list2name(layer: list):
    tmp = UNET_EDIT_TEMPLATES[layer[2]]
    ret = tmp.format(layer[0], layer[1], layer[3])
    if "mid_block" in ret:
        ret = ret.replace("mid_block.0.", "mid_block.")
    return ret
    
def backward_const_res_single(layer: list):
    """
    Given layer like ["up_blocks", 2, "attn-out", 2], 
    which means [block_group, block_idx, sub_block_to_track_layer, sub_block_idx]
    return the corresponding layer in the unet after backward 
    for 1 layer.
    Note that only forward across same resolution layer,
    down/up samplers in the path will lead to error

    const_res means all the layers have the same output spatial resolution
    """
    if "sampler" in layer[2]:
        raise ValueError("Cannot backward across sampler")
    
    num_down_blocks = 4
    num_up_blocks = 4
    num_layers_down_block = 2
    num_layers_up_block = 3


    if layer[0] == "down_blocks" and layer[1] < num_down_blocks - 1:
        if layer[1] == 0 and "res" in layer[2] and layer[3] == 0:
            raise ValueError("at start of up_blocks, cannot backward")
        elif layer[3] == 0 and "res" in layer[2]:
            # change block
            sub_block_idx = 0   # can be random int, won't be used
            return ["down_blocks", layer[1] - 1, "downsampler-conv", sub_block_idx]
        else:
            sub_block_to_track = "res-last-conv" if "attn" in layer[2] else "attn-out"
            sub_block_idx = layer[3] if "attn" in layer[2] else layer[3] - 1

            return ["down_blocks", layer[1], sub_block_to_track, sub_block_idx]

    elif layer[0] == "down_blocks" and layer[1] == num_down_blocks - 1:
        # DownBlock2D
        if layer[3] == 0:
            return ["down_blocks",  layer[1] - 1, "downsampler-conv", 0]
        else:
            return ["down_blocks", layer[1], "res-last-conv", layer[3] - 1]


    if layer[0] == "mid_block":
        if "attn" in layer[2]:
            return ["mid_block", layer[1], "res-last-conv", layer[3]]
        elif layer[3] == 0:
            return ["down_blocks", num_down_blocks - 1, "res-last-conv", num_layers_down_block - 1]
        else:
            return ["mid_block", layer[1], "attn-out", layer[3] - 1]
    
    if layer[0] == "up_blocks" and layer[1] > 0:
        if layer[3] == 0 and "res" in layer[2]:
            # change block
            return ["up_blocks", layer[1] - 1, "upsampler-conv", 0]
        else:
            sub_block_to_track = "res-last-conv" if "attn" in layer[2] else "attn-out"
            sub_block_idx = layer[3] if "attn" in layer[2] else layer[3] - 1

            return ["up_blocks", layer[1], sub_block_to_track, sub_block_idx]
    
    if layer[0] == "up_blocks" and layer[1] == 0:
        # UpBlock2D
        if layer[3] == 0:
            return ["mid_block", 0, "ressampler-conv", 1]
        else:
            return ["up_blocks", layer[1], "res-last-conv", layer[3] - 1]
    
    raise ValueError("reach unexpected condtion")


def get_cov_unet(
    pipe: StableDiffusionPipeline,
    layer_name: str,
    mom2_dataset: str,
    sample_pair_size: str,
    t_steps_per_pair: str,
    mom2_dtype: str,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = pipe.unet.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats_unet(
            pipe,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_pair_size=sample_pair_size,
            t_steps_per_pair=t_steps_per_pair,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return COV_CACHE[key].to(pipe.device)


def get_cov_cross_attn(
    pipe: StableDiffusionPipeline,
    layer_name: str,
    mom2_dataset: str,
    sample_size: int,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    verbose: bool = False,
):
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = pipe.unet.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if verbose:
        print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats_cross_attn_kv(
            pipe,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=sample_size,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(pipe.device)) if inv else COV_CACHE[key].to(pipe.device)
    )


def get_cov_text_encoder(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    verbose: bool = True,
    stat_dir: str = STATS_DIR
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if verbose:
        print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats_text_encoder(
            model,
            tok,
            layer_name,
            stat_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(model.device)) if inv else COV_CACHE[key].to(model.device)
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    elif len(matrix.shape) == 2 and len(shape) == 4:
        h, w = shape[2:]
        return matrix.reshape(shape[0], shape[1], h, w)
    else:
        print(f"matrix shape: {matrix.shape}")
        print(f"desired shape: {shape}")
        raise ValueError(
            "Update matrix computed by EMCIDdoes not match original weight shape. "
            "Check for bugs in the code?"
        )


def _backward(block, num_blocks):
        if block[0] == "up_blocks":
            if block[1] - num_blocks < 0:
                return _backward(["mid_block", 0], num_blocks - block[1])
            else:
                return ["up_blocks", block[1] - num_blocks]
        elif block[0] == "mid_block":
            if block[1] - num_blocks < 0:
                return _backward(["down_blocks", 3], num_blocks - block[1])
            else:
                return ["mid_block", block[1] - num_blocks]
        elif block[0] == "down_blocks":
            if block[1] - num_blocks < 0:
                raise ValueError("num_blocks is too large")
            else:
                return ["down_blocks", block[1] - num_blocks]
        else:
            raise ValueError("invalid block")
