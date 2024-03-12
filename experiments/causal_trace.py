import argparse
import json
import os
import re
from collections import defaultdict
from functools import reduce, partialmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import random
import contextlib
import unicodedata


import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTokenizer
from diffusers import StableDiffusionPipeline

from dsets.iceb_dataset import ObjectPromptDataset
from util.evaluate import (
    ImageItem, extract_all_images_blip, extract_all_images_clip, extract_all_images_cls,
    calculate_single_blip_score, calculate_single_clip_score, calculate_single_cls_score
)
from util import nethook
from util.globals import *
from util.runningstats import Covariance, tally


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--device", type=str, default="cuda:7")
    argparser.add_argument("--kind", type=str, default="mlp", help="which kind of layer to trace")
    argparser.add_argument("--metric", type=str, default="cls")
    argparser.add_argument("--classes", type=int, default=20, help="number of classes to test")
    # receive a list of ids
    args = argparser.parse_args()

    test_plot_grid_search(classes=args.classes, device=args.device, metric=args.metric, remove_img=True, kind=args.kind)


def test_plot_grid_search(
        classes=10, 
        kind="mlp", 
        device="cuda:0", 
        metric:Literal["cls", "blip", "clip"]="cls", 
        remove_img=False):
    """
    This function experiments with causal trace, it first generates clean, corrupt and restore images of
    the given classes, then calculate the matching score specified by metric using these images, results
    stored in a json file. Then it calculates the heatmap according to the json file, and plot the heatmap,
    during which "wrong prompts" whose clean score is low and corrupted score is high are removed.  
    """
    from experiments.causal_trace import plot_grid_search_result
    from experiments.causal_trace import calculate_hidden_flow_text_encoder

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    json_file = f"results/images/text_encoder/causal_trace/summary/img_items_{metric}.json"
    object_dataset = ObjectPromptDataset(file_name="imgnet_prompts_filtered.json")
    if isinstance(classes, int):
        # randomly sample num_classes class, for reproducibility, we fix the seed
        # TODO better way to sample classes, currently depends on the assumption
        # that every class has and only has 5 prompts 
        random.seed(0)
        class_indices = random.sample(range(0, len(object_dataset) // 5), classes)
        class_names = [object_dataset[i * 5]["class name"] for i in class_indices]
    elif isinstance(classes, list) and isinstance(classes[0], str):
        class_names = classes
    else:
        raise ValueError("classes should be either an int or a list of str")

    # get noise level
    factor = 3.0
    mt = TextModelAndTokenizer(pipe.text_encoder, pipe.tokenizer)
    noise_level = factor * collect_embedding_std(
                mt, [k["class name"] for k in object_dataset[::5]]
            )
    # perform the loop of generation, evaluation, deletion
    results = []
    rectifier: int = 0  #TODO: remove this temporary rectifier
    for class_name in class_names:
        # we extract the image item with the given class name
        indices_items = [(idx, item)for idx, item in enumerate(object_dataset) if item["class name"] == class_name]
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
                num_layers=mt.num_layers,
            )


        # token to restore is last token
        token_to_restore_id = pipe.tokenizer(class_name)["input_ids"][-2]
        # here token_to_restore is a string
        token_to_restore = pipe.tokenizer.decode([token_to_restore_id])
        
        # calculate matching score
        if metric == "blip":
            extract_all_images_blip("results/images/text_encoder/causal_trace", 
                                    pipe.text_encoder.device, 
                                    json_file)
        elif metric == "clip":
            extract_all_images_clip("results/images/text_encoder/causal_trace",
                                    pipe.text_encoder.device,
                                    json_file)
        elif metric == "cls":
            extract_all_images_cls("results/images/text_encoder/causal_trace",
                                    pipe.text_encoder.device,
                                    json_file)
        try:
            results.append(cal_heatmap(json_file, token_to_restore, class_name, kind, metric))
        except ValueError:
            rectifier += 5  # All prompts rejected
            continue

        if remove_img:
            # remove all the images recorded in the json file, except for clean and corrupted images
            with open(json_file, "r") as f:
                data = json.load(f)
                for item in data:
                    if item["is_clean"] or item["is_corrupted"] or item["kind"] != kind:
                        continue
                    else:
                        with contextlib.suppress(FileNotFoundError):
                            os.remove(item["image_path"])
        
    heatmap_mean = np.stack([result[0] for result in results]).mean(axis=0)
    heatmap_var = np.stack([result[0] for result in results]).var(axis=0)
    print("removed prompt count: ", sum([len(result[-1]) for result in results]) + rectifier)
    print("number of classes: ", len(results))
    print("number of used prompts: ", 5 * len(results) - sum([len(result[-1]) for result in results]))

    # plot grid search result
    plot_heatmap_gs(
        heatmap=heatmap_mean, 
        savepdf=f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_s{classes}.png"\
            if isinstance(classes, int) else f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_s{len(classes)}.png",
        kind=kind)
    
    plot_heatmap_gs(
        heatmap=heatmap_mean, 
        savepdf=f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_s{classes}.pdf"\
            if isinstance(classes, int) else f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_s{len(classes)}.pdf",
        kind=kind)
    
    plot_heatmap_gs(
        heatmap=heatmap_var,
        savepdf=f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_var_s{classes}.png"\
            if isinstance(classes, int) else f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_var_s{len(classes)}.png",
        kind=kind
    )

    plot_heatmap_gs(
        heatmap=heatmap_var,
        savepdf=f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_var_s{classes}.pdf"\
            if isinstance(classes, int) else f"results/images/text_encoder/causal_trace/summary/grid_search_{metric}_{kind}_var_s{len(classes)}.pdf",
        kind=kind
    )




def trace_with_patch_text_encoder(
    pipe: StableDiffusionPipeline,
    item, # input item, including prompt, class name, seed 
    idx, # index of the item 
    states_to_patch,  # A list of (token index, layername) tuples to restore
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    sub_dir="causal_trace"
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    
    After aapted to Stable Diffusion, the generated images will be saved.
    No metric is computed in this function. Only images generated.

    **This function can only deal with text_encoder blocks!**
    """

    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    # patch_spec is a dict whose values are list
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername_text_encoder(pipe.text_encoder, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        """
        if layer == embed_layername, corrupt x[1:]
        else if layer != embed_layername, restore tokens in patch_spec 
        """
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    if not os.path.exists(f"{RESULTS_DIR}/images"):
        os.makedirs(f"{RESULTS_DIR}/images") 

    def generate_save(path, clean_path, prompt: str, seed):
        """
        Won't generate if the image in path already exists
        """
        # create the dir if not exists
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if not os.path.isfile(path):
            generator = torch.Generator("cuda").manual_seed(int(seed))
            images = pipe([prompt] * 2, guidance_scale=7.5, generator=generator).images
            images[1].save(path)
            if not os.path.isfile(clean_path):
                images[0].save(clean_path)

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        pipe,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        # generate images and save them
        # currently only support text encoder!
        # the format of the filename: 
        #   restoration: {class_name}_{idx}_{kind}_s{start_layer}_w{window_size}_restore_{token_to_restore}.png
        #              or {class_name}_{idx}_r{restore_single_layer}_restore_{token_to_restore}.png
        #              **token_to_restore == None means last token by default**
        #   pure corruption: {class_name}_{idx}_corrupt.png
        if len(states_to_patch) == 0:
            # pure corruption
            corrupted_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                                 f"{item['class name']}_{idx}_corrupt.png"
            clean_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                             f"{item['class name']}_{idx}_clean.png"
            generate_save(corrupted_img_path, clean_img_path, item["text prompt"], item["random seed"]) 

        else:
            # restoration
            # note that we assume only one token is restored!
            inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)

            if len(states_to_patch) == 1 and len(states_to_patch[0][1].split(".")) == 4:
                # decode the token to restore
                token_to_restore = pipe.tokenizer.decode(inp["input_ids"][0][states_to_patch[0][0]])
                # restore single layer
                restore_layer = states_to_patch[0][1].split(".")[3]
                restore_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                                   f"{item['class name']}_{idx}_r{restore_layer}_restore_{token_to_restore}.png"
                clean_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                                 f"{item['class name']}_{idx}_clean.png"
                generate_save(restore_img_path, clean_img_path, item["text prompt"], item["random seed"]) 
            else:
                # decode the token to restore, we assume only one token is restored!
                token_to_restore = pipe.tokenizer.decode(inp["input_ids"][0][states_to_patch[0][0]])
                # for start of <|startoftext|> token and end of token, we use SRT and EOT for naming
                if token_to_restore == "<|startoftext|>":
                    token_to_restore = "SRT"
                elif token_to_restore == "<|endoftext|>":
                    token_to_restore = "EOT"

                windwo_size = len(states_to_patch)
                start_layer = states_to_patch[0][1].split(".")[3]
                kind = "attn" if "attn" in states_to_patch[0][1] else "mlp"
                restore_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                    f"{item['class name']}_{idx}_{kind}_s{start_layer}_w{windwo_size}_restore_{token_to_restore}.png"
                clean_img_path = f"{RESULTS_DIR}/images/text_encoder/{sub_dir}/"\
                                 f"{item['class name']}_{idx}_clean.png"
                generate_save(restore_img_path, clean_img_path, item["text prompt"], item["random seed"]) 

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return all_traced
    return 


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername_text_encoder(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow_text_encoder(
    pipe: StableDiffusionPipeline,
    item: dict,
    idx: int,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    num_layers=None,
    kind=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.

    Note that this function only calculates the effect of a single object.
    And the random seed is fixed by input

    Args:
        pipe: the stable diffusion pipeline
        item: the input item, including text prompt, class name, attribute, random seed
        idx: the index of the item
        noise: the noise level
        token_range: the range of object tokens 
        uniform_noise: whether to use uniform noise
        replace: whether to replace with noise instead of adding noise
        num_layers: the number of layers of the text encoder model
        window: the window size
        kind: the kind of the layer, should be in [None, "attn", "mlp"], None for whole layer
    """
    # TODO consider adding strategies excluding examples where original prediction is bad
    # by setting BLIP text-image similarity threshold

    inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)
    object_range = find_token_range(pipe.tokenizer, inp["input_ids"][0], item["class name"])
    if token_range == "subject_last":
        token_range = [object_range[1] - 1]
    elif token_range is not None:
        # only restore the last object token, otherwise test every token
        raise ValueError(f"Unsupported token_range: {token_range}")
    
    # generate the corrupted image
    trace_with_patch_text_encoder(
        pipe, item, idx, [], object_range, noise=noise, uniform_noise=uniform_noise
    )
    if not kind:
        # track all the layers
        trace_important_states(
            pipe,
            item, 
            idx,
            num_layers,
            object_range,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        # trace all the kind sublayers
        trace_important_window_gs(
            pipe,
            item,
            idx,
            num_layers,
            object_range,
            kind=kind,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    return 


def trace_cross_proj(
    pipe: StableDiffusionPipeline,
    item,
    idx,
    num_layers,
    object_range,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)
    ntoks = inp["input_ids"].shape[1]

    if token_range is None:
        # test with every token
        token_range = range(ntoks)
    for tnum in token_range:
        for layer in tqdm(range(num_layers), disable=True):
            # no progress bar for single image generation
            # get decoded token of the token to restore
            trace_with_patch_text_encoder(
                pipe,
                item,
                idx, 
                [(tnum, layername_text_encoder(pipe.text_encoder, layer))],
                tokens_to_mix=object_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
    return 
    pass

def trace_important_states(
    pipe: StableDiffusionPipeline,
    item,
    idx,
    num_layers: int,
    object_range,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    """"
    trace all the layers of the text encoder
    """
    inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)
    ntoks = inp["input_ids"].shape[1]

    if token_range is None:
        # test with every token
        token_range = range(ntoks)
    for tnum in token_range:
        for layer in tqdm(range(num_layers), disable=True):
            # no progress bar for single image generation
            # get decoded token of the token to restore
            trace_with_patch_text_encoder(
                pipe,
                item,
                idx, 
                [(tnum, layername_text_encoder(pipe.text_encoder, layer))],
                tokens_to_mix=object_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
    return 


def trace_important_window_gs(
        pipe,
        item,
        idx,
        num_layers,
        object_range,
        kind,
        noise=0.1,
        uniform_noise=False,
        replace=False,
        token_range=None,
):
    """
    perform grid search on start layer and window size, and
    possibly the token to restore if token_range is None
    It is recommended to perform grid search on a single token

    Args:
        pipe: the stable diffusion pipeline
        item: the input item, including text prompt, class name, random seed
        idx: the index of the item
        num_layers: the number of layers of the text encoder model
        object_range: the range of object tokens, [start, end)
        kind: the kind of the layer, should be in [None, "attn", "mlp", "self_attn"], None for whole layer
        noise: the noise level
        uniform_noise: whether to use uniform noise
        replace: whether to replace with noise instead of adding noise
        token_range: an iterable of tokens to restore, None for all tokens
    """

    inp = make_inputs(pipe.tokenizer, [item["text prompt"]], device=pipe.device)
    ntoks = inp["input_ids"].shape[1]

    if token_range is None:
        # test with every token
        token_range = range(ntoks)

    start_layers = range(0, num_layers)
    window_sizes = range(1, num_layers + 1)

    # get full text_encoder layer names
    layer_names = [layername_text_encoder(pipe.text_encoder, L, kind) for L in range(num_layers)]
    states_to_patch = [[(tnum, layername) for layername in layer_names] for tnum in token_range]    

    for i, tum in enumerate(token_range):
        for start_layer in start_layers:
            for window_size in tqdm(window_sizes, disable=True):
                if start_layer + window_size > num_layers:
                    continue 
                states_to_patch_slice = states_to_patch[i][start_layer: start_layer + window_size] 
                trace_with_patch_text_encoder(
                    pipe,
                    item,
                    idx,
                    states_to_patch_slice,
                    object_range,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                )


            




def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in tqdm(range(num_layers), disable=True):
            layerlist = [
                (tnum, layername_text_encoder(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch_text_encoder(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class TextModelAndTokenizer:

    def __init__(
        self,
        model,
        tokenizer
    ):
        # the device of the model should be cuda
        self.model = model
        self.tokenizer = tokenizer
        # layer_names pattern : text_model.encoder.layers.{num} and nothing else
        self.layer_names = [
            n
            for n, m in self.model.named_modules()
            if re.match(r"text_model\.encoder\.layers\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername_text_encoder(model, num, kind=None):
    """
    Get the name of a layer or the name of module in the layer,
    which is specified by kind
    For UNet, unet_block should be in [up_blocks, down_blocks, mid_block]
    and num should be in the range of [0, 3]
    """
    if hasattr(model, "text_model"):
        # for CLIP text encoder
        if kind == "embed":
            return "text_model.embeddings"
        if kind is not None and kind not in ["self_attn", "mlp", "attn"]:
            raise ValueError(f"Unknown kind: {kind}", "should be in [self_attn, mlp, embed]")
        if kind == "attn":
            # attn is equal to self_attn
            kind = "self_attn"
        return f'text_model.encoder.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "transformer structure not supported"


def layername_unet(
        model, 
        unet_part, 
        level: int, 
        block,
        layer: int, 
        attn_type="self_attn", 
        kind=None):
        
    # TODO: use python typing for better type checking
    assert hasattr(model, "up_blocks"), "model is not a UNet"
    assert unet_part in ["up_blocks", "down_blocks", "mid_block"], \
        "unknown unet part, should be in [up_blocks, down_blocks, mid_block]"
    assert block in ["resnets", "attentions"] 
    assert attn_type in ["self_attn", "cross_attn"]   
    assert kind in [None, "attn", "mlp"]

    if unet_part == "up_blocks":
        if block == "attentions":
            # check whether the layer exsits
            attn_layer = "attn1" if attn_type == "self_attn" else "attn2"
            try:
                attn = reduce(getattr, \
                       [model, unet_part, str(level), "attentions", str(layer),\
                        "transformer_blocks", "0", attn_layer])
                if kind is not None:
                    attn = reduce(getattr, [attn, kind])
            except AttributeError:
                raise ValueError(f"{level} layer of {unet_part} does not have self attention")
            return f'{unet_part}.{level}.{block}.{layer}.transformer_blocks.0.{attn_layer}{"" if kind is None else "." + kind}'
        elif block == "resnets":
            raise NotImplementedError("resnets in up_blocks is not implemented")
    elif unet_part == "down_blocks":
        if block == "attentions":
            # check whether the layer exsits
            attn_layer = "attn1" if attn_type == "self_attn" else "attn2"
            try:
                attn = reduce(getattr, \
                       [model, unet_part, str(level), "attentions", str(layer),\
                        "transformer_blocks", "0", attn_layer])
                if kind is not None:
                    attn = reduce(getattr, [attn, kind])
            except AttributeError:
                raise ValueError(f"{level} layer of {unet_part} does not have self attention")
            return f'{unet_part}.{level}.{block}.{layer}.transformer_blocks.0.{attn_layer}{"" if kind is None else "." + kind}'
        elif block == "resnets":
            raise NotImplementedError("resnets in down_blocks is not implemented")
    elif unet_part == "mid_block":
        if block == "attentions":
            # check whether the layer exsits
            attn_layer = "attn1" if attn_type == "self_attn" else "attn2"
            try:
                attn = reduce(getattr, \
                       [model, unet_part, "attentions", str(layer),\
                        "transformer_blocks", "0", attn_layer])
                if kind is not None:
                    attn = reduce(getattr, [attn, kind])
            except AttributeError:
                raise ValueError(f"{level} layer of {unet_part} does not have self attention")
            return f'{unet_part}.{block}.{layer}.transformer_blocks.0.{attn_layer}{"" if kind is None else "." + kind}'
        elif block == "resnets":
            raise NotImplementedError("resnets in mid_block is not implemented")


def cal_heatmap(
    result_json,
    token_to_restore,
    class_name,
    kind,
    metric="blip",
):
    """
    Calculate the heatmap of a single class according to the result json file
    Return:
        heatmap: the heatmap
        low_score: the low score of the heatmap
        max_window_size: the max window size of the heatmap
        max_start_layer: the max start layer of the heatmap
        idx_to_remove: the idx of the prompts to remove if BLIP performs bad
    """
    with open(result_json, "r") as f:
        result = json.load(f)
    
    # change the reuslt into a list of ImageItem objects
    img_items: List[ImageItem]  = []
    for i, item in enumerate(result):
        try:
            img_items.append(ImageItem(item["image_path"]))
        except ValueError:
            print(item["image_path"])
        img_items[i].matching_score = item["matching_score"]

    # find the pure corruption item
    # since every promt has a pure corruption item,
    # we use the average as the low score
    # find the pure corruption item, and for these items, if matching score still bigger than 0.5,
    # the prompt is considered out of distribution for BLIP. We will remove the prompt
    idx_to_remove: List[int] = []
    pure_corruption_scores = []
    if metric == "blip":
        for item in img_items:
            if item.class_name != class_name: continue
            if item.is_corrupted and item.matching_score > 0.6:
                print(f"remove prompt: {item.idx}")
                idx_to_remove.append(item.idx)
            elif item.is_clean and item.matching_score <= 0.6:
                print(f"remove prompt: {item.idx}")
                idx_to_remove.append(item.idx)
            elif item.is_corrupted:
                pure_corruption_scores.append(item.matching_score)
    low_score = None if len(pure_corruption_scores) == 0 \
                     else np.average(pure_corruption_scores)

    # keep only the items with the same token to restore and class name
    assert kind in ["attn", "mlp"],\
          "This is grid search plot, kind should be in [attn, mlp]"
    img_items = [item for item in img_items \
                if item.token_to_restore == token_to_restore \
                    and item.class_name == class_name\
                    and item.kind == kind\
                    and item.idx not in idx_to_remove]
    
    # get the max window size and start layer
    max_window_size = max([item.restore_window for item in img_items])
    max_start_layer = max([item.start_layer for item in img_items]) 
    # start layer is 0-indexed

    # create the heatmap
    heatmap = np.zeros((max_window_size, max_start_layer + 1))
    num_items = np.zeros((max_window_size, max_start_layer + 1))

    for item in img_items:
        heatmap[item.restore_window - 1, item.start_layer] += item.matching_score
        num_items[item.restore_window - 1, item.start_layer] += 1
    
    # normalize the heatmap, for zero division, we use 0
    for i in range(max_window_size):
        for j in range(max_start_layer + 1):
            if num_items[i, j] != 0:
                heatmap[i, j] /= num_items[i, j]
    
    # plot the difference heatmap if low score is not None
    if low_score is not None:
        # an array whose values are low_score
        low = np.full((max_window_size, max_start_layer + 1), low_score)
        low = np.where(num_items == 0, 0, low)
        heatmap = heatmap - low
    return heatmap, low_score, max_window_size, max_start_layer, idx_to_remove
    

def plot_heatmap_gs(
    heatmap,
    low_score=0,
    kind="mlp",
    max_window_size=12,
    max_start_layer=11,
    savepdf=f"{RESULTS_DIR}/plots/text_encoder/causal_trace/heatmap.png",
    title=None,
    xlabel=None,
    ylabel=None
):
    """
    Given 
    """
    # plot the heatmap
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            heatmap,
            cmap={"mlp": "Greens", "attn": "Reds"}[kind],
            vmin=low_score,
        )
        ax.set_yticks([0.5 + i for i in range(max_window_size)])
        ax.set_xticks([0.5 + i for i in range(max_start_layer + 1)])
        ax.set_xticklabels(list(range(max_start_layer + 1)))
        ax.set_yticklabels(list(range(1, max_window_size + 1)))

        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(f"Impact of restoring {kindname} after corrupted input")
        ax.set_xlabel(f"start layer")
        ax.set_ylabel(f"window size")

        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def plot_grid_search_result(
        result_json, 
        token_to_restore, 
        class_name,
        kind,
        metric="blip",
        savepdf=f"{RESULTS_DIR}/plots/text_encoder/causal_trace/heatmap.png",
        title=None,
        xlabel=None,
        ylabel=None
        ):

    """
    Read a json file which records a list of ImageItem objects,
    and plot the result as a heatmap, X axis is the start layer, Y axis is the window size.
    For non-exist point, we use white color
    """

    result = cal_heatmap(result_json, token_to_restore, class_name, kind, metric=metric)
    heatmap, low_score, max_window_size, max_start_layer, idx_to_remove = result
    
    plot_heatmap_gs(heatmap)
    
    print("Done. Removed prompt length: ", len(idx_to_remove))
        







# --------------------------------------------

def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow_text_encoder(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer: CLIPTokenizer, prompts, device="cuda:0"):
    """
    Here we assume that we are using CLIPTokenizer
    """
    inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt", 
        truncation=True, 
        max_length=77)
    
    # move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def normalize_unicode_string(string):
    normalized_string = unicodedata.normalize('NFKC', string)
    return normalized_string


def find_token_range(tokenizer, token_array, substring_orig):
    """
    The returened range is [start, end)
    """
    substring = substring_orig[:]
    if substring == "[CLS]":
        return (0, 1)
    elif substring == "[EOS]" or substring == "" or substring == " ":
        return (len(token_array) - 1, len(token_array))

    substring = substring.replace(" ", "")
    substring = substring.lower()
    # this may have two unrecognized tokens for a single ń
    # TODO: current solution is temporary, need to find a better way
    # tokenizer of StableDiffusonPipeline cannot correctly tokenize '
    # so we need to mannually replace ' 
    toks = decode_tokens(tokenizer, token_array)
    decoded_toks = tokenizer.decode(token_array)
    whole_string = decoded_toks.replace(" ", "")
    # temporary fix
    if "’" in substring:
        whole_string = whole_string.replace("'", "’")

    # normalize the string
    whole_string = normalize_unicode_string(whole_string)
    substring = normalize_unicode_string(substring)
    try:
        char_loc = whole_string.index(substring)
    except ValueError:
        print("Cannot find substring in tokens")
        print("substring: ", substring)
        print("whole string: ", whole_string)
        raise ValueError
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        if "ń" in substring and token_array[i] == 78:
            # ń is encoded as 2 tokens, but with only 1 char
            pass
        else:
            loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s], device=mt.model.device)
        with nethook.Trace(mt.model, layername_text_encoder(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level




if __name__ == "__main__":
    main()
