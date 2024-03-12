"""Adatped from UCE https://github.com/rohitgandikota/unified-concept-editing/blob/main/train-scripts/train_erase.py"""
import argparse
import random
import os
import json
from typing import List, Tuple, Union, Dict, Any, Optional, Literal
import copy
import ast

import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

from dsets.iceb_dataset import ObjectPromptDataset, RequestDataset, compose_alias_test_requests
from dsets.global_concepts import NSFWEditRequestDataset
from dsets.artist_requests import ArtistRequestsDataset
from emcid.layer_stats import get_attr_through_name, get_all_cross_attn_kv_layer_names


from diffusers.models import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm

from util.globals import *
from util import nethook


def edit_text_encoder_uce(
        pipe, 
        old_text_, 
        new_text_, 
        retain_text_, 
        add=False, 
        layer_to_edit=11, 
        lamb=0.1, 
        erase_scale = 0.1, 
        preserve_scale = 0.1, 
        with_to_k=True, 
        technique='tensor'):
    ### collect all the cross attns modules
    # collect one text encoder layer
    module_name = f"text_model.encoder.layers.{layer_to_edit}.mlp.fc2"
    module = get_attr_through_name(pipe.text_encoder, module_name)

    projection_matrix = module

    ### get the value and key modules
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    ######################## START ERASING ###################################
    #### prepare input k* and v*
    with torch.no_grad():
        #mat1 = \lambda W + \sum{v k^T}
        mat1 = lamb * projection_matrix.weight

        #mat2 = \lambda I + \sum{k k^T}
        mat2 = lamb * torch.eye(projection_matrix.weight.shape[1], device=projection_matrix.weight.device)

        for cnt, t in enumerate(zip(old_texts, new_texts)):
            old_text = t[0]
            new_text = t[1]
            texts = [old_text, new_text]
            
            with nethook.TraceDict(
                    module=pipe.text_encoder,
                    layers=[
                        module_name
                    ],
                    retain_input=True,
                    retain_output=True,
                    edit_output=None,
                    clone=True
                ) as td:

                text_input = pipe.tokenizer(
                        texts,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
                input_embeddings = td[module_name].input
                output_embeddings = td[module_name].output
                print(f"input_embeddings shape: {input_embeddings.shape}")
                print(f"output_embeddings shape: {output_embeddings.shape}")

                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])

                old_input_emb = input_embeddings[0]
                ## all the tokens after last subject token
                old_input_emb = old_input_emb[final_token_idx:len(old_input_emb)-max(0,farthest-final_token_idx)]

                ## last subject token and EOS and another padding
                # old_emb = old_emb[final_token_idx:final_token_idx+3]

                ## only the last token
                # old_emb = old_emb[final_token_idx:final_token_idx+1]

                ## only the EOS token
                # old_emb = old_emb[final_token_idx+1:final_token_idx+2]

                ## only the last subject token and EOS
                # old_emb = old_emb[final_token_idx:final_token_idx+2]

                # print(f"orig old_emb shape: {old_emb.shape}")

                new_input_emb = input_embeddings[1]

                ## all the tokens after EOS
                new_input_emb = new_input_emb[final_token_idx_new:len(new_input_emb)-max(0,farthest-final_token_idx_new)]

                # last subject token and EOS and another padding
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+3]

                ## only the last subject token
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+1]

                ## only the EOS token
                # new_emb = new_emb[final_token_idx_new+1:final_token_idx_new+2]

                ## only the last subject token and EOS
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+2]

                # print(f"new_emb shape: {new_emb.shape}")
                
                context = old_input_emb.detach()

                value = None
                layer = module
                with torch.no_grad():
                    if technique == 'tensor':
                        print(layer.weight.shape)
                        o_embs = layer(old_input_emb).detach()
                        u = o_embs
                        u = u / u.norm()
                        
                        new_embs = layer(new_input_emb).detach()
                        new_emb_proj = (u*new_embs).sum()
                        
                        source = new_embs - (new_emb_proj)*u 
                        # print(f"source shape: {source.shape}")
                        value = source.detach()
                    elif technique == 'replace':
                        value = layer(new_input_emb).detach()
                    else:
                        value = layer(new_input_emb).detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = value.reshape(value.shape[0], value.shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = pipe.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                with nethook.TraceDict(
                    module=pipe.text_encoder,
                    layers=[
                        module_name
                    ],
                    retain_input=True,
                    retain_output=True,
                    edit_output=None,
                    clone=True
                ) as td:
                    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
                    old_input_emb, new_input_emb = td[module_name].input
                    context = old_input_emb.detach()
                    value = None
                    with torch.no_grad():
                        value = layer(new_input_emb[:]).detach()
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = value.reshape(value.shape[0], value.shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += preserve_scale*for_mat1
                    mat2 += preserve_scale*for_mat2
        #update projection matrix
        module.weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))
    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return pipe


def edit_model_uce(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    max_bias_diff = 0.05
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    
    ca_layer_names = get_all_cross_attn_kv_layer_names(ldm_stable)
    # above has to_k, to_v, remove them
    ca_layer_names = [name.replace('.to_k', '').replace('.to_v', '') for name in ca_layer_names]
    ca_layers = [get_attr_through_name(ldm_stable.unet, name) for name in ca_layer_names]
    

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    projection_matrices_names = [name + '.to_v' for name in ca_layer_names]
    for l in ca_layers:
        print(f"layer: {l}")
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
        projection_matrices_names = projection_matrices_names + [name + '.to_k' for name in ca_layer_names]

    for idx_, l in enumerate(ca_layers):
        # print the name of the layer
        print(l)
    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    ######################## START ERASING ###################################
    for layer_num in range(len(projection_matrices)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                
                
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = text_embeddings[0]

                ## all the tokens after last subject token
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]

                ## last subject token and EOS and another padding
                # old_emb = old_emb[final_token_idx:final_token_idx+3]

                ## only the last token
                # old_emb = old_emb[final_token_idx:final_token_idx+1]

                ## only the EOS token
                # old_emb = old_emb[final_token_idx+1:final_token_idx+2]

                ## only the last subject token and EOS
                # old_emb = old_emb[final_token_idx:final_token_idx+2]

                # print(f"orig old_emb shape: {old_emb.shape}")

                new_emb = text_embeddings[1]

                ## all the tokens after EOS
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]

                # last subject token and EOS and another padding
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+3]

                ## only the last subject token
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+1]

                ## only the EOS token
                # new_emb = new_emb[final_token_idx_new+1:final_token_idx_new+2]

                ## only the last subject token and EOS
                # new_emb = new_emb[final_token_idx_new:final_token_idx_new+2]

                # print(f"new_emb shape: {new_emb.shape}")
                
                context = old_emb.detach()

                # print((old_emb[-1] - old_emb[-2]).norm().item())
                # print((old_emb[-1] - old_emb[1]).norm().item())
                # print((old_emb[-1] - old_emb[0]).norm().item())
                # raise Exception("stop")

                # # use optimized old_emb
                # cache_full = f"cache/dest_s-200_c-3.0_cross_lr-0.2_wd-5e-04/i2p_test/source_{old_text}.npz"
                
                # data = np.load(cache_full, allow_pickle=True)
                # layer_name = projection_matrices_names[layer_num]
                # old_emb = torch.from_numpy(data[layer_name].item()["v_star"]).to(ldm_stable.device)
                # print(f"old_emb shape: {old_emb.shape}")
                
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            
                            source = new_embs - (new_emb_proj)*u 
                            # print(f"source shape: {source.shape}")
                            values.append(source.detach()) 
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable

def edit_model_uce_modified(
    ldm_stable: StableDiffusionPipeline,
    old_text_: List[str],
    new_text_: List[str],
    lamb=0.1,
    mom2_weight=0,
    erase_scale=0.1,
    preserve_scale=0.1,
    with_to_k=True,
    layers_to_edit=None,
    technique='tensor',
):
    """
    Invariant: object to edit or to preserve must be the last word in the text
    """
    for old, new in zip(old_text_, new_text_):
        print(f'[{old}] -> [{new}]')
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)

    print(old_texts, new_texts)

    # get cov
    from emcid.emcid_main import get_cov_cross_attn

    cov = get_cov_cross_attn(
        ldm_stable, 
        layer_name="up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v",
        mom2_dataset="ccs_filtered",
        sample_size=100000,
        mom2_dtype="float32"
    )



    ######################## START ERASING ###################################
    for layer_num in range(len(projection_matrices)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                
                
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = text_embeddings[0]
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                new_emb = text_embeddings[1]
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]
                
                context = old_emb.detach()

                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            
                            source = new_embs - (new_emb_proj)*u 
                            values.append(source.detach()) 
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2
                
                # layer = projection_matrices[layer_num]
                # if technique == 'tensor':
                #     o_embs = layer(old_emb).detach()
                #     u = o_embs
                #     u = u / u.norm()
                    
                #     new_embs = layer(new_emb).detach()
                #     new_emb_proj = (u*new_embs).sum()
                    
                #     source = new_embs - (new_emb_proj)*u 
                #     value = source.detach() 
                # elif technique == 'replace':
                #     value = layer(new_emb).detach()
                # else:
                #     value = layer(new_emb).detach()

                # context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])

                # value_vector = value.reshape(value.shape[0], value.shape[1], 1)
                # for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                # for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                # mat1 += erase_scale*for_mat1
                # mat2 += erase_scale*for_mat2
            
            
            for_mat1 = mom2_weight * projection_matrices[layer_num].weight @ cov
            for_mat2 = mom2_weight * cov
            mat1 += preserve_scale*for_mat1
            mat2 += preserve_scale*for_mat2
            #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    return ldm_stable



def get_ratios(ldm_stable, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
    global clip_processor, clip_model
    seeds = np.random.randint(5000,size=5) 
    ratios = []
    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            if ratio_diff[idx] < max_ratio_gap:
                print(f'Bypassing Concept {idx+1}')
                ratios.append(prev_ratio[idx])
                continue
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed in seeds:
    #             if i == num_loops:
    #                 break
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed))
                images = ldm_stable(prompt,num_images_per_prompt=num_samples, num_inference_steps=20, generator = g).images

                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))
#     male = float(probs[0][0])
    return ratios

## get arguments for our script
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def edit_model_debias(
        ldm_stable, 
        old_text_, 
        new_text_, 
        retain_text_, 
        add=True, 
        layers_to_edit=None, 
        lamb=0.1, 
        erase_scale = 0.1, 
        preserve_scale = 0.1, 
        with_to_k=True, 
        num_images=1):

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ### collect all the cross attns modules
    max_bias_diff = 0.05
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = []
        for t in new_text:
            if (old_text.lower() not in t.lower()) and add:
                n_t.append(t + ' ' +old_text)
            else:
                n_t.append(t)
        if len(n_t) == 1:
            n_t = n_t*2
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    desired_ratios = [torch.ones(len(c))/len(c) for c in new_texts ]
    weight_step = 0.1
    weights = [torch.zeros(len(c)) for c in new_texts ]
    #################################### START OUTER LOOP #########################
    for i in range(30):
        max_ratio_gap = max_bias_diff
        if i == 0:
            prev_ratio = None
            ratio_diff = None
        else:
            prev_ratio = ratios
            ratio_diff = max_change
        ratios = [0 for _ in desired_ratios]
        ratios = get_ratios(ldm_stable=ldm_stable, prev_ratio = prev_ratio, ratio_diff=ratio_diff, max_ratio_gap=max_ratio_gap, concepts=old_texts, classes=new_texts, num_samples= num_images)
        if i == 0 :
            init_ratios = ratios
        print(ratios)
        max_change = [(ratio - desired_ratio).abs().max() for ratio, desired_ratio in zip(ratios,desired_ratios)]


        if max(max_change) < max_bias_diff:
            print(f'All concepts are debiased at Iteration:{i}')
            break
         #### restart LDM parameters
#         num_ca_clip_layers = len(ca_layers)
#         for idx_, l in enumerate(ca_layers):
#             l.to_v = copy.deepcopy(og_matrices[idx_])
#             projection_matrices[idx_] = l.to_v
#             if with_to_k:
#                 l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
#                 projection_matrices[num_ca_clip_layers + idx_] = l.to_k
        
        weights_delta = [weight_step * (desired_ratio - ratio) for ratio, desired_ratio in zip(ratios, desired_ratios)]
        weights_delta = [weights_delta[idx] if max_c>max_bias_diff else weights_delta[idx]*0 for idx, max_c in enumerate(max_change)]
        
        # check if the ratio is attained. If so, add it to preservation and skip the ratios check again
        ret_text_add = [old_texts[idx] for idx, weight in enumerate(weights_delta) if weight[0]==0]
        if len(ret_text_add)>0:
            ret_texts = ret_texts+ret_text_add
            ret_texts = list(np.unique(ret_texts))
        weights = weights_delta
#         weights = [weight + weights_delta[idx] for idx, weight in enumerate(weights)]
        ### START EDIT

        for layer_num in range(len(projection_matrices)):
            if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
                continue

            #### prepare input k* and v*
            with torch.no_grad():
                #mat1 = \lambda W + \sum{v k^T}
                mat1 = lamb * projection_matrices[layer_num].weight

                #mat2 = \lambda I + \sum{k k^T}
                mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

                for cnt, t in enumerate(zip(old_texts, new_texts)):
                    old_text = t[0]
                    new_text = t[1]
                    texts = [old_text]
                    texts = texts + new_text
                    text_input = ldm_stable.tokenizer(
                        texts,
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                    old_emb = text_embeddings[0]
                    final_token_idx = text_input.attention_mask[0].sum().item()-2
                    final_token_idx_new = [text_input.attention_mask[i].sum().item()-2 for i in range(1, len(text_input.attention_mask))]
                    farthest = max(final_token_idx_new+[final_token_idx])
                    new_emb = text_embeddings[1:]

                    context = old_emb.detach()[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            o_embs = layer(old_emb).detach()
                            o_embs = o_embs[final_token_idx:len(o_embs)-max(0,farthest-final_token_idx)]
#                             print(f'O_EMBS: {final_token_idx}-{len(o_embs)-max(0,farthest-final_token_idx)}')
                            embs = layer(new_emb[:]).detach()
                            source = o_embs
                            for j, emb in enumerate(embs):
                                u = emb
                                u = u[final_token_idx_new[j]:len(u)-max(0,farthest-final_token_idx_new[j])]
#                                 print(f'U_{j}: {final_token_idx_new[j]}-{len(u)-max(0,farthest-final_token_idx_new[j])}')
                                u = u / u.norm()
                                o_emb_proj = (u*o_embs).sum()
                                source += (weights[cnt][j]*o_embs.norm())*u 
                            values.append(source.detach())    
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += erase_scale*for_mat1
                    mat2 += erase_scale*for_mat2

                for old_text, new_text in zip(ret_texts, ret_texts):
                    text_input = ldm_stable.tokenizer(
                        [old_text, new_text],
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                    old_emb, new_emb = text_embeddings
                    context = old_emb.detach()
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            values.append(layer(new_emb[:]).detach())
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += preserve_scale*for_mat1
                    mat2 += preserve_scale*for_mat2
                    #update projection matrix
                projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable, weights, init_ratios, ratios