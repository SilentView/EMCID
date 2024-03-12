import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Any, Tuple, Dict
import argparse

import torch
import numpy as np

from util.hparams import HyperParams
from util.globals import *


@dataclass
class ContrastEMCIDHyperParams(HyperParams):
    layers: List[int]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]

    mom2_update_weight: int

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # optimization
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    v_loss_layer: int
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool   
    use_negative_images: bool
    num_negative_images: int

    objective: str = "contrastive"
    v_prob_threshold: float = 0.99
    edit_weight: float = 0.5
    sld_supervision: bool=False
    follow_refact: bool = True
    use_diff_clip: bool = False # use openai/clip-vit-large-patch14 or openai/clip-vit-large-patch14-336


@dataclass
class EMCIDHyperParams(HyperParams):
    # TODO remove redundant fields
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]

    mom2_update_weight: int

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str


    # optimization
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float    # clamp norm means that the norm of the gradient is clamped to this value

    mom2_adjustment: bool   
    objective: Literal["esd", "ablate-dest", "ablate-source"]
    esd_mu: Optional[Any]


    train_prompt_choice: Literal["complicated", "simple"] = "simple"
    use_new_compute_z: bool = False
    num_edit_tokens: int = 1    # 1 for last subject token ,2 will add eos token, more tokens will be padding tokens

    samples_per_prompt: int = 1
    edit_weight: float = 0.5
    cal_text_repr_loss: bool = False
    align_obj_eos_pad: bool = False
    text_repr_loss_scale_factor: float = 0.0
    txt_img_align_scale_factor: float = 0.0
    txt_img_align_loss_metric: Literal["l2", "cos"] = "l2"
    contrastive_text_loss: bool = False
    align_object_token: bool = False
    follow_refact: bool = True
    use_ewc: bool = False
    ewc_lambda: int = 1e4
    no_noise_loss: bool = False
    ddim_steps: int = None,    # only considered for UCE and i2p complementary evaluation
    scheduler: str = None           # only considered for UCE and i2p complementary evaluation
    
    # global concepts related
    sld_supervision: bool=False
    sld_type: str="max"
    all_safe: bool=False    # use all the sld safety concepts to generate
                            # sld supervision

    add_uce_edit: bool = False
    
    # adding new concepts related
    use_sampled_noise: bool = False
    replace_repr:bool = False



    @classmethod
    def get_name(cls, hparam: "EMCIDHyperParams"):
        """
        example "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-4"
        """
        prefix = ""
        if hparam.use_sampled_noise:
            prefix += "add_dest"
        elif hparam.objective == "esd":
            prefix += "esd"
            prefix += f"-{hparam.esd_mu}"
        elif hparam.objective == "ablate-dest":
            prefix += "dest"
        elif hparam.objective == "ablate-source":
            prefix += "source"
        else:
            raise ValueError("objective not supported")
        
        suffix = ""
        if hparam.cal_text_repr_loss and not hparam.contrastive_text_loss:
            suffix += f"_txt-align-{hparam.text_repr_loss_scale_factor}"
        elif hparam.contrastive_text_loss:
            suffix += f"_txt-cont-{hparam.text_repr_loss_scale_factor}"

        return f"{prefix}_s-{hparam.v_num_grad_steps}_"\
                f"c-{hparam.clamp_norm_factor}_ly-{len(hparam.layers)}_"\
                f"lr-{hparam.v_lr}_wd-{hparam.v_weight_decay:.0e}"\
                f"{suffix}"
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    

    @classmethod
    def to_json(cls, hparam: "UNetEMCIDHyperParams"):
        file_name = cls.get_name(hparam)
        with open(HPARAMS_DIR / f"{file_name}.json", "w") as f:
            json.dump(hparam.__dict__, f, indent=4)


@dataclass
class EMCIDXLHyperParams(HyperParams):
    # TODO remove redundant fields
    # Method
    layers: List[int]
    layers_2: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]

    mom2_update_weight: int
    mom2_update_weight_2: int

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str


    # optimization
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float    # clamp norm means that the norm of the gradient is clamped to this value

    mom2_adjustment: bool   
    objective: Literal["esd", "ablate-dest", "ablate-source"]
    esd_mu: Optional[Any]


    train_prompt_choice: Literal["complicated", "simple"] = "simple"
    use_new_compute_z: bool = False
    num_edit_tokens: int = 1    # 1 for last subject token ,2 will add eos token, more tokens will be padding tokens

    samples_per_prompt: int = 1
    edit_weight: float = 0.5
    cal_text_repr_loss: bool = False
    align_obj_eos_pad: bool = False
    text_repr_loss_scale_factor: float = 0.0
    txt_img_align_scale_factor: float = 0.0
    txt_img_align_loss_metric: Literal["l2", "cos"] = "l2"
    contrastive_text_loss: bool = False
    align_object_token: bool = False
    follow_refact: bool = True
    use_ewc: bool = False
    ewc_lambda: int = 1e4
    no_noise_loss: bool = False
    ddim_steps: int = None,    # only considered for UCE and i2p complementary evaluation
    scheduler: str = None           # only considered for UCE and i2p complementary evaluation
    
    # global concepts related
    sld_supervision: bool=False
    sld_type: str="max"
    all_safe: bool=False    # use all the sld safety concepts to generate
                            # sld supervision

    add_uce_edit: bool = False
    
    # adding new concepts related
    use_sampled_noise: bool = False
    replace_repr:bool = False



    @classmethod
    def get_name(cls, hparam: "EMCIDHyperParams"):
        """
        example "sdxl-dest_s-200_c-1.5_ly-11_ly2-31_lr-0.2_wd-5e-4_text-align_0.01"
        """
        prefix = "sdxl-"
        if hparam.use_sampled_noise:
            prefix += "add_dest"
        elif hparam.objective == "esd":
            prefix += "esd"
            prefix += f"-{hparam.esd_mu}"
        elif hparam.objective == "ablate-dest":
            prefix += "dest"
        elif hparam.objective == "ablate-source":
            prefix += "source"
        else:
            raise ValueError("objective not supported")
        
        suffix = ""
        if hparam.cal_text_repr_loss and not hparam.contrastive_text_loss:
            suffix += f"_txt-align-{hparam.text_repr_loss_scale_factor}"
        elif hparam.contrastive_text_loss:
            suffix += f"_txt-cont-{hparam.text_repr_loss_scale_factor}"

        return f"{prefix}_s-{hparam.v_num_grad_steps}_"\
                f"c-{hparam.clamp_norm_factor}_ly-{len(hparam.layers)}_"\
                f"lr-{hparam.v_lr}_wd-{hparam.v_weight_decay:.0e}"\
                f"{suffix}"
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    

    @classmethod
    def to_json(cls, hparam: "EMCIDHyperParams"):
        file_name = cls.get_name(hparam)
        with open(HPARAMS_DIR / f"{file_name}.json", "w") as f:
            json.dump(hparam.__dict__, f, indent=4)
    
    

@dataclass
class UNetEMCIDHyperParams(HyperParams):
    # Method
    final_layer: list
    spread_sub_block_cnt: int
    skip_res_conv: bool
    v_reduce_inside_img: bool
    v_reduce_for_concept: bool
    gloabl_sample: bool
    num_t_blocks: int
    even_sample: bool

    # optimization
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float    
    objective: Literal["esd", "ablate-source"]
    esd_mu: Optional[Any] 
                            
    mom2_update_weight: int

    # Module templates
    rewrite_module_tmp: Dict[str, str]

    # Statistics
    mom2_dataset: str
    mom2_n_samples_prompts: int
    mom2_n_steps_per_prompt: int
    mom2_dtype: str
    
    use_sampled_noise: bool = False

    @classmethod
    def get_name(cls, hparam: "UNetEMCIDHyperParams"):
        """
        example "unet_source_s-200_c-1.5_fb-down3_spread-2_lr-0.2_wd-5e-4"
        """
        prefix = "unet_"
        if hparam.use_sampled_noise:
            prefix += "add_dest"
        elif hparam.objective == "esd":
            prefix += "esd"
            prefix += f"-{hparam.esd_mu}"
        elif hparam.objective == "ablate-source":
            prefix += "source"
        else:
            raise ValueError("objective not supported")
        
        return f"{prefix}_s-{hparam.v_num_grad_steps}_"\
                f"c-{hparam.clamp_norm_factor}_"\
                f"ly-{hparam.final_layer[0]}{hparam.final_layer[1]}-{hparam.final_layer[2]}_"\
                f"spread-{hparam.spread_sub_block_cnt}_"\
                f"tb-{hparam.num_t_blocks}_"\
                f"lr-{hparam.v_lr}_wd-{hparam.v_weight_decay:.0e}"\
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

template_unet = {
    "final_layer": ["up_blocks", 3, "attn-out"],
    "spread_sub_block_cnt": 5,
    "gloabl_sample": True,
    "v_num_grad_steps": 200,
    "v_lr": 0.2,
    "v_weight_decay": 5e-4,
    "clamp_norm_factor": 1.5,
    "objective": "source",
    "esd_mu": None,
    "mom2_update_weight": 10000,
    "rewrite_module_tmp": {
        "mlp": "{}.{}.attentions.{}.transformer_blocks.0.ff.net.2",
        "conv-res": "{}.{}.resnets.{}.conv2",
        "conv-sample": "{}.{}.{}.0.conv"
    },
    "mom2_dataset": "css_filtered",
    "mom2_n_samples_prompts": 10000,
    "mom2_n_steps_per_prompt": 100,
    "mom2_dtype": "float32",
    "use_sampled_noise": False,
    "skip_res_conv": False,
    "v_reduce_inside_img": True,
    "v_reduce_for_concept": True,
    "num_t_blocks": 50,
    "even_sample": True
}



template_text_encoder = {
    "layers": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ],
    "clamp_norm_factor": 1.5,
    "layer_selection": "all",
    "fact_token": "subject_last",
    "v_num_grad_steps": 200,
    "v_lr": 0.2,
    "v_weight_decay": 5e-4,
    "mom2_adjustment": True,
    "mom2_update_weight": 10000,
    "rewrite_module_tmp": "text_model.encoder.layers.{}.mlp.fc2",
    "layer_module_tmp": "text_model.encoder.layers.{}",
    "mlp_module_tmp": "text_model.encoder.layers.{}.mlp",
    "attn_module_tmp": "text_model.encoder.layers.{}.self_attn",
    "ln_f_module": "text_model.final_layer_norm",
    "mom2_dataset": "ccs_filtered",
    "mom2_n_samples": 100000,
    "mom2_dtype": "float32",
    "objective": "ablate-source",
    "esd_mu": "None" 
}

def make_hparam(
   clamp_norm_factor: float=None,
    v_num_grad_steps: int=None,
    v_lr: float=None,
    v_weight_decay: float=None,
):
    template_text_encoder["clamp_norm_factor"] = clamp_norm_factor \
        if clamp_norm_factor is not None else template_text_encoder["clamp_norm_factor"]
    template_text_encoder["v_num_grad_steps"] = v_num_grad_steps \
        if v_num_grad_steps is not None else template_text_encoder["v_num_grad_steps"]
    template_text_encoder["v_lr"] = v_lr \
        if v_lr is not None else template_text_encoder["v_lr"]
    template_text_encoder["v_weight_decay"] = v_weight_decay \
        if v_weight_decay is not None else template_text_encoder["v_weight_decay"]
    
    hparam = EMCIDHyperParams.from_dict(template_text_encoder)
    # make a file if not exists
    name = EMCIDHyperParams.get_name(hparam)
    if not os.path.exists("hparams/" + name + ".json"):
        with open("hparams/" + name + ".json", "w") as f:
            json.dump(template_text_encoder, f, indent=4)
    return hparam


def get_accum_time_blocks(num_block=50, is_even=True, time_steps=1000):
    """
    We split 1000 time steps into num_block blocks.
    The return value is the accumulated sum results of the size
    of each block.
    for example, [50, 100, ....] means
    [[0, 50), [50, 100), ...
    """
    if is_even:
        sizes = [time_steps // num_block] * num_block
    else:
        raise NotImplementedError
    return torch.cumsum(torch.tensor(sizes), dim=0).tolist() 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clamp_norm_factor", type=float, default=None)
    parser.add_argument("--v_num_grad_steps", type=int, default=None)
    parser.add_argument("--v_lr", type=float, default=None)
    parser.add_argument("--v_weight_decay", type=float, default=None)

    args = parser.parse_args()

    make_hparam(
        clamp_norm_factor=args.clamp_norm_factor,
        v_num_grad_steps=args.v_num_grad_steps,
        v_lr=args.v_lr,
        v_weight_decay=args.v_weight_decay,
    )