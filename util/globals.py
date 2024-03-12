from pathlib import Path

import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR, CACHE_DIR,
 XL_STATS_DIR1, XL_STATS_DIR2) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
        data["CACHE_DIR"],
        data["XL_STATS_DIR1"],
        data["XL_STATS_DIR2"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
RESOLUTION = data["RESOLUTION"]
EDITING_PROMPTS_CNT = data["EDITING_PROMPTS_CNT"]
LATENT_SIZE = 64

UNET_EDIT_TEMPLATES = {
        "attn-mlp": "{}.{}.attentions.{}.transformer_blocks.0.ff.net.2",
        "attn-out": "{}.{}.attentions.{}.proj_out",
        "res-last-conv": "{}.{}.resnets.{}.conv2",
        "upsampler-conv": "{}.{}.upsamplers.0.conv",
        "downsampler-conv": "{}.{}.downsamplers.0.conv",
        "cross-k": "{}.{}.attentions.{}.transformer_blocks.0.attn2.to_k",
        "cross-v": "{}.{}.attentions.{}.transformer_blocks.0.attn2.to_v"
    }