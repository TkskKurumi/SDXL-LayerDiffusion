from diffusers import StableDiffusionXLPipeline
from .. import globals
import torch
from typing import Union
from types import NoneType
from threading import Lock
from accelerate import cpu_offload
MODEL_LOCK = Lock()
BASE_MODEL: Union[NoneType, StableDiffusionXLPipeline] = None
def goc_base():
    global BASE_MODEL, MODEL_LOCK
    if (BASE_MODEL is not None):
        return BASE_MODEL
    with MODEL_LOCK:
        model_pth = globals.get_cfg("model", None)
        if (model_pth is None):
            raise globals.CFGError("model not specified")
        pipeline = StableDiffusionXLPipeline.from_single_file(model_pth,
                                                              torch_dtype=torch.float16,
                                                              variant="fp16",
                                                              use_safetensors=True, add_watermark=False)
        pipeline.enable_xformers_memory_efficient_attention()
        device = globals.get_cfg("device", "cuda:1")
        pipeline = pipeline.to(device)
        pipeline = pipeline.to(torch.float16)
        enable_offload(pipeline)
        BASE_MODEL = pipeline
        return BASE_MODEL
def enable_offload(model=None):
    if (model is None):
        model = goc_base()
    for submodel in model.vae, model.text_encoder, model.text_encoder_2:
        cpu_offload(submodel, globals.get_device())
def disable_offload():
    model = goc_base()
    return StableDiffusionXLPipeline._optionally_disable_offloading(model)