from diffusers import StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from dataclasses import dataclass
from transformers import CLIPTokenizer, CLIPTextModel
class TypeHintSDXL(StableDiffusionXLPipeline):
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModel
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
class UnetNonStepArgs:
    timestep_cond: dict
    cross_attention_kwargs: dict
    added_cond_kwargs: dict
    def __init__(self, timestep_cond: dict, cross_attention_kwargs: dict, added_cond_kwargs: dict):
        self.timestep_cond = timestep_cond
        self.cross_attention_kwargs = cross_attention_kwargs
        self.added_cond_kwargs = added_cond_kwargs
    def keys(self):
        return self._as_dict().keys()
    def _as_dict(self):
        return {
            "timestep_cond": self.timestep_cond,
            "cross_attention_kwargs": self.cross_attention_kwargs,
            "added_cond_kwargs": self.added_cond_kwargs
        }
    def __getitem__(self, key):
        return self._as_dict()[key]