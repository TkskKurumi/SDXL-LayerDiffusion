from typing import Any
from diffusers import StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List
from ..data_classes import TypeHintSDXL, UnetNonStepArgs
from math import sqrt
from ...utils.debug_tensor import *
from PIL import Image

def decode_latents(model: TypeHintSDXL, latents) -> Image.Image:
    needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
    if (needs_upcasting):
        model.upcast_vae()
        latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
    vaecfg = model.vae.config
    has_mean = hasattr(vaecfg, "latents_mean") and vaecfg.latents_mean is not None
    has_std  = hasattr(vaecfg, "latents_std") and vaecfg.latents_std is not None
    if has_mean and has_std:
        mn = torch.tensor(vaecfg.latents_mean).view(1, 4, 1, 1).to(latents.device)
        st = torch.tensor(vaecfg.latents_std).view(1, 4, 1, 1).to(latents.device)
        latents = latents * st / vaecfg.scaling_factor + mn
    else:
        latents = latents / vaecfg.scaling_factor
    # print("latents for decode", str_mean_std(latents))
    image = model.vae.decode(latents, return_dict=False)[0]
    image = model.image_processor.postprocess(image, output_type="pil")[0]
    return image

def step_pred_single_embd(model: TypeHintSDXL, latent: torch.tensor, embd: torch.tensor, embd_pooled: torch.tensor, t: int, non_step_args: UnetNonStepArgs):
    # latent = model.scheduler.scale_model_input(latent, t)
    # non_step_args.added_cond_kwargs["text_embeds"] = embd_pooled
    noise_pred = model.unet(latent, t,
                            encoder_hidden_states=embd,
                            return_dict=False,
                            **non_step_args)[0]
    return noise_pred
def mix_cfg_noises(noises: List[Tuple[float, torch.tensor]]):
    pro_noises = []
    neg_noises = []
    pro_noise_sum = 0
    neg_noise_sum = 0
    pro_ws = []
    neg_ws = []
    for w, noise in noises:
        if (w>0):
            pro_noises.append((w, noise))
            pro_ws.append(w)
            pro_noise_sum = pro_noise_sum+noise*w
        else:
            neg_noises.append((w, noise))
            neg_ws.append(w)
            neg_noise_sum = neg_noise_sum-noise*w
    pro0 = next(iter(pro_noises))[-1]
    neg0 = next(iter(neg_noises))[-1]
    std_pro0 = torch.std(pro0)
    std_neg0 = torch.std(neg0)
    std_pros = torch.std(pro_noise_sum)
    std_negs = torch.std(neg_noise_sum)

    guidance = sqrt(sum(i*i for i in pro_ws))
    
    
    # print("DEBUG: pros", str_mean_std(pro_noise_sum))
    # print("DEBUG: pro0", str_mean_std(next(iter(pro_noises))[-1]))
    # print("DEBUG: negs", str_mean_std(neg_noise_sum))
    # print("DEBUG: neg0", str_mean_std(next(iter(neg_noises))[-1]))
    # print("DEBUG: guidance", pro_ws, guidance)

    ret0 = pro_noise_sum/std_pros*std_pro0*(guidance+1) - neg_noise_sum/std_negs*std_neg0*guidance
    std_ret0 = torch.std(ret0)
    ret = ret0/std_ret0*std_pro0
    # print("DEBUG: mix noise pred", "pro0", str_mean_std(pro0), "ret", str_mean_std(ret))
    return ret

def step_pred_cfg_embd(model: TypeHintSDXL, latent: torch.tensor, embds: List[Tuple[float, torch.tensor, torch.tensor]], t: int, non_step_args: UnetNonStepArgs):
    noises = []
    for w, embd, embd_pooled in embds:
        noise_pred = step_pred_single_embd(model, latent, embd, embd_pooled, t, non_step_args)
        noises.append((w, noise_pred))
    return mix_cfg_noises(noises)

