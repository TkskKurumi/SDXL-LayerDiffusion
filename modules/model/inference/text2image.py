from ..data_classes import *
from tqdm import tqdm
from ...prompt_expr import PromptExpr
from .process_long_prompt import slice_one_prompt
from .encode_prompt import encode_2prompt
from ...globals import get_device, get_size, load_cfg
import torch
from .single_step import step_pred_cfg_embd, decode_latents
from ..model import goc_base
from ...utils.debug_tensor import str_mean_std
from ...utils.candy import _debug
# def decode_latents(model: TypeHintSDXL, latents):
#     needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
#     if (needs_upcasting):
#         model.upcast_vae()
#         latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
#     vaecfg = model.vae.config
#     has_mean = hasattr(vaecfg, "latents_mean") and vaecfg.latents_mean is not None
#     has_std  = hasattr(vaecfg, "latents_std") and vaecfg.latents_std is not None
#     if has_mean and has_std:
#         mn = torch.tensor(vaecfg.latents_mean).view(1, 4, 1, 1).to(latents.device)
#         st = torch.tensor(vaecfg.latents_std).view(1, 4, 1, 1).to(latents.device)
#         latents = latents * st / vaecfg.scaling_factor + mn
#     else:
#         latents = latents / vaecfg.scaling_factor
#     # print("latents for decode", str_mean_std(latents))
#     image = model.vae.decode(latents, return_dict=False)[0]
#     image = model.image_processor.postprocess(image, output_type="pil")[0]
#     return image
@torch.no_grad()
def text2image(model: TypeHintSDXL, prompt_expr: str, guidance: float, n_steps=50):
    pe = PromptExpr(prompt_expr)
    with pe.with_lora:


        pros = slice_one_prompt(pe.pro, [model.tokenizer, model.tokenizer_2])
        negs = slice_one_prompt(pe.neg, [model.tokenizer, model.tokenizer_2])
        _debug("text2image prompts", pros, negs)
        size = get_size(pe.width, pe.height)
        width, height = size

        embds = []
        pro_embds = []
        for pro in pros:
            embd, pro_pooled = encode_2prompt(pro, model.tokenizer, model.text_encoder, pro, model.tokenizer_2, model.text_encoder_2)
            pro_embds.append(embd)
            embds.append((guidance, embd, pro_pooled))
        neg_embds = []
        for neg in negs:
            embd, neg_pooled = encode_2prompt(neg, model.tokenizer, model.text_encoder, neg, model.tokenizer_2, model.text_encoder_2)
            neg_embds.append(embd)
            embds.append((-1, embd, neg_pooled))
        # print("DEBUG", embds)
        dtype = next(iter(pro_embds)).dtype

        model.scheduler.set_timesteps(n_steps, device=get_device())
        timesteps = model.scheduler.timesteps
        te_dim = model.text_encoder_2.config.projection_dim
        latents = model.prepare_latents(1, model.unet.config.in_channels, height, width, dtype, get_device(), None, None)
        extra_step_kwargs = model.prepare_extra_step_kwargs(None, 0)
        ts_cond = None
        if (model.unet.config.time_cond_proj_dim is not None):
            g_scale_t = torch.tensor(guidance-1).repeat(1)
            ts_cond = model.get_guidance_scale_embedding(g_scale_t, embedding_dim=model.unet.config.time_cond_proj_dim).to(get_device(), latents.dtype)
        add_time_ids = model._get_add_time_ids(size, (0, 0), size, latents.dtype, te_dim).to(get_device())
        
        non_step_args = UnetNonStepArgs(timestep_cond=ts_cond,
                                        cross_attention_kwargs={},
                                        added_cond_kwargs={
                                            "text_embeds": pro_pooled,
                                            'time_ids': add_time_ids
                                        })
        
        ts_enumerate = list(enumerate(timesteps))
        print("DEBUG: txt2img init latents", str_mean_std(latents))
        _debug("txt2img init latents", str_mean_std(latents))
        for i, t in tqdm(ts_enumerate):
            latents_in = model.scheduler.scale_model_input(latents, t)
            
            noise_pred = step_pred_cfg_embd(model, latents_in, embds, t, non_step_args)
            latents = model.scheduler.step(noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]
            if (False):
                print("DEBUG: step noise_pred", t, str_mean_std(noise_pred), "latents", str_mean_std(latents), "latents_inp", str_mean_std(latents_in),
                    "ts_cond", "None" if ts_cond is None else str_mean_std(ts_cond))
        image = decode_latents(model, latents)    
    return image

if (__name__=="__main__"):
    load_cfg("./example.yml")
    base_model = goc_base()
    # img = base_model(prompt="1girl, pink hair, cat ears", negative_prompt="lowres", num_inference_steps=20, guidance_scale=5, width=512, height=512).images[0]
    # img.save("0.png")
    # img = text2image(base_model, "1girl, night, best quality, masterpiece, highres, absurdres, rain, raindrop, wet, black jacket, empty eyes, from side, against wall, pink hair, cat ears, heterochromia, red eyes, blue eyes, hood up, cityscape, street, alley, dark, open clothes, black bra, black panties, mini shorts, thick thighs, tears, expressionless, smoking, torn clothes, torn thighhighs, torn legwear, black thighhighs, facing down, shaded face/*lowres, blurry, bad anatomy*/", 5, n_steps=20)
    # img = text2image(base_model, "1girl, pink hair, cat ears, animal ears fluff, heterochromia, red eyes, blue eyes, sailor collar, beachside, sun, lens flare, coconut tree, micro bikini swimsuit, front-tie bikini bra, side-tie bikini panties, side-tie knots, water, wading, wet, sweat, from below, ass visible through thighs, cameltoe, dramatic angle, smile, arms up, armpits, holding ball, blue sky, white clouds, boat, navel, stomach, groin tendon, medium breasts, sideboob, o-ring swimsuit, wide hips, thick thighs, ocean, seabird, splash, running, black swimsuit, black bikini, petite, sunshine, rays of light, /*lowres, blurry, bad anatomy*/", 5, n_steps=20)
    # img.save("1.png")
    img = text2image(base_model, "<lora:250407_D05_14_wd3-s4k:1>togawa sakiko, sailor collar, beachside, sun, lens flare, coconut tree, micro bikini swimsuit, front-tie bikini bra, side-tie bikini panties, side-tie knots, water, wading, wet, sweat, from below, ass visible through thighs, cameltoe, dramatic angle, smile, arms up, armpits, holding ball, blue sky, white clouds, boat, navel, stomach, groin tendon, medium breasts, sideboob, o-ring swimsuit, ocean, splash, black swimsuit, black bikini, petite, /*lowres, blurry, bad anatomy*/", 5, n_steps=25)
    img.save("2.png")