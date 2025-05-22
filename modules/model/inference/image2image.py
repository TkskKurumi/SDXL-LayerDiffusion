from ..data_classes import *
from tqdm import tqdm
import torch, math
import numpy as np
from PIL import Image
from ...globals import get_size, get_device, load_cfg
from ...prompt_expr import PromptExpr
from .process_long_prompt import slice_one_prompt
from .encode_prompt import encode_2prompt, encode_2prompt_model
from ..model import goc_base
from .single_step import *
from diffusers.image_processor import VaeImageProcessor
from ...utils.candy import _debug

# copy from pipeline_stable_diffusion_xl_img2img.py
def retrieve_encoder_output(encoder_output, generator=None, sample_mode="sample"):
    if (hasattr(encoder_output, "latent_dist") and sample_mode=="sample"):
        return encoder_output.latent_dist.sample(generator)
    elif (hasattr(encoder_output, "latent_dist") and sample_mode=="argmax"):
        return encoder_output.latent_dist.mode()
    elif (hasattr(encoder_output, "latents")):
        return encoder_output.latents
    else:
        raise AttributeError("Cannot access latetns of encoder output")

# copy from pipeline_stable_diffusion_xl_img2img.py prepare_latents
def vae_encode(img_proc, vae, image: Image.Image, dtype):
    cfg = vae.config
    mn, std = None, None
    if (getattr(cfg, "latents_mean", None) is not None):
        mn = cfg.latents_mean
    if (getattr(cfg, "latents_std", None) is not None):
        std = cfg.latents_std

    img_tensor = img_proc.preprocess(image).to(get_device())
    if img_tensor.shape[1] == 4:
        print("DEBUG: img_tensor.shape[1] == 4")
        init_latents = image
    else:
        
        if cfg.force_upcast:
            img_tensor = img_tensor.float()
            vae.to(dtype=torch.float32)
        init_latents = retrieve_encoder_output(vae.encode(img_tensor))
        if (cfg.force_upcast):
            vae.to(dtype)
    init_latents = init_latents.to(dtype)

    if ((mn is not None) and (std is not None)):
        init_latents = (init_latents-mn)*cfg.scaling_factor / std
    else:
        init_latents = cfg.scaling_factor * init_latents

    return init_latents
@torch.no_grad()
def image2image_skip(model: TypeHintSDXL, prompt_expr: str, guidance: float, image: Image.Image, n_steps=50, skip=0.3):
    if (0<=skip and skip<1):
        skip = round(n_steps*skip)
    pe = PromptExpr(prompt_expr)
    pe.with_lora.apply(True)
    pros = slice_one_prompt(pe.pro, [model.tokenizer, model.tokenizer_2])
    
    negs = slice_one_prompt(pe.neg, [model.tokenizer, model.tokenizer_2])
    print(pros, negs)
    size = get_size(image.width, image.height)
    image = image.resize(size, Image.Resampling.LANCZOS)
    width, height = size

    embds = []
    for pro in pros:
        embd, pro_pooled = encode_2prompt(pro, model.tokenizer, model.text_encoder, pro, model.tokenizer_2, model.text_encoder_2)
        dtype = embd.dtype
        embds.append((guidance, embd, pro_pooled))
    for neg in negs:
        embd, neg_pooled = encode_2prompt(neg, model.tokenizer, model.text_encoder, neg, model.tokenizer_2, model.text_encoder_2)
        embds.append((-1, embd, neg_pooled))
    
    model.scheduler.set_timesteps(n_steps, device=get_device())
    timesteps_all = model.scheduler.timesteps
    timesteps = timesteps_all[skip:]
    timestep0 = timesteps_all[skip]
    
    img_encoded = vae_encode(model.image_processor, model.vae, image, dtype)
    # latent_rand = model.prepare_latents(1, model.unet.config.in_channels, height, width, dtype, get_device(), None)
    noise = torch.randn_like(img_encoded)
    latents = model.scheduler.add_noise(img_encoded, noise, timesteps[:1])
    # print("DEBUG: dtype", dtype, "img_encoded", str_mean_std(img_encoded), "latents", str_mean_std(latents))
    
    ex_step_kwa = model.prepare_extra_step_kwargs(None, 0)
    ts_cond = None
    if (model.unet.config.time_cond_proj_dim is not None):
        g_scale_t = torch.tensor(guidance-1).repeat(1)
        ts_cond = model.get_guidance_scale_embedding(size, (0, 0), size, latents.dtype)
    te_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids(size, (0, 0), size, latents.dtype, te_dim).to(get_device())
    non_step_args = UnetNonStepArgs(timestep_cond=ts_cond, cross_attention_kwargs={},
                                    added_cond_kwargs={
                                        "text_embeds": pro_pooled,
                                        "time_ids": add_time_ids
                                    })
    torch.cuda.empty_cache()
    ts_enumerate = list(enumerate(timesteps))
    for i, t in tqdm(ts_enumerate):
        latents_in = model.scheduler.scale_model_input(latents, t)
        noise_pred = step_pred_cfg_embd(model, latents_in, embds, t, non_step_args)
        latents = model.scheduler.step(noise_pred, t, latents, return_dict=False, **ex_step_kwa)[0]
    torch.cuda.empty_cache()
    image = decode_latents(model, latents)
    return image

def foo(n_steps, alpha=0.7):
    def test(step_alpha):
        meow = 0
        for i in range(n_steps):
            meow = meow*i/(i+1)
            meow = meow*(1-step_alpha) + step_alpha
        return meow
    l = 0
    r = 1
    while (l+1e-8 < r):
        m = (l+r)/2
        if (test(m) < alpha):
            l = m
        else:
            r = m
    return r

def calc_sched_image_blend(i_alpha_steps):
    ret = 0
    for i, a in enumerate(i_alpha_steps):
        ret = ret*i/(i+1)
        ret = ret*(1-a) + a
    return ret

# this method is good
# should work out how to implement this for i_alpha: ndarray for mask image
def schedule_image_blend(n_steps, i_alpha):
    i_alpha_step_base = 1-np.arange(n_steps).astype(np.float32)/(n_steps-1)
    def test(x):
        if (x<0):
            i_alpha_step = i_alpha_step_base*(1+x)
        else:
            i_alpha_step = i_alpha_step_base*(1-x) + x
        return calc_sched_image_blend(i_alpha_step), i_alpha_step
    l = -1
    r = 1
    while (l+1e-8<r):
        m = (l+r)/2
        a, ret = test(m)
        if (a<i_alpha):
            l = m
        else:
            r = m
    # _debug("m", m, "a", a, "ret", ret)
    return ret

@torch.no_grad()
def image2image_blend(model: TypeHintSDXL, prompt_expr: str, guidance: float, image: Image.Image, n_steps=50, alpha=0.7):
    pe = PromptExpr(prompt_expr)
    pe.with_lora.apply(True)
    # --- encode prompts
    pros = slice_one_prompt(pe.pro, [model.tokenizer, model.tokenizer_2])
    negs = slice_one_prompt(pe.neg, [model.tokenizer, model.tokenizer_2])
    _debug("prompt pro/neg", pros, "/", negs)
    embds = []
    for pro in pros:
        embd, pro_pooled = encode_2prompt_model(model, pro, pro)
        dtype = embd.dtype
        embds.append((guidance, embd, pro_pooled))
    for neg in negs:
        embd, neg_pooled = encode_2prompt_model(model, neg, neg)
        embds.append((-1, embd, neg_pooled))
    
    # --- timesteps
    model.scheduler.set_timesteps(n_steps, device=get_device())
    timesteps = model.scheduler.timesteps

    # --- encode image
    size = get_size(image.width, image.height)
    image = image.resize(size, Image.Resampling.LANCZOS)
    img_encoded = vae_encode(model.image_processor, model.vae, image, dtype)
    noise = torch.randn_like(img_encoded)
    latents = noise*model.scheduler.init_noise_sigma
    _debug("size:", size, "latents", str_mean_std(latents))

    # --- misc args
    te_dim = model.text_encoder_2.config.projection_dim
    ex_step_kwa = model.prepare_extra_step_kwargs(None, 0)
    ts_cond = None
    if (model.unet.config.time_cond_proj_dim is not None):
        ts_cond = model._get_add_time_ids(size, (0, 0), size, dtype)
    add_time_ids = model._get_add_time_ids(size, (0, 0), size, dtype, te_dim).to(get_device())
    non_step_args = UnetNonStepArgs(timestep_cond=ts_cond,
                                    cross_attention_kwargs={},
                                    added_cond_kwargs={
                                        "text_embeds": pro_pooled,
                                        "time_ids": add_time_ids
                                    })
    
    # l_alpha = np.array(list(range(n_steps)), np.float64)
    # l_alpha = np.exp2(-np.arange(n_steps))
    # l_alpha = np.ones((n_steps, ))
    # _debug("l_alpha", l_alpha)
    # l_alpha = np.exp(l_alpha/l_alpha.sum()*math.log(1-alpha))
    # l_alpha = (1-foo(n_steps, alpha))*np.ones((n_steps, ))
    l_alpha = 1-schedule_image_blend(n_steps, alpha)
    # _debug("l_alpha", l_alpha)


    # do denoise
    ts_enumerate = list(enumerate(timesteps))
    for i, t in tqdm(ts_enumerate):
        latents_in = model.scheduler.scale_model_input(latents, t)
        noise_pred = step_pred_cfg_embd(model, latents_in, embds, t, non_step_args)
        latents = model.scheduler.step(noise_pred, t, latents, return_dict=False, **ex_step_kwa)[0]

        latents_step_alpha = l_alpha[i]
        if (i+1 < len(timesteps)):
            t_next = timesteps[i+1]
            latent_correct = model.scheduler.add_noise(img_encoded, noise, torch.tensor([t_next]))
        else:
            latent_correct = img_encoded
        _latents = latents
        # print("DEBUG: latents", str_mean_std(latents), "latents_c", str_mean_std(latent_correct), "img_encoded", str_mean_std(img_encoded))
        latents = latents*latents_step_alpha + latent_correct*(1-latents_step_alpha)
        _debug("_latents", str_mean_std(_latents), "l_cor", str_mean_std(latent_correct), "latents", str_mean_std(latents))

    image = decode_latents(model, latents)
    return image

    



    

if (__name__=="__main__"):
    load_cfg("./example.yml")
    # print(goc_base().scheduler)
    im_base = Image.open(r"D:\StableDiffusion\XLAPI\V1\enmiaohaqisese.png").convert("RGB")
    # im_base = Image.fromarray(255-np.asarray(im_base))
    # im = image2image_skip(goc_base(),
    #                  "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, /*worst quality, lowres*/",
    #                  7,
    #                  im,
    #                  50,
    #                  0.7)
    # im.save("1.png")
    # pro = "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, side slit, heterochromia, red eye, blue eye, sailor collar, petite, /*worst quality, lowres*/"
    # pro = "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, open clothes, looking to side, averting eyes, bra, cleavage, blush, shy, /*worst quality, lowres, blurry, jpeg artifacts, mosaic, bad anatomy*/"
    # pro = "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, angry, furrowed brow, open mouth, paw pose, shaded face, disgust, wet, indoor, sailor collar, serafuku, pleated skirt, desks, classroom, light rays, /*worst quality, lowres, blurry, jpeg artifacts, mosaic, bad anatomy, simple background*/"
    # pro = "absurdres, masterpiece, highres, best quality, gray hair, animal ears fluff, wolf ears, thighs, crown, jewelry, luxury, black dress, gold trim, angry, furrowed brow, open mouth, paw pose, shaded face, disgust, indoor, light rays, /*worst quality, lowres, blurry, jpeg artifacts, mosaic, bad anatomy, simple background, multiple views*/"
    # pro = "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, black chinadress, side slit, gold trim, gold print, dragon print, side-tie knots, thighs, /*worst quality, lowres, blurry, jpeg artifacts, mosaic, bad anatomy*/"
    # pro = "<lora:250407_D05_14_wd3-s4k:1>solo, 1girl, togawa sakiko, sailor collar, beachside, sun, lens flare, coconut tree, micro bikini swimsuit, front-tie bikini bra, side-tie bikini panties, side-tie knots, water, wading, wet, sweat, from below, ass visible through thighs, cameltoe, dramatic angle, smile, arms up, armpits, holding ball, blue sky, white clouds, boat, navel, stomach, groin tendon, medium breasts, sideboob, o-ring swimsuit, ocean, splash, black swimsuit, black bikini, petite, /*lowres, blurry, bad anatomy*/"
    pro = "pink hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, sailor collar, beachside, sun, lens flare, coconut tree, micro bikini swimsuit, front-tie bikini bra, side-tie bikini panties, side-tie knots, water, wading, wet, sweat, from below, ass visible through thighs, cameltoe, dramatic angle, smile, arms up, armpits, holding ball, blue sky, white clouds, boat, navel, stomach, groin tendon, medium breasts, sideboob, o-ring swimsuit, ocean, splash, black swimsuit, black bikini, petite, /*lowres, blurry, bad anatomy*/"

    from .text2image import text2image

    # im0 = text2image(goc_base(), pro, guidance=5, n_steps=25)
    # im1 = image2image_blend(goc_base(), pro, guidance=5, image=im_base, n_steps=25, alpha=0.001)
    # im0.save("./img0.png")
    # im1.save("./img1.png")
    # quit()
    # img = text2image(goc_base(), pro, 7, 30)
    # img.save("0.png")

    # im = image2image_skip(goc_base(),
    #                       "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, /*worst quality, lowres*/",
    #                       7,
    #                       im_base,
    #                       50,
    #                       0.5)
    # im.save("1.png")

    
    # im = image2image_blend(goc_base(),
    #                        "absurdres, masterpiece, highres, best quality, pink hair, cat ears, animal ears fluff, /*worst quality, lowres*/",
    #                        7,
    #                        im_base,
    #                        30,
    #                        0.5)
    # im.save("2.png")
    mn = 0.288
    mx = 0.330
    n = 5
    ls = [mn+(mx-mn)/(n-1)*i for i in range(n)]
    print(ls)
    for i in ls:
        im = image2image_blend(goc_base(),
                               pro,
                               7,
                               im_base,
                               30,
                               i)
        pth = "./%.4f.png"%i
        im.save(pth)
        print(pth)
    print(ls)