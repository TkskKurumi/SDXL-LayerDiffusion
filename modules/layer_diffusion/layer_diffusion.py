from __future__ import annotations
from ..model.inference.single_step import *
from ..prompt_expr import PromptExpr
from ..model.data_classes import TypeHintSDXL, UnetNonStepArgs
from ..model.inference.single_step import *
from ..model.inference.encode_prompt import encode_2prompt_model
from PIL import Image
import numpy as np
from tqdm import tqdm
from .linear_sched_blend_2d import LinearSchedBlend
from ..prompt_expr.process_long_prompt import slice_one_prompt
from ..utils.candy import _debug
from ..utils.misc import curr_memory_str
from ..globals import get_device, get_size, get_cfg, load_cfg
from typing import Union
from types import NoneType
from ..model import goc_base, MODEL_LOCK
from modules.utils.img2noise import img2noise
def retrieve_encoder_output(encoder_output, generator=None, sample_mode="sample"):
    if (hasattr(encoder_output, "latent_dist") and sample_mode=="sample"):
        return encoder_output.latent_dist.sample(generator)
    elif (hasattr(encoder_output, "latent_dist") and sample_mode=="argmax"):
        return encoder_output.latent_dist.mode()
    elif (hasattr(encoder_output, "latents")):
        return encoder_output.latents
    else:
        raise AttributeError("Cannot access latetns of encoder output")
def vae_encode(img_proc, vae, image: Image.Image, dtype):
    cfg = vae.config
    mn, std = None, None
    if (getattr(cfg, "latents_mean", None) is not None):
        mn = cfg.latents_mean
    if (getattr(cfg, "latents_std", None) is not None):
        std = cfg.latents_std

    img_tensor = img_proc.preprocess(image).to(get_device())
    if img_tensor.shape[1] == 4:
        _debug("img_tensor.shape[1] == 4")
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

class Layer:
    def __init__(self, prompt_expr=None, image: Union[NoneType, Image.Image]=None, mask=None, mask_scale=1):
        if (prompt_expr is not None):
            self.pe = PromptExpr(prompt_expr)
            self.with_lora = self.pe.with_lora
        else:
            self.pe = None
            self.with_lora = None
        self.image = image
        self.mask = mask
        self.mask_scale = mask_scale
        
        self._a3d_cache = {}
        self._mask_cache = {}
        
        self.embds = []
        self.unet_nonstep_args = None
        self.img_encoded = None
    @property
    def is_image(self):
        return self.image is not None
    @property
    def is_text(self):
        return self.pe is not None
    def get_a3d(self, h, w, n_steps, mask_orig=None):
        key = (h, w, n_steps)
        if (key in self._a3d_cache):
            return self._a3d_cache[key]
        if (mask_orig is None):
            raise KeyError(key)
        mask_orig = mask_orig.reshape((h, w, 1))
        _debug("mask_orig, max", mask_orig.max(axis=0).max(axis=0), "min", mask_orig.min(axis=0).min(axis=0))
        a3d = LinearSchedBlend(mask_orig, n_steps).a3d
        _debug("a3d, max", a3d.max(axis=0).max(axis=0), "min", a3d.min(axis=0).min(axis=0))
        self._a3d_cache[key] = a3d
        return a3d
    def get_mask_like(self, latents: torch.Tensor):
        h, w = latents.shape[-2:]
        key = (h, w)
        if (key in self._mask_cache):
            return self._mask_cache[key]

        if (isinstance(self.mask, Image.Image)):
            alpha_mask_i = self.mask.resize((w, h), Image.Resampling.LANCZOS).convert("L")
            alpha_mask_n = np.asarray(alpha_mask_i).astype(np.float32)/255
            alpha_mask_t = torch.from_numpy(alpha_mask_n).to(device=latents.device, dtype=latents.dtype)
            ret = alpha_mask_t
        elif (self.mask is None):
            ret = torch.ones((h, w), device=latents.device, dtype=latents.dtype)
        else:
            alpha_mask_n = np.ones((h, w), np.float32)*self.mask
            alpha_mask_t = torch.from_numpy(alpha_mask_n).to(device=latents.device, dtype=latents.dtype)
            ret = alpha_mask_t
        ret = ret*self.mask_scale
        self._mask_cache[key] = ret
        return ret
    def get_step_mask_like(self, latents, step_i, step_n):
        if (self.is_image):
            h, w = latents.shape[-2:]
            orig = self.get_mask_like(latents)
            if (isinstance(orig, torch.Tensor)):
                orig = orig.detach().cpu().numpy()
            return torch.from_numpy(self.get_a3d(h, w, step_n, orig)[:, :, step_i]).to(dtype=latents.dtype, device=latents.device)
        else:
            return self.get_mask_like(latents)
    
    def _prepare_inference_t(self, run: LayerDiffusionRun):
        model = run.model
        pe = self.pe
        # need apply to text_encoders for creating embeddings
        pe.with_lora.apply(False)

        # --- encode prompts
        pros = slice_one_prompt(pe.pro, [model.tokenizer, model.tokenizer_2])
        negs = slice_one_prompt(pe.neg, [model.tokenizer, model.tokenizer_2])
        _debug("prompt pro/neg", pros, "/", negs)
        embds = []
        for pro in pros:
            embd, pro_pooled = encode_2prompt_model(model, pro, pro)
            dtype = embd.dtype
            embds.append((run.guidance, embd, pro_pooled))
        for neg in negs:
            embd, neg_pooled = encode_2prompt_model(model, neg, neg)
            embds.append((-1, embd, neg_pooled))
        self.embds = embds
        run.dtype = dtype

        # --- misc args
        te_dim = model.text_encoder_2.config.projection_dim
        ts_cond = None
        if (model.unet.config.time_cond_proj_dim is not None):
            ts_cond = model._get_add_time_ids(run.size, (0, 0), run.size, dtype)
        add_time_ids = model._get_add_time_ids(run.size, (0, 0), run.size, dtype, te_dim).to(get_device())
        self.unet_nonstep_args = UnetNonStepArgs(timestep_cond=ts_cond,
                                                 cross_attention_kwargs={},
                                                 added_cond_kwargs={
                                                     "text_embeds": pro_pooled,
                                                     "time_ids": add_time_ids
                                                 })

    def _prepare_inference_i(self, run: LayerDiffusionRun):
        model = run.model
        im = self.image.resize(run.size, Image.Resampling.LANCZOS).convert("RGB")
        self.img_encoded = vae_encode(run.model.image_processor, model.vae, im, run.dtype)
        

    def prepare_inference(self, run: LayerDiffusionRun):
        if (self.is_text):
            return self._prepare_inference_t(run)
        else:
            return self._prepare_inference_i(run)

    def _step_t(self, run: LayerDiffusionRun, latents0, latents_in, ts_idx, ts, ts_n):
        _debug_step_index = False
        self.pe.with_lora.apply(False)
        
        noise_pred = step_pred_cfg_embd(run.model, latents_in, self.embds, ts, self.unet_nonstep_args)
        if (_debug_step_index):
            run.model.scheduler._init_step_index(ts)
            _debug("ts_idx", ts_idx, "sched init idx", run.model.scheduler._step_index)
        run.model.scheduler._step_index = ts_idx
        
        idx_before_step = run.model.scheduler._step_index
        latents = run.model.scheduler.step(noise_pred, ts, latents0, return_dict=False, **run.ex_step_kwa)[0]
        if (_debug_step_index):
            _debug("step_index", idx_before_step, "->", run.model.scheduler._step_index)
        return latents
    def _step_i(self, run: LayerDiffusionRun, latents0, latents_in, ts_idx, ts, ts_n):
        if (ts_idx+1 < ts_n):
            next_ts = run.model.scheduler.timesteps[ts_idx+1]
            latents = run.model.scheduler.add_noise(self.img_encoded, run.noise, torch.tensor([next_ts]))
        else:
            latents = self.img_encoded
        return latents
    def step(self, run, latents0, latents_in, ts_idx, ts, ts_n):
        if (self.is_text):
            return self._step_t(run, latents0, latents_in, ts_idx, ts, ts_n)
        else:
            return self._step_i(run, latents0, latents_in, ts_idx, ts, ts_n)
class LayerDiffusionRun:
    def __init__(self, model: TypeHintSDXL, layers: List[Layer], guidance=5, n_steps=25, noise=None):
        self.model  = model
        self.layers = layers
        self.guidance = guidance
        self.n_steps = n_steps
        self.noise = noise
    @torch.no_grad()
    def run(self):
        with MODEL_LOCK:
            model = self.model

            self.guidance = self.guidance
            self.model.scheduler.set_timesteps(self.n_steps)
            self.ex_step_kwa = self.model.prepare_extra_step_kwargs(None, 0)
            self.dtype = torch.float16
            self.size = (210, 297)
            for i in self.layers:
                if (i.is_image):
                    self.size = i.image.size
            self.size = get_size(*self.size)
            
            for l in self.layers:
                l.prepare_inference(self)
            width, height = self.size
            n_ch = self.model.unet.config.in_channels
            noise_shape = (1, n_ch, height//self.model.vae_scale_factor, width//self.model.vae_scale_factor)
            if (self.noise is None):
                noise = torch.randn(noise_shape).to(dtype=self.dtype, device=get_device())
            elif (isinstance(self.noise, Image.Image)):
                noise = torch.from_numpy(img2noise(self.noise, noise_shape)).to(dtype=self.dtype, device=get_device())
            else:
                raise TypeError("Unknown noise type %s"%type(noise))
            latents = noise*self.model.scheduler.init_noise_sigma
            
            self.noise = noise

            lh, lw = latents.shape[-2:]

            ts = model.scheduler.timesteps
            ts_enum = list(enumerate(ts))
            for idx, t in tqdm(ts_enum):
                weights = [torch.zeros((lh, lw), dtype=self.dtype, device=get_device()) for i in self.layers]
                for ldx, l in enumerate(self.layers):
                    if (ldx):
                        layer_alpha = l.get_step_mask_like(latents, idx, self.n_steps)
                        # _debug("layer", ldx, "mask", str_mean_std(layer_alpha))
                        for jdx in range(ldx):
                            if (weights[jdx].device != layer_alpha.device):
                                _debug("w", str_mean_std(weights[jdx]), "la", str_mean_std(layer_alpha))
                            weights[jdx] *= 1-layer_alpha
                        weights[ldx] = layer_alpha
                    else:
                        weights[ldx] = torch.ones((lh, lw), dtype=self.dtype, device=get_device())
                
                latents0 = latents
                latents_in = model.scheduler.scale_model_input(latents, t)
                latents1 = torch.zeros_like(latents0)
                for ldx, l in enumerate(self.layers):
                    weight = weights[ldx]
                    if (weight.sum() < 0.1):
                        continue
                    latents_pred = l.step(self, latents0, latents_in, idx, t, self.n_steps)
                    latents1 += latents_pred*weight
                latents = latents1
            torch.cuda.empty_cache()
            image = decode_latents(model, latents)
            _debug(curr_memory_str())
        return image

        


if (__name__=="__main__"):
    load_cfg("./example.yml")
    goc_base()

    img_text = Image.open(r"D:\StableDiffusion\XLAPI\V1\0.7000.png")

    negative = "best quality, absurdres, masterpiece, highres, /*lowres, blurry, worst quality, bad anatomy, unfinished, sketch, comics*/"
    qianqian = "pink hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, sailor collar"
    caicai = "black hair, cat ears, red eyes, animal ears fluff, sailor collar, low twintails, short twintails, short hair, petite"
    blue_qq = "blue hair, cat ears, animal ears fluff, heterochromia, red eye, blue eye, sailor collar"
    tgw = "<lora:250407_D05_14_wd5-epo5:1>togawa sakiko, "
    tgw0 = "togawa sakiko"
    mtm = "<lora:250407_D05_14_wd5-epo5:1>wakaba mutsumi"
    concept = "solo, 1girl, beachside, sun, lens flare, coconut tree, micro bikini swimsuit, front-tie bikini bra, side-tie bikini panties, side-tie knots, water, wading, wet, sweat, from below, ass visible through thighs, cameltoe, smile, arms up, armpits, holding ball, blue sky, white clouds, boat, navel, stomach, medium breasts, sideboob, o-ring swimsuit, ocean, black bikini, petite"
    concept1 = "indoors, classroom, desks, window, rays of light, light rays"
    concept2 = "kiss, 2girls, yuri, face to face, symmetrical docking, from side, heart"
    mask_lr = r"D:\StableDiffusion\XLAPI\V1\mask_left_right.png"

    pr1 = ", ".join([tgw, concept2, negative])
    pr2 = ", ".join([qianqian, concept2, negative])
    l0 = Layer(pr1)
    l1 = Layer(pr2, mask=Image.open(mask_lr))
    l2 = Layer(image=img_text, mask=img_text, mask_scale=0.75)
    layer_run = LayerDiffusionRun(goc_base(), [l0, l1], n_steps=35, guidance=7)
    im = layer_run.run()
    im.save("./5.png")
    # from ..model.inference.text2image import text2image
    # text2image(goc_base(), pr1, guidance=5, n_steps=30).save("./0.png")
    