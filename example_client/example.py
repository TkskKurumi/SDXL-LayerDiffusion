from PIL import Image
from modules.client import Layer, LayerDiffusionRun
from modules.utils import save_image
import numpy as np
host = "http://localhost:8001"

def example_txt2img():
    layer0 = Layer(host, prompt_expr="1girl, pink hair, pleated skirt/*worst quality*/")
    ldr = LayerDiffusionRun(host, layers=[layer0])
    im = ldr.run()
    save_image(im, "./example_client/out/txt2img.png")

def example_img2img():
    layer0 = Layer(host, prompt_expr="chibi, minigirl, cat ears, pink hair, sailor collar, blue skirt, pleated skirt/*worst quality*/")
    layer1 = Layer(host, image=Image.open(r"D:\StableDiffusion\XLAPI\V1\example_client\input\img2img_base.jpg"), mask=0.45)
    ldr = LayerDiffusionRun(host, layers=[layer0, layer1])
    im = ldr.run()
    save_image(im, "./example_client/out/img2img.png")

def example_img2img_t():
    layer0 = Layer(host, prompt_expr="1girl, white hair, silver hair, red eyes, black dress, gold trim, dark, rays of light, indoors, messy room/*worst quality, simple background*/")
    layer1 = Layer(host, image=Image.open(r"./example_client/input/龍.png"), mask=0.3)
    ldr = LayerDiffusionRun(host, layers=[layer0, layer1])
    im = ldr.run()
    save_image(im, "./example_client/out/img2img_t.png")

def example_partial_redraw():
    im_base = Image.open(r"example_client/input/no_cat_ear.png")
    im_mod  = Image.open(r"example_client/input/manual_cat_ear.png")
    diff = np.asarray(im_base).astype(np.float32) - np.asarray(im_mod).astype(np.float32)
    diff = np.sqrt(np.square(diff).sum(axis=-1))
    diff = (diff-diff.min())/(diff.max()-diff.min())
    lo_over, hi_over = -0.1, 2
    diff = diff*(hi_over-lo_over)+lo_over
    diff[diff>1] = 1
    diff[diff<0] = 0
    redraw_strength, preserve_strength = 0.75, 0
    mask = diff*(redraw_strength-preserve_strength)+preserve_strength
    mask = Image.fromarray(((1-diff)*255).astype(np.uint8))
    save_image(mask, "./example_client/out/diff_mask.png")
    
    layer0 = Layer(host, prompt_expr="cat ears, 1girl, pink hair, pleated skirt/*worst quality*/")
    layer1 = Layer(host, image=im_base, mask=mask)
    ldr = LayerDiffusionRun(host, layers=[layer0, layer1])
    im = ldr.run()
    save_image(im, "./example_client/out/partial_redraw.png")

def example_deterministic():
    np.random.seed(0)
    noise = np.random.normal(0, 1, (233, 233, 4))
    layer0 = Layer(host, prompt_expr="1girl, pink hair, pleated skirt/*worst quality*/")
    ldr = LayerDiffusionRun(host, layers=[layer0], noise=noise)
    im = ldr.run()
    save_image(im, "./example_client/out/determin_0.png")
    ldr = LayerDiffusionRun(host, layers=[layer0], noise=noise)
    im = ldr.run()
    save_image(im, "./example_client/out/determin_1.png")

example_txt2img()
example_img2img()
example_img2img_t()
example_partial_redraw()
example_deterministic()