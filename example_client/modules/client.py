from io import BytesIO
import requests
from typing import List, Dict
from PIL import Image
from .utils import *
def noise2img(noise: np.ndarray):
    n = (noise-noise.mean())/(noise.max()-noise.min())
    return Image.fromarray((n*255).astype(np.uint8))
class Layer:
    def __init__(self, host, prompt_expr=None, image=None, mask=None, mask_scale=None):
        self.host = host
        self.prompt_expr = prompt_expr
        if (image is not None):
            self.image = upload_image_get_id(host, image)
        else:
            self.image = image
        if (isinstance(mask, Image.Image)):
            self.mask = upload_image_get_id(host, mask)
        elif (mask is not None):
            self.mask = mask
        else:
            self.mask = None
        self.mask_scale = mask_scale
    def asdict(self):
        return pop_none({
            "prompt_expr": self.prompt_expr,
            "image": self.image,
            "mask": self.mask,
            "mask_scale": self.mask_scale
        })
class LayerDiffusionRun:
    def __init__(self, host, layers: List[Layer], guidance=5, n_steps=30, noise=None):
        self.host = host
        self.layers = [l.asdict() for l in layers]
        self.guidance = guidance
        self.n_steps = n_steps
        if (isinstance(noise, Image.Image)):
            self.noise = upload_image_get_id(host, noise)
        elif (isinstance(noise, np.ndarray)):
            self.noise = upload_image_get_id(host, noise2img(noise))
        else:
            self.noise = None
    def asdict(self):
        return pop_none({
            "layers": self.layers,
            "guidance": self.guidance,
            "n_steps": self.n_steps,
            "noise": self.noise
        })
    def run(self):
        r = requests.post(self.host+"/layer_diffusion", json=self.asdict())
        if (r.status_code==201):
            image_id = r.json()["image_id"]
        else:
            print("Error", r, r.json())
            r.raise_for_status()
        r = requests.get(f"{self.host}/image?image_id={image_id}")
        im = bytes_as_image(r.content)
        return im


        
