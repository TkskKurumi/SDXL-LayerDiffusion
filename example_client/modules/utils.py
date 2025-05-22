from io import BytesIO
import requests
from typing import List, Dict
from PIL import Image
import numpy as np
import os
def noise_as_image(noise: np.ndarray):
    noise = (noise-noise.min())/(noise.max()-noise.min())*255
    return Image.fromarray(noise.astype(np.uint8))
def upload_image_get_id(host, im):
    if (("A" in im.mode) or (im.mode=="P")):
        fmt = "PNG"
        fmt_kwa = {}
        fn = "data.png"
    else:
        fmt = "JPEG"
        fmt_kwa = {"quality": 98}
        fn = "data.png"
    bio = BytesIO()
    im.save(bio, format=fmt, **fmt_kwa)
    bio.seek(0)
    files = {"data": (fn, bio)}
    r = requests.post(f"{host}/upload_image", files=files)
    if (r.status_code==201):
        return r.json()["image_id"]
    else:
        r.raise_for_status()
def pop_none(d):
    for k, v in list(d.items()):
        if (v is None):
            d.pop(k)
    return d
def bytes_as_image(bytes):
    bio = BytesIO()
    bio.write(bytes)
    bio.seek(0)
    im = Image.open(bio)
    return im
def save_image(img, pth):
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    img.save(pth)
    print(pth)