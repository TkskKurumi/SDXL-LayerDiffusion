import numpy as np
from PIL import Image
def img2noise(img: Image.Image, shape):
    img_arr = np.asarray(img).astype(np.float32).flatten()
    shape_prod = np.prod(shape)
    img_shape_prod = np.prod(img_arr.shape)
    ret = np.zeros((shape_prod, ), np.float32)
    for i in range(0, shape_prod, img_shape_prod):
        ed = min(i+img_shape_prod, shape_prod)
        ret[i:ed] = img_arr[:ed-i]
    ret = (ret-ret.mean())/np.std(ret)
    return ret.reshape(shape)
def noise2img(noise: np.ndarray):
    n = (noise-noise.mean())/(noise.max()-noise.min())
    return Image.fromarray((n*255).astype(np.uint8))


    