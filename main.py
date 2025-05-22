from typing import Annotated, Any, Dict
from PIL import Image
from litestar import Litestar, post, get
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import Response
from litestar import status_codes
from uuid import uuid4
from io import BytesIO
from modules.globals import load_cfg, get_cfg
from modules.layer_diffusion import *
from modules.utils.lazy_future import retrieve_future
from modules.utils.candy import _debug
from modules.model.model import goc_base
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import sys
import traceback

THREAD_POOL = ThreadPoolExecutor(max_workers=4)
IMAGES = {}

def get_image_by_id(image_id):
    ret = IMAGES[image_id]
    ret = retrieve_future(ret)
    return ret
def save_image(obj):
    while (len(IMAGES) >= 128):
        IMAGES.pop(next(iter(IMAGES)), None)
    uuid = str(uuid4())
    IMAGES[uuid] = obj
    _debug("image saved", uuid, obj)
    return uuid

def response_err(msg, code):
    return Response(
        {"msg": msg, "code": code},
        status_code=code
    )
def layer_from_json(layer_arg: Dict) -> Layer:
    for k, v in layer_arg.items():
        if ((k == "image") and (v is not None)):
            layer_arg[k] = get_image_by_id(v)
        elif ((k == "mask" ) and isinstance(v, str)):
            layer_arg[k] = get_image_by_id(v)
    return Layer(**layer_arg)
def response_img(img: Image.Image):
    if ("A" in img.mode):
        fmt = "PNG"
        mime = "image/png"
        fmt_kwargs = {}
    else:
        fmt = "JPEG"
        mime = "image/jpeg"
        fmt_kwargs = {"quality": 98}
    bio = BytesIO()
    img.save(bio, format=fmt, **fmt_kwargs)
    bio.seek(0)
    return Response(
        bio.read(),
        media_type=mime
    )



@post(path="/upload_image")
async def handle_post_upload_image(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)]
) -> Any:
    bio = BytesIO()
    bio.write(await data.read())
    bio.seek(0)
    im = Image.open(bio)
    image_id = save_image(im)
    return Response(
        {"image_id": image_id},
        status_code=status_codes.HTTP_201_CREATED
    )


@post(path="/layer_diffusion", sync_to_thread=True)
def handle_post_layer_diffusion(
    data: Dict[str, Any]
) -> Any:
    try:
        if ("layers" in data):
            layer_args = data["layers"]
        else:
            return response_err("layers is required", status_codes.HTTP_400_BAD_REQUEST)
        layers = []
        for layer_arg in layer_args:
            layers.append(layer_from_json(layer_arg))
        
        ldr_args = {}
        ldr_args.update(data)
        ldr_args["layers"] = layers
        for k, v in ldr_args.items():
            if ((k in ["noise"] ) and (v is not None)):
                ldr_args[k] = get_image_by_id(v)

        
        ldr = LayerDiffusionRun(goc_base(), **ldr_args)
        task = THREAD_POOL.submit(ldr.run)
        image_id = save_image(task)
        return Response(
            {"image_id": image_id},
            status_code=status_codes.HTTP_201_CREATED
        )
    except Exception as e:
        traceback.print_exc()
        return response_err(str(e), 500)

@get(path="/image", sync_to_thread=True)
def handle_get_image(image_id: str) -> Any:
    try:
        im = get_image_by_id(image_id)
        return response_img(im)
    except Exception as e:
        traceback.print_exc()
        return response_err(str(e), 500)

if (__name__=="__main__"):
    import uvicorn
    app = Litestar(route_handlers=[handle_get_image, handle_post_layer_diffusion, handle_post_upload_image])
    cfg = sys.argv[-1]
    load_cfg(cfg)
    model = goc_base()
    uvicorn.run(app,
                host=get_cfg("host", "0.0.0.0"),
                port=get_cfg("port", 8001))

