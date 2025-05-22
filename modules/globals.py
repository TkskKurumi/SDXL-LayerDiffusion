from os import path
import yaml
from math import sqrt
class CFGError(Exception):
    pass
CFG = None
PROMPT_TAG_SEP = ", "
def _process_cfg(cfg):
    return cfg

def load_cfg(cfg="default.yml"):
    global CFG
    if (isinstance(cfg, str) and path.exists(cfg)):
        with open(cfg, "r") as f:
            yml_str = f.read()
        CFG = yaml.safe_load(yml_str)
        CFG = _process_cfg(CFG)
    return CFG

def check_cfg():
    global CFG
    if (CFG is None):
        raise Exception("cfg is not initialized")

def get_cfg(*args):
    check_cfg()
    ret = CFG
    dft = args[-1]
    for i in args[:-1]:
        if (not isinstance(ret, dict)):
            raise KeyError(i)
        if (i not in ret):
            return dft
        ret = ret[i]
    return ret

def get_device():
    return get_cfg("device", "cuda:0")

def get_size(w, h, mo=32, area=None):
    if (area is None):
        area = get_cfg("area", 512*512)
    r = sqrt(area/w/h)/mo

    return mo*round(w*r), mo*round(h*r)