from safetensors import safe_open
import torch
from os import path
import os
def load_tensor(pth, device="cpu"):
    if(pth.endswith("pt")):
        return torch.load(pth)
    else:
        ret = {}
        with safe_open(pth, framework="pt", device=device) as f:
            print(type(f))
            for k in f.keys():
                ret[k] = f.get_tensor(k)
        return ret
def save_img_and_notice(img, pth):
    os.makedirs(path.dirname(pth), exist_ok=True)
    img.save(pth)
    print("saved", pth)

def bytes_h(bytes):
    div = [1, 1<<10, 1<<20, 1<<30]
    pre = ["", "K", "M", "G"]
    ret = "%dB"%bytes
    for i in range(len(div)):
        if (bytes/div[i] < 1):
            break
        ret = "%.1f%sB"%(bytes/div[i], pre[i])
    return ret

def curr_memory_str(device=None):
    current = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    return "VRAM %s, PEAK %s"%(bytes_h(current), bytes_h(peak))


