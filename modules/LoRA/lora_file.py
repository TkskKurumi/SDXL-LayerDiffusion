from __future__ import annotations
from os import path
from ..utils.misc import load_tensor
from ..globals import get_cfg, get_device
from glob import glob
from typing import Dict, Iterator, Union
from ..utils.lazy_future import Lazy, retrieve_future
import re
import torch
from ..globals import PROMPT_TAG_SEP as SEP
from ..utils.candy import _debug
from ..model import disable_offload, enable_offload
class LoRAFile:
    _opened: Dict[LoRAFile] = {}
    @classmethod
    def open(cls, pth):
        if (pth in cls._opened):
            return cls._opened[pth]
        ret = cls(pth)
        cls._opened[pth] = ret
        return ret

    @classmethod
    def list_all_lora_file(cls) -> Iterator[LoRAFile]:
        lora_dir = get_cfg("lora_dir", "./files/LoRA")
        for pth in glob(path.join(lora_dir, "*.safetensors")):
            f = cls.open(pth)
        for pth, lora_file in list(cls._opened.items()):
            if (lora_file.check_deleted()):
                cls._opened.pop(pth)
        return cls._opened.values()

    @classmethod
    def list_all_lora(cls) -> Dict[str, Union[torch.Tensor, Lazy]]:
        ret = {}
        for lora_file in cls.list_all_lora_file():
            ret.update(lora_file.variants)
        return ret

    def __init__(self, pth):
        self.pth = pth
        self.loaded_mtime = 0
        self._variants = {}
    def check_deleted(self):    
        return not path.exists(self.pth)

    def check_valid(self):
        if (self.check_deleted()):
            return False
        if (path.getmtime(self.pth) > self.loaded_mtime):
            return False
        return True

    @property
    def variants(self):
        if (self.check_deleted()):
            return {}
        if (self.check_valid()):
            return self._variants
        mtime = path.getmtime(self.pth)
        sd = Lazy("load lora %s"%self.name, load_tensor, self.pth)
        variants = {self.name:sd}
        # may enhance on-the-fly lower rank compression
        self._variants = variants
        self.loaded_mtime = mtime
        return self._variants


    def _refresh_variants(self):
        mtime = path.getmtime(self.pth)
        sd = load_tensor(self.pth)
        self.statedict = sd
        self.variants = {self.name: sd}
        self.loaded_mtime = mtime
    
    @property
    def name(self):
        return path.splitext(path.basename(self.pth))[0]
    
class WithLoRA:
    def __init__(self, prompt_expr, model=None, clean_unused=True):
        self.model = model
        
        pttn = r"<lora:\S+:\S+>"
        found = re.findall(pttn, prompt_expr)
        remain = re.split(pttn, prompt_expr)
        
        self.prompt = SEP.join(i.strip() for i in remain if i.strip())
        self.lora_set = set()
        self.lora_weight = dict()
        self.clean_unused = clean_unused
        for i in found:
            prefix, name, weight = [j.strip("<>") for j in i.split(":")]
            self.lora_set.add(name)
            self.lora_weight[name] = float(weight)
        # _debug(found, remain, self.prompt, self.lora_weight)

    def __enter__(self):    
        self.apply(self.clean_unused)
    def apply(self, clean_unused, model=None):
        if (model is None):
            model = self.model
        disabled_offload = False

        # --- list required lora
        all_available_lora = LoRAFile.list_all_lora()
        lora_set = set()
        lora_weight = dict()
        for lora_name, weight in self.lora_weight.items():
            if (lora_name not in all_available_lora):
                raise KeyError("LoRA not found '%s'"%lora_name)
            lora_set.add(lora_name)
            lora_weight[lora_name] = weight
        
        # --- load required lora weight
        model_loaded_loras = set()
        for submodel, loaded_loras in model.get_list_adapters().items():
            for i in loaded_loras:
                model_loaded_loras.add(i)
        for i in model_loaded_loras:
            if (i not in lora_weight):
                if (clean_unused):
                    model.delete_adapters(i)
                else:
                    lora_weight[i] = 0
        for lora_name, w in lora_weight.items():
            if (lora_name in model_loaded_loras):
                continue
            _debug("loading LoRA", lora_name)
            t = retrieve_future(all_available_lora[lora_name])
            # _debug("loaded  LoRA", k, t)
            for lora_sd_key, lora_sd_weight in t.items():
                t[lora_sd_key] = lora_sd_weight.to(get_device()).to(torch.float16)
            
            if (not disabled_offload):
                disabled_offload = True
                disable_offload()
            model.load_lora_weights(t, adapter_name=lora_name)
        if (disabled_offload):
            enable_offload()


        # --- turn on required lora adapter
        
        adapters = list(lora_weight)
        weights  = [lora_weight[k] for k in adapters]
        if (adapters):
            _debug("set adapters", adapters, weights)
            model.set_adapters(adapters, weights)

        torch.cuda.empty_cache()






    def __exit__(self, *args):
        pass
        

if (__name__=="__main__"):
    WithLoRA("cat <lora:meowmeow:1> ears")