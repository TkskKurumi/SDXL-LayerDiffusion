import re
from ..globals import PROMPT_TAG_SEP as SEP
from ..LoRA.lora_file import WithLoRA
from ..model import model
FLOAT_PTN = r"-?\d\.?\d*"

class PromptExpr:
    def __init__(self, prompt):
        self.width  = 210
        self.height = 297
        self.seed   = None
        prompt = self._process_img_size(prompt)
        self.with_lora = WithLoRA(prompt, model.BASE_MODEL)
        prompt = self.with_lora.prompt
        pro, neg = self._process_neg(prompt)
        self.pro = self._dedup_sep(pro)
        self.neg = self._dedup_sep(neg)
    def _process_img_size(self, prompt):
        def wh(prompt):
            pttn = r"<(width|w|height|h):(FLOAT_PTN)>"
            found = re.findall(pttn, prompt)
            remain = re.split(pttn, prompt)
            for s, w in found:
                if (s.startswith("w")):
                    self.width = float(w)
                else:
                    self.height = float(w)
            return SEP.join(remain)
        def size(prompt):
            pttn = r"<size:(FLOAT_PTN)[x:](FLOAT_PTN)>"
            found = re.findall(pttn, prompt)
            remain = re.split(pttn, prompt)
            for w, h in found:
                self.width = float(w)
                self.height = float(h)
            return SEP.join(remain)
        def preset(prompt):
            pttn = r"<LANDSCAPE>"
            found = re.findall(pttn, prompt)
            remain = re.split(pttn, prompt)
            if (found):
                self.width = 16
                self.height = 9
            return SEP.join(remain)
        prompt = wh(prompt)
        prompt = size(prompt)
        prompt = preset(prompt)
        return prompt
            
    def _dedup_sep(self, prompt: str):
        return SEP.join(i.strip() for i in prompt.split(SEP.strip()) if i.strip())
    def _process_neg(self, prompt):
        pro = []
        neg = []
        p0n1 = False
        PTTN =r"(/\*|\*/)"
        print("DEBUG: split", prompt, PTTN, re.findall(PTTN, prompt), re.split(PTTN, prompt))
        remain = re.split(PTTN, prompt)
        for i in remain:
            if (re.match(PTTN, i)):
                continue
            if (p0n1):
                neg.append(i)
            else:
                pro.append(i)
            p0n1 = not p0n1
        return SEP.join(pro), SEP.join(neg)