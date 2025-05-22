from ..data_classes import *
from ...globals import get_device
import torch
def encode_1prompt(p: str, t: CLIPTokenizer, e: CLIPTextModel):
    tokens = t(p, padding="max_length", max_length=t.model_max_length, truncation=True, return_tensors="pt").input_ids
    embd = e(tokens.to(get_device()), output_hidden_states=True)
    return embd


def encode_2prompt(p1, t1, e1, p2, t2, e2):
    pe_list = []
    pooled = None
    for p, t, e in [(p1, t1, e1), (p2, t2, e2)]:
        embd = encode_1prompt(p, t, e)
        if ((pooled is None) and embd[0].ndim == 2):
            pooled = embd[0].to(get_device())
        pe_list.append(embd.hidden_states[-2].to(get_device()))
    return torch.concat(pe_list, dim=-1), pooled

def encode_2prompt_model(model, p1, p2):
    return encode_2prompt(p1, model.tokenizer, model.text_encoder, p2, model.tokenizer_2, model.text_encoder_2)