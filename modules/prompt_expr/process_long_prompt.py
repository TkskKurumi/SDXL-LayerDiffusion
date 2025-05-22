from .prompt_expr import PromptExpr, SEP
from typing import List
from transformers import CLIPTokenizer


def try_n_buckets(n, tags, tokenizers):
    print("DEBUG: trying puting %s into %d buckets"%(tags, n))
    buckets = [[] for i in range(n)]
    max_slots = 0
    def update_free():
        nonlocal max_slots
        frees = []
        for bucket in buckets:
            st = SEP.join(bucket)
            if (st):
                pass


def slice_one_prompt(prompt: str, tokenizers: List[CLIPTokenizer]):
    tags = [i.strip() for i in prompt.split(SEP)]
    # print("DEBUG: slicing", prompt, tags)
    def test(n):
        buckets = [[] for i in range(n)]
        max_slots = 0
        def calc_buckets_free():
            nonlocal max_slots
            ret = []
            for i in buckets:
                st = SEP.join(i)
                if (not st):
                    free = min(i.model_max_length for i in tokenizers)
                else:
                    frees = []
                    for tokenizer in tokenizers:
                        untruncated_ids = tokenizer(st, padding="longest", return_tensors="pt").input_ids
                        slots = untruncated_ids.shape[-1]
                        # print("DEBUG: ", st, untruncated_ids, slots)
                        max_slots = max(max_slots, slots)
                        frees.append(tokenizer.model_max_length-slots)
                    free = min(frees)
                ret.append(free)
            return ret
        def select(tag):
            frees = calc_buckets_free()
            def score(idx):
                free = frees[idx]
                exist = tag in buckets[idx]
                return (free, not exist)
            return max(range(n), key=score)
        for i in tags:
            selected = select(i)
            # print("DEBUG: inserting", i, "into", buckets[selected])
            buckets[selected].append(i)
        # print("DEBUG: test %d buckets"%n, buckets)
        return min(calc_buckets_free())>=0, max_slots, buckets
    n = 1
    while (True):
        ok, max_slots, buckets = test(n)
        if (ok):
            break
        else:
            n = n+1
    # print("DEBUG: slice prompt into %d buckets, max bucket length = %d"%(len(buckets), max_slots), prompt, buckets)
    return [SEP.join(i) for i in buckets]

