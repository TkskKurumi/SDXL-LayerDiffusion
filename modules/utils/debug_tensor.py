import torch
import numpy as np
class PrintBuf:
    def __init__(self):
        self.buf = []
    def get(self):
        return "".join(self.buf)
    def __call__(self, *args, sep=" ", end="\n"):
        for idx, i in enumerate(args):
            if (idx):
                self.buf.append(sep)
            self.buf.append(str(i))
        self.buf.append(end)
def str_mean_std(t):
    if (isinstance(t, torch.Tensor)):
        return "<tensor.shape=%s, std=%.6f, mean=%.6f>"%(t.shape, torch.std(t), torch.mean(t))
    else:
        return "<ndarray.shape=%s, std=%.6f, mean=%.6f>"%(t.shape, np.std(t), np.mean(t))
def list_schema_str(obj, indent=0, visited_id=None, max_len=3000):
    if (not len(obj)):
        return "[]"
    if (visited_id is None):
        visited_id = {}
    prt = PrintBuf()
    prt("[ id=%X"%id(obj))
    visited_id[id(obj)] = obj
    ls_repr = []
    if (len(obj)):
        mx_elem = max(max_len//len(obj), 100)
    else:
        mx_elem = max_len
    for i in obj:
        ls_repr.append(obj_schema_str(i, indent+2, visited_id=visited_id, max_len=mx_elem))
    for i in ls_repr:
        if (len(i) > mx_elem):
            i = "..." + i[-mx_elem:]
        prt(" "*(indent+2), i, ",")
    prt(" "*indent+"]", end="")
    return prt.get()
def dict_schema_str(obj, indent=0, visited_id=None, max_len=3000):
    if (not len(obj)):
        return "{}"
    if (visited_id is None):
        visited_id = {}
    prt = PrintBuf()
    prt("{ id=%X"%(id(obj)))
    visited_id[id(obj)] = obj
    kv_repr = []
    if (len(obj)):
        mx_elem = max(max_len//len(obj), 100)
    else:
        mx_elem = max_len
    for k, v in obj.items():
        kv_repr.append((repr(k), obj_schema_str(v, indent+2, visited_id=visited_id, max_len=mx_elem)))
    kv_repr.sort(key=lambda x:len(x[0])+len(x[1]), reverse=True)
    for k_repr, v_repr in kv_repr:
        if (len(v_repr) > mx_elem):
            v_repr = "..." + v_repr[-mx_elem:]
        prt(" "*(indent+2)+k_repr, ':', v_repr)
    prt(" "*indent+"}", end="")
    return prt.get()

def obj_schema_str(obj, indent=0, visited_id=None, max_len=10000):
    if (visited_id is None):
        visited_id = {}
    if (isinstance(obj, dict)):
        return dict_schema_str(obj, indent, visited_id=visited_id, max_len=max_len)
    elif (isinstance(obj, str)):
        return repr(obj)
    elif (isinstance(obj, int)):
        return repr(obj)
    elif (isinstance(obj, list)):
        return list_schema_str(obj, indent, visited_id=visited_id, max_len=max_len)
    elif (isinstance(obj, tuple)):
        return repr(obj)
    elif (isinstance(obj, torch.Tensor)):
        return str_mean_std(obj)
    elif (obj is None):
        return "None"
    elif (hasattr(obj, "__dict__")):
        if (id(obj) in visited_id):
            return "%s, id=%X, ...(canbe reference cycle)"%(repr(obj), id(obj))
        else:
            visited_id[id(obj)] = obj
            return "%s, id=%X, __dict__ = %s"%(repr(obj), id(obj), obj_schema_str(obj.__dict__, indent=indent, visited_id=visited_id, max_len=max_len))
    else:
        return repr(obj)
    
def debug_fn(fn, *args, **kwargs):
    print("DEBUG: calling", "args", obj_schema_str(list(args)), "kwargs", obj_schema_str(kwargs))
    return fn(*args, **kwargs)