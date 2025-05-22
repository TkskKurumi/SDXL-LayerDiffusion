from threading import Lock
from concurrent.futures import Future
class Lazy:
    def __init__(self, msg, fn, *args, **kwargs):
        self.msg = msg
        self.cached = False
        self.cache_result = None
        self.lck = Lock()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
    def result(self):
        with self.lck:
            if (self.cached):
                return self.cache_result
            if (self.msg):
                print(self.msg)
            result = self.fn(*self.args, **self.kwargs)
            self.cached = True
            self.cache_result = result
            return self.cache_result
def retrieve_future(obj):
    while (isinstance(obj, Lazy) or isinstance(obj, Future)):
        obj = obj.result()
    return obj
solve_lazy = retrieve_future