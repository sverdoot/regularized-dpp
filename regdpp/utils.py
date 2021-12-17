from functools import wraps
from time import time


def timing_decorator(func):
    @wraps(func)
    def wrap(self, *args, **kw):
        ts = time()
        result = func(self, *args, **kw)
        te = time()
        self.time_cnts.append(te - ts)
        return result

    return wrap
