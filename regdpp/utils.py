from functools import wraps
from os import PathLike
from pathlib import Path
import numpy as np
from time import time
from typing import Union

from .general import DATA_DIR


def timing_decorator(func):
    @wraps(func)
    def wrap(self, *args, **kw):
        ts = time()
        result = func(self, *args, **kw)
        te = time()
        self.time_cnts.append(te - ts)
        return result

    return wrap


def load_libsvm_data(name_or_path: Union[str, Path]) -> np.ndarray:
    if Path(name_or_path).exists():
        data = np.loadtxt(Path(name_or_path), dtype=str)
    else:
        data = np.loadtxt(Path(DATA_DIR, name_or_path), dtype=str)

    f = lambda x: x[2:]
    f = np.vectorize(f)
    X = f(data[:, 1:]).astype(float)
    Y = data[:, :1].astype(float)
    
    return X