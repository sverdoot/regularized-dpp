from functools import wraps
from pathlib import Path
from time import time
from typing import Union

import numpy as np
from pdf2image import convert_from_path

from .general import DATA_DIR


def timing_decorator(func):
    @wraps(func)
    def wrap(self, *args, **kw):
        self.time_cnts.append(0)
        ts = time()
        result = func(self, *args, **kw)
        te = time()
        self.time_cnts[-1] += te - ts
        return result

    return wrap


def load_libsvm_data(name_or_path: Union[str, Path]) -> np.ndarray:
    if Path(name_or_path).exists():
        data = np.loadtxt(Path(name_or_path), dtype="str")
    else:
        data = np.loadtxt(Path(DATA_DIR, name_or_path), dtype=str)

    f = lambda x: x.split(":")[1]
    f = np.vectorize(f)
    X = f(data[:, 1:]).astype(float)
    Y = data[:, :1].astype(float)

    return X


# def convert_pdf_to_png(path: Union[str, Path]):
#     convert_from_path(path)[0].save(path, 'PNG')
