import math
from typing import Union

import numpy as np


def randround(x: Union[float, int]) -> int:
    d = math.floor(x)
    return d + int(np.random.random() < x - d)
