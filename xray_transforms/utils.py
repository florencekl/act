from typing import Tuple, Union, TypeVar
import numpy as np

T = TypeVar('T')  # Define a generic type variable
SampleArg = Union[Tuple[T, T], T]


def sample(x: SampleArg) -> T:
    return np.random.uniform(x[0], x[1]) if isinstance(x, tuple) else x