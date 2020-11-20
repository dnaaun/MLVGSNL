from typing import Any, Generic, TypeVar, Callable
import torch
from torch.nn import Module


_T = TypeVar("_T")
class Base(Module):

    def forward(self, my_nice_arg: int=4) -> torch.Tensor:
        return torch.tensor(1)


reveal_type(Base().__call__)
