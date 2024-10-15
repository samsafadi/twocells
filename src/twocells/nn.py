from __future__ import annotations
import twocells.tensor as tensor
from twocells.tensor import Tensor

from abc import abstractmethod

import numpy as np


class Network:
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

    def clear_grad(self) -> None:
        for layer in self.layers().values():
            layer.clear_grad()

    def layers(self) -> dict[str, Layer]:
        layers = {x: v for x, v in vars(self).items() if isinstance(v, Layer)}
        return layers

    def parameter_dict(self) -> dict[str, Tensor]:
        parameter_dict = {}
        for x, v in self.layers().items():
            for x1, v1 in v.parameter_dict().items():
                parameter_dict[f"{x}.{x1}"] = v1

        return parameter_dict

    def parameters(self) -> list[Tensor]:
        parameters = []
        for layer in self.layers().values():
            parameters += layer.parameters()

        return parameters

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Layer:
    def clear_grad(self) -> None:
        for param in self.parameters():
            param.clear_grad()

    def parameter_dict(self) -> dict[str, Tensor]:
        return {x: v for x, v in vars(self).items() if isinstance(v, Tensor)}

    def parameters(self) -> list[Tensor]:
        return [x for x in list(vars(self).values()) if isinstance(x, Tensor)]

    @abstractmethod
    def forward(self, *args): raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Layer):
    def __init__(self, in_size: int, out_size: int, init="kaiming"):
        super().__init__()
        if init == "kaiming":
            self.W = Tensor(
                np.random.normal(0, 2 / in_size, (in_size, out_size)),
                requires_grad=True,
            )
        elif init == "uniform": 
            self.W = Tensor(np.ones((in_size, out_size), dtype=np.float64) * 1 / (in_size * out_size), requires_grad=True)
        else:
            raise ValueError(f"{init} is not a valid init, innit")

        self.b = Tensor(np.zeros(out_size, dtype=np.float64), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.relu = tensor.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu.forward(x)


class Softmax(Layer):

    def __init__(self, dim=1):
        super().__init__()
        self.softmax = tensor.Softmax()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(x, dim=self.dim)
