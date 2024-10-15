from __future__ import annotations
from abc import abstractmethod
from typing import override, Any, cast
from itertools import zip_longest
import math

import numpy as np
from numpy.typing import NDArray


class Base:
    @abstractmethod
    def backward(self, grad) -> None:
        raise (NotImplementedError("Not implemented"))


class Tensor(Base):
    def __init__(
        self,
        data: NDArray[Any] | np.floating[Any] | list[Any] | int | float | Tensor,
        parent: Base | None = None,
        requires_grad: bool = False,
    ):
        self.data: NDArray[Any]
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, type(np.ndarray)):
            self.data = data
        else:
            self.data = np.array(data)

        self._grad: Tensor | None = None
        self.parent = parent
        self.requires_grad = requires_grad

    def backward(self, grad: Tensor | None = None):
        if grad is not None and self.shape != grad.shape:  # Broadcasting shenanigans
            grad = reduce_grad_by_sum(self, grad)
        if grad is not None:
            self._grad = grad if self._grad is None else self._grad + grad
        if self.parent is not None:
            self.parent.backward(grad)

    def clear_grad(self):
        self._grad = Tensor(np.zeros(self.shape, dtype=self.data.dtype))

    @property
    def grad(self) -> Tensor | None:
        return self._grad

    @grad.setter
    def grad(self, grad: Tensor) -> None:
        self._grad = grad

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def is_leaf(self) -> bool:
        return self.parent is None

    def tpose(self, dims: tuple[int, ...]) -> Tensor:
        return Transpose()(self, dims)

    def squash(self, num_dims: int, side: str):
        return Squash()(self, num_dims, side)

    @property
    def T(self) -> Tensor:
        return self.tpose((1, 0))

    def reshape(self, dim: tuple[int, ...]):
        return Reshape()(self, dim)

    def mean(self, dim: int | tuple[int] | None = None):
        return Mean()(self, dim)

    @override
    def __repr__(self) -> str:
        x_str = f"tensor({self.data}"
        if self.requires_grad:
            x_str += f", grad={self._grad})"
        else:
            x_str += ")"
        return x_str

    @override
    def __str__(self) -> str:
        x_str = f"tensor({self.data}"
        if self.requires_grad:
            x_str += f", grad={self._grad})"
        else:
            x_str += ")"
        return x_str

    # unary ops
    def __neg__(self) -> Tensor:
        return Neg()(self)

    # binary ops
    def __add__(self, other: Tensor) -> Tensor:
        return Add()(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return Add()(self, -1 * other)

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        return Mul()(self, other)

    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self.__mul__(other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Matmul()(self, other)

    def __getitem__(self, indices):
        return Tensor(self.data[indices])

    def __eq__(self, other: object):
        if isinstance(other, Tensor):
            other = cast(Tensor, other)
            return np.array_equal(self.data, other.data)
        return False


# *** Some broadcasting utils ***
def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def reduce_grad_by_sum(x: Tensor, grad: Tensor):
    assert is_broadcastable(
        x.shape, grad.shape
    ), f"{x.shape} is not broadcastable with {grad.shape}"

    dims_to_sum_by = []
    for i, dims in enumerate(
        reversed(list(zip_longest(x.shape[::-1], grad.shape[::-1])))
    ):
        if dims[0] != dims[1]:
            dims_to_sum_by.append(i)

    return Tensor(np.sum(grad.data, tuple(dims_to_sum_by)))


# *** Function Definitions ***
class FunctionBase(Base):
    def __init__(self):
        self.saved_vars: dict[str, Any] = dict()

    def __call__(self, *args: object, **kwargs: object):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise (NotImplementedError("Not implemented"))

    @abstractmethod
    def backward(self, grad: Tensor) -> None:
        raise (NotImplementedError("Not implemented"))

    def save_vars(self, data: dict[str, Any]) -> None:
        self.saved_vars.update(data)


# *** Unary ***
class Neg(FunctionBase):
    def forward(self, x: Tensor) -> Tensor:
        self.save_vars({"x": x})
        return Tensor(-x.data, self, x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        self.saved_vars["x"].backward(-grad)


class Transpose(FunctionBase):
    def forward(self, x: Tensor, dims: tuple[int]) -> Tensor:
        self.save_vars({"x": x, "dims": dims})
        return Tensor(np.transpose(x.data, dims), self, x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x, dims = (
            self.saved_vars["x"],
            self.saved_vars["dims"],
        )
        x.backward(grad.tpose((dims)))


class Reshape(FunctionBase):
    def forward(self, x: Tensor, dim: tuple[int]) -> Tensor:
        self.save_vars({"x": x, "dim": dim})
        return Tensor(np.reshape(x.data, dim), self, x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x, dim = (
            self.saved_vars["x"],
            self.saved_vars["dim"],
        )
        x.backward(grad.reshape(dim))


class Squash(FunctionBase):
    def forward(self, x: Tensor, num_dims: int, side: str) -> Tensor:
        assert (
            num_dims < x.dim and num_dims >= 0
        ), f"num_dims({num_dims}) should be non-negative and less than x.dim({x.dim})"
        if side == "left":
            new_dims = (math.prod(x.shape[: num_dims + 1]), *x.shape[num_dims + 1 :])
        elif side == "right":
            new_dims = (*x.shape[: num_dims - 1], math.prod(x.shape[num_dims - 1 :]))
        else:
            raise ValueError(f"side must be one of left or right, was {side}")

        self.save_vars({"x": x, "num_dims": num_dims, "side": side})
        return x.reshape(new_dims)

    def backward(self, grad) -> None:
        x, num_dims, side = (
            self.saved_vars["x"],
            self.saved_vars["num_dims"],
            self.saved_vars["side"],
        )
        x.backward(grad.squash(num_dims, side))


class ReLU(FunctionBase):
    def forward(self, x: Tensor) -> Tensor:
        self.save_vars({"x": x})
        return Tensor(np.maximum(x.data, np.zeros_like(x.data)), self, x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x = self.saved_vars["x"]
        x.backward(Tensor(np.maximum(grad.data, np.zeros_like(grad.data))))


class Softmax(FunctionBase):
    def forward(self, x: Tensor, dim: int) -> Tensor:
        exps = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
        s = Tensor(exps / np.sum(exps, axis=dim, keepdims=True), self, x.requires_grad)
        self.save_vars({"x": x, "dim": dim})
        return s

    def backward(self, grad: Tensor) -> None:
        x, dim = (
            self.saved_vars["x"],
            self.saved_vars["dim"],
        )
        exps = np.exp(x.data - np.max(x.data))
        s = exps / np.sum(exps, axis=dim, keepdims=True)
        grad_input = s * (grad.data - np.sum(grad.data * s, axis=dim, keepdims=True))
        x.backward(Tensor(grad_input))


class Mean(FunctionBase):
    def forward(self, x: Tensor, axis: int | tuple[int] | None = None) -> Tensor:
        self.save_vars({"x": x, "axis": axis})
        return Tensor(np.mean(x.data), self, x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x, axis = self.saved_vars["x"], self.saved_vars["axis"]

        shrunk_by: int
        if axis is None:
            shrunk_by = x.size
        elif isinstance(axis, int):
            shrunk_by = x[axis].size
        else:
            shrunk_by = sum([x[ax] for ax in axis])

        x.backward(Tensor(np.ones_like(x.data) * (1 / shrunk_by)))


# *** Binary ***
class Mul(FunctionBase):
    def forward(self, x: Tensor | int | float, y: Tensor | int | float) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        self.save_vars({"x": x, "y": y})
        return Tensor(x.data * y.data, self, x.requires_grad or y.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x, y = self.saved_vars["x"], self.saved_vars["y"]
        x_back = grad @ y if grad.dim > 0 and y.dim > 0 else grad * y
        y_back = grad @ x if grad.dim > 0 and x.dim > 0 else grad * x
        x.backward(x_back)
        y.backward(y_back)


class Add(FunctionBase):
    def forward(self, x: Tensor | int | float, y: Tensor | int | float) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        self.save_vars({"x": x, "y": y})
        return Tensor(x.data + y.data, self, x.requires_grad or y.requires_grad)

    def backward(self, grad: Tensor) -> None:
        x, y = self.saved_vars["x"], self.saved_vars["y"]
        x.backward(grad)
        y.backward(grad)


class Matmul(FunctionBase):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert (
            x.dim == y.dim
        ), f"x.dim({x.dim}) must equal y.dim({y.dim}), broadcasting not supported yet"
        assert (
            x.dim > 0
        ), f"both tensors need to be at least 1D but are {x.dim}D and {y.dim}D"
        self.save_vars({"x": x, "y": y})
        return Tensor(x.data @ y.data, self, x.requires_grad or y.requires_grad)

    def backward(self, grad) -> None:
        x, y = self.saved_vars["x"], self.saved_vars["y"]
        if x.dim == 1:
            x.backward(grad * y)
            y.backward(grad * x)
        else:
            dims = [*np.arange(x.dim - 2), x.dim - 1, x.dim - 2]
            x.backward(grad @ y.tpose(dims))
            y.backward(x.tpose(dims) @ grad)
