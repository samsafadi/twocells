from twocells.tensor import Tensor, Base

import numpy as np


class LossBase(Base):
    def __call__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        return self.forward(x, y)

    def forward(self, *args): raise (NotImplementedError("Not implemented"))
    def backward(self, *args): raise (NotImplementedError("Not implemented"))


class MeanSquareError(LossBase):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.data.size == y.data.size, "Data size doesn't match!"
        return Tensor(
            (1 / x.data.size) * np.sum((x.data - y.data) ** 2), self, x.requires_grad and y.requires_grad
        )

    def backward(self, *args):
        k = self.x.data.size

        x_grad = Tensor(
            [2 * (self.x.data[i] - self.y.data[i]) / k for i in range(k)],
            requires_grad=False
        )
        y_grad = Tensor(
            [2 * (self.y.data[i] - self.x.data[i]) / k for i in range(k)],
            requires_grad=False
        )

        self.x.backward(x_grad)
        self.y.backward(y_grad)


class CrossEntropyLoss(LossBase):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.data.shape == y.data.shape, "Data shape doesn't match!"

        epsilon = 1e-12  # To avoid log(0)
        x_clipped = np.clip(x.data, epsilon, 1.0 - epsilon)
        loss = -np.sum(y.data * np.log(x_clipped)) / x.data.shape[0]

        return Tensor(loss, self, x.requires_grad or y.requires_grad)

    def backward(self, *args):
        x_grad = (self.x.data - self.y.data) / self.x.data.shape[0]
        y_grad = (self.y.data - self.x.data) / self.y.data.shape[0]
        self.x.backward(Tensor(x_grad, requires_grad=False))
        self.y.backward(Tensor(y_grad, requires_grad=False))
