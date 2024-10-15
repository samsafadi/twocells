from twocells.tensor import Tensor
import numpy as np

class OptimBase:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError(f"step not implemented for {type(self).__name__}")

class SGD(OptimBase):
    def __init__(self, params: list[Tensor], lr: float=1e-3):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= (param.grad.data * self.lr)


class Adam(OptimBase):
    def __init__(self, params: list[Tensor], lr: float=1e-3, b1: float=0.9, b2: float=0.999, wd=0, eps=1e-8):
        super().__init__(params, lr)
        self.b1 = b1
        self.b2 = b2
        self.b1_t = 1.
        self.b2_t = 1.
        self.wd = wd
        self.eps = eps

        self.m = [np.zeros(x.shape) for x in params]
        self.v = [np.zeros(x.shape) for x in params]

    def step(self):
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, p in enumerate(self.params):
            assert p.grad is not None, "Parameter's grad is None. Is requires_grad False?"
            grad = p.grad.data
            self.m[i] = (self.b1 * self.m[i] + (1.0 - self.b1)) * grad
            self.v[i] = (self.b2 * self.v[i] + (1.0 - self.b2)) * grad * grad
            m_hat = self.m[i] / (1.0 - self.b1_t)
            v_hat = self.v[i] / (1.0 - self.b2_t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p.data)