from twocells import Tensor

import numpy as np

def one_hot(x: Tensor, num_classes: int) -> Tensor:
    assert x.dim == 1, "Cannot one hot encode an nD tensor for n != 1"
    return Tensor(np.squeeze(np.eye(num_classes)[x.data.astype(int).reshape(-1)]), x.parent, x.requires_grad)
