import twocells.nn as nn
from twocells.tensor import Tensor
from twocells.optim import Adam
from twocells.loss import CrossEntropyLoss
from twocells.utils.dataset_utils import fetch_mnist_data
from twocells.utils.tensor_utils import one_hot

import numpy as np
from tqdm import trange

# net structure
class MnistFc(nn.Network):

    def __init__(self, debug=False):
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

    def forward(self, x) -> Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# load data
x_train, y_train, x_test, y_test = fetch_mnist_data()
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

print(f"Shapes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

# define and train net
net = MnistFc()
criterion = CrossEntropyLoss()
optim = Adam(net.parameters(), 5e-5)
print(net.parameter_dict().keys())

BATCH_SIZE = 1500
epochs = 25
for epoch in (t := trange(epochs)):
    for i in range(0, x_train.shape[0], BATCH_SIZE):
        net.clear_grad()
        x_batch = x_train[i:i+BATCH_SIZE, :]
        y_batch = y_train[i:i+BATCH_SIZE, :]

        out = net(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optim.step()

    out_test = net(x_test)
    out_labels = np.argmax(out_test.data, axis=1)
    y_labels = np.argmax(y_test.data, axis=1)
    accuracy = np.sum(out_labels == y_labels) / len(out_labels)
    t.set_description(f"Test accuracy: {accuracy}")