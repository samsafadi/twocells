from twocells import Tensor

import requests  # type: ignore
import gzip
import os

import numpy as np

mnist_urls: list[tuple[str, str, int, tuple[int, ...]]] = [
    (
        r"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_images.gz",
        16,
        (-1, 784)
    ),
    (
        r"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "train_labels.gz",
        8,
        (-1,)
    ),
    (
        r"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_images.gz",
        16,
        (-1, 784)
    ),
    (
        r"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "test_labels.gz",
        8,
        (-1,)
    ),
]


def _download_url(url, path, chunk_size=128):
    r = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def fetch_mnist_data(prefix="data/mnist_data") -> list[Tensor]:
    data = []
    for url, path, offset, shape in mnist_urls:
        full_path = os.path.join(prefix, path)
        if not os.path.exists(full_path):
            _download_url(url, full_path)

        data.append(Tensor(np.frombuffer(gzip.open(full_path).read()[offset:], dtype=np.uint8).reshape(shape)))

    print([d.shape for d in data])
    return data
