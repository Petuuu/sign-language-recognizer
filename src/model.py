import numpy as np
import torch
import torch.nn.functional as F
import math
from time import time


def binary_cross_entropy(input: np.ndarray, target: np.ndarray) -> float:
    # res = 0.0
    # for p, y in zip(input.flat, target.flat):
    #    res -= y * math.log(p) + (1 - y) * math.log(1 - p)
    # return res / input.size
    return -float(np.mean(target * np.log(input) + (1 - target) * np.log1p(-input)))


if __name__ == "__main__":
    input = np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    input = np.random.rand(10000000)
    target = np.array([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]])
    target = np.random.rand(10000000)
    start = time()
    print(binary_cross_entropy(input, target))
    end = time()
    print(end - start)
    start = time()
    print(F.binary_cross_entropy(torch.tensor(input), torch.tensor(target)))
    end = time()
    print(end - start)
