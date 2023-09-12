import torch
from torchvision import datasets, transforms

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
import numpy as np

import matplotlib.pyplot as plt

def hopfield(Xi, beta, X):
    # Xi: N x n, beta: int, X: b x n
    return torch.softmax(beta * X @ Xi.t(), dim=1) @ Xi


def ksoftmax(x, k):
    # x: torch array: , X: b x n
    