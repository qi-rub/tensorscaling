from tensorscaling import *
import numpy as np
import scipy.linalg
import pytest
from functools import wraps


def oracle(shape):
    def decorate(f):
        f.shape = shape
        return f

    return decorate


# 2x2x2 Kronecker polytope
@oracle([2, 2, 2])
def oracle_222(target):
    p, q, r = target[0][0], target[1][0], target[2][0]
    if p + q > 1 + r:
        return False
    if p + r > 1 + q:
        return False
    if q + r > 1 + p:
        return False
    return True


# 2x2x2 entanglement polytope of W state |100> + |010> + |001>
@oracle([2, 2, 2])
def oracle_222_w(target):
    if not oracle_222(target):
        return False
    return target[0][0] + target[1][0] + target[2][0] >= 2


w_tensor = np.array([0, 1, 1, 0, 1, 0, 0, 0]).reshape([2, 2, 2]) / np.sqrt(3)


@oracle([2, 2, 3])
def oracle_223(target):
    la = target[0][0] - 1 / 2
    mu = target[1][0] - 1 / 2
    nu = target[2]
    if 2 * la > nu[0] + nu[1] - nu[2]:
        return False
    if 2 * mu > nu[0] + nu[1] - nu[2]:
        return False
    if la + mu > nu[0]:
        return False
    if la - mu > nu[1]:
        return False
    if mu - la > nu[1]:
        return False
    if la - mu > nu[0] - nu[2]:
        return False
    if mu - la > nu[0] - nu[2]:
        return False
    return True


@oracle([2, 2, 3])
def oracle_223_interesting(target):
    if not oracle_223(target):
        return False
    if (
        (target[0][0] + target[1][0] + target[2][0] + target[2][1] < 2)
        or (target[0][0] + target[2][0] < 1)
        or (target[1][0] + target[2][0] < 1)
    ):
        return False
    return True


tensor_223_interesting = np.zeros([2, 2, 3])
tensor_223_interesting[0, 0, 0] = 1 / np.sqrt(3)
tensor_223_interesting[0, 1, 1] = 1 / np.sqrt(3)
tensor_223_interesting[1, 1, 2] = 1 / np.sqrt(3)


@oracle([2, 2, 4])
def oracle_224(target):
    la = target[0][0] - 1 / 2
    mu = target[1][0] - 1 / 2
    nu = target[2]
    if 2 * la > nu[0] + nu[1] - nu[2] - nu[3]:
        return False
    if 2 * mu > nu[0] + nu[1] - nu[2] - nu[3]:
        return False
    if la + mu > nu[0] - nu[3]:
        return False
    if la - mu > nu[1] - nu[3]:
        return False
    if mu - la > nu[1] - nu[3]:
        return False
    if la - mu > nu[0] - nu[2]:
        return False
    if mu - la > nu[0] - nu[2]:
        return False
    return True


@oracle([3, 3, 3])
def oracle_333(target):
    target = np.array(target).reshape(-1)
    if np.dot((-2, 1, 1, -2, 1, 1, 2, -1, -1), target) < -2:
        return False
    if np.dot((-2, 1, 1, -1, -1, 2, 1, 1, -2), target) < -2:
        return False
    if np.dot((-2, 1, 1, -1, 2, -1, 1, -2, 1), target) < -2:
        return False
    if np.dot((-2, 1, 1, 1, -2, 1, -1, 2, -1), target) < -2:
        return False
    if np.dot((-2, 1, 1, 1, 1, -2, -1, -1, 2), target) < -2:
        return False
    if np.dot((-2, 1, 1, 2, -1, -1, -2, 1, 1), target) < -2:
        return False
    if np.dot((-1, -1, 2, -2, 1, 1, 1, 1, -2), target) < -2:
        return False
    if np.dot((-1, -1, 2, 0, 0, 0, 0, 0, 0), target) < -1:
        return False
    if np.dot((-1, -1, 2, 1, 1, -2, -2, 1, 1), target) < -2:
        return False
    if np.dot((-1, 0, 1, -1, 1, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, -1, 1, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 1, 0, -1, -1, 1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 1, 0, -1, 0, -1, 1), target) < -1:
        return False
    if np.dot((-1, 1, 0, -1, 0, 1, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 1, 0, -1, 1, 0, 1, -1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, -1, 1, 0, 1, -1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 1, -1, 0, -1, 1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 1, -1, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 1, 0, -1, -1, 0, 1), target) < -1:
        return False
    if np.dot((-1, 2, -1, -2, 1, 1, 1, -2, 1), target) < -2:
        return False
    if np.dot((-1, 2, -1, 1, -2, 1, -2, 1, 1), target) < -2:
        return False
    if np.dot((0, -1, 1, -1, 0, 1, 1, 0, -1), target) < -1:
        return False
    if np.dot((0, -1, 1, -1, 1, 0, 0, 1, -1), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, 1, -1, -1, 1, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 1, 0, -1, -1, 0, 1), target) < -1:
        return False
    if np.dot((0, 0, 0, -1, -1, 2, 0, 0, 0), target) < -1:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, -1, -1, 2), target) < -1:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 0, 1, -1), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 1, -1, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 1, -1, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 1, -1, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 1, -1, -1, 1, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((0, 1, -1, 0, -1, 1, -1, 1, 0), target) < -1:
        return False
    if np.dot((0, 1, -1, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((1, -2, 1, -2, 1, 1, -1, 2, -1), target) < -2:
        return False
    if np.dot((1, -2, 1, -1, 2, -1, -2, 1, 1), target) < -2:
        return False
    if np.dot((1, -1, 0, -1, 1, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((1, -1, 0, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((1, 0, -1, -1, 0, 1, -1, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, -1, -1, 0, 1, 0, -1, 1), target) < -1:
        return False
    if np.dot((1, 0, -1, -1, 1, 0, -1, 0, 1), target) < -1:
        return False
    if np.dot((1, 0, -1, 0, -1, 1, -1, 0, 1), target) < -1:
        return False
    if np.dot((1, 1, -2, -2, 1, 1, -1, -1, 2), target) < -2:
        return False
    if np.dot((1, 1, -2, -1, -1, 2, -2, 1, 1), target) < -2:
        return False
    if np.dot((2, -1, -1, -2, 1, 1, -2, 1, 1), target) < -2:
        return False
    return True


@oracle([4, 4, 4])
def oracle_444(target):
    target = np.array(target).reshape(-1)
    if np.dot((-5, -1, 3, 3, -5, 3, 3, -1, 5, 1, -3, -3), target) < -5:
        return False
    if np.dot((-5, -1, 3, 3, 1, -3, -3, 5, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((-5, -1, 3, 3, 3, 3, -1, -5, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((-5, -1, 3, 3, 5, 1, -3, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, -5, 3, -1, 3, 5, 1, -3, -3), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, -5, 3, 3, -1, 5, -3, 1, -3), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, -3, 1, -3, 5, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, -3, 5, 1, -3, 3, -5, 3, -1), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 1, -3, -3, 5, 3, -1, 3, -5), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 1, -3, 5, -3, 3, -1, -5, 3), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 3, -5, 3, -1, -3, 5, 1, -3), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 3, -1, -5, 3, 1, -3, 5, -3), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 3, -1, 3, -5, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 3, 3, -1, -5, -3, 1, -3, 5), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 5, -3, 1, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-5, 3, -1, 3, 5, 1, -3, -3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -5, -1, 3, 3, 5, 1, -3, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -5, 3, -1, 3, 5, -3, 1, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -5, 3, 3, -1, 5, -3, -3, 1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, -3, 1, 5, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, -3, 5, 1, 3, 3, -5, -1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, 1, -3, 5, 3, -1, 3, -5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, 1, 5, -3, 3, -1, -5, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, 5, -3, 1, 3, -5, 3, -1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -3, 5, 1, -3, 3, -5, -1, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -1, -5, 3, 3, 1, 5, -3, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -1, 3, -5, 3, 1, -3, 5, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, -1, 3, 3, -5, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 1, -3, -3, 5, -1, 3, 3, -5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 1, -3, 5, -3, -1, 3, -5, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 1, 5, -3, -3, -1, -5, 3, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, -5, -1, 3, -3, 5, 1, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, -5, 3, -1, -3, 5, -3, 1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, -1, -5, 3, -3, 1, 5, -3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, -1, 3, -5, -3, 1, -3, 5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, 3, -5, -1, -3, -3, 5, 1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 3, 3, -1, -5, -3, -3, 1, 5), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 5, -3, -3, 1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 5, -3, 1, -3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((-5, 3, 3, -1, 5, 1, -3, -3, -5, -1, 3, 3), target) < -5:
        return False
    if np.dot((-3, -3, 1, 5, -5, 3, 3, -1, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((-3, -3, 1, 5, 3, 3, -1, -5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, -3, 5, 1, -5, 3, 3, -1, 3, 3, -5, -1), target) < -5:
        return False
    if np.dot((-3, -3, 5, 1, 3, 3, -5, -1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, -1, 3, 1, -3, 3, 1, -1, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((-3, -1, 3, 1, 1, -1, -3, 3, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((-3, -1, 3, 1, 3, 1, -1, -3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-3, -1, 3, 1, 3, 1, -1, -3, 1, -1, -3, 3), target) < -3:
        return False
    if np.dot((-3, 1, -3, 5, -5, 3, -1, 3, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((-3, 1, -3, 5, -5, 3, 3, -1, 3, -1, 3, -5), target) < -5:
        return False
    if np.dot((-3, 1, -3, 5, 3, -1, 3, -5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, 1, -3, 5, 3, 3, -1, -5, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((-3, 1, 1, 1, -3, 1, 1, 1, 3, -1, -1, -1), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -2, -2, 2, 2, 2, 2, -2, -2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -2, 2, -2, 2, 2, -2, 2, -2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -2, 2, 2, -2, 2, -2, -2, 2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -1, -1, -1, 3, 1, 1, 1, -3), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -1, -1, 3, -1, 1, 1, -3, 1), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, -1, 3, -1, -1, 1, -3, 1, 1), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 1, -3, 1, 1, -1, 3, -1, -1), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 1, 1, -3, 1, -1, -1, 3, -1), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 1, 1, 1, -3, -1, -1, -1, 3), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 2, -2, -2, 2, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 2, -2, 2, -2, -2, 2, -2, 2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 2, 2, -2, -2, -2, -2, 2, 2), target) < -3:
        return False
    if np.dot((-3, 1, 1, 1, 3, -1, -1, -1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-3, 1, 5, -3, -5, 3, 3, -1, 3, -1, -5, 3), target) < -5:
        return False
    if np.dot((-3, 1, 5, -3, 3, -1, -5, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, 3, 1, -1, -3, -1, 3, 1, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, -3, 3, 1, -1, 3, -1, -3, 1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, -1, -3, 1, 3, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, -1, -3, 3, 1, 1, 3, -1, -3), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, -1, -3, 3, 1, 3, 1, -3, -1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, -1, 3, 1, -3, 1, -1, -3, 3), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 1, -1, -3, 3, -1, 3, 1, -3), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 1, 3, -1, -3, -1, -3, 3, 1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 3, -1, -3, 1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 3, 1, -3, -1, -1, -3, 3, 1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 3, 1, -1, -3, -3, -1, 3, 1), target) < -3:
        return False
    if np.dot((-3, 3, 1, -1, 3, 1, -1, -3, -1, -3, 1, 3), target) < -3:
        return False
    if np.dot((-3, 5, -3, 1, -5, 3, 3, -1, 3, -5, 3, -1), target) < -5:
        return False
    if np.dot((-3, 5, -3, 1, 3, -5, 3, -1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, 5, 1, -3, -5, 3, -1, 3, 3, -5, 3, -1), target) < -5:
        return False
    if np.dot((-3, 5, 1, -3, -5, 3, 3, -1, 3, -5, -1, 3), target) < -5:
        return False
    if np.dot((-3, 5, 1, -3, 3, -5, -1, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-3, 5, 1, -3, 3, -5, 3, -1, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((-2, -2, 2, 2, -3, 1, 1, 1, 2, 2, -2, -2), target) < -3:
        return False
    if np.dot((-2, -2, 2, 2, -2, 2, 2, -2, 1, 1, -3, 1), target) < -3:
        return False
    if np.dot((-2, -2, 2, 2, 1, 1, -3, 1, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((-2, -2, 2, 2, 2, 2, -2, -2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-2, 2, -2, 2, -3, 1, 1, 1, 2, -2, 2, -2), target) < -3:
        return False
    if np.dot((-2, 2, -2, 2, -2, 2, 2, -2, 1, -3, 1, 1), target) < -3:
        return False
    if np.dot((-2, 2, -2, 2, 1, -3, 1, 1, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((-2, 2, -2, 2, 2, -2, 2, -2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, -3, 1, 1, 1, 2, -2, -2, 2), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, -2, -2, 2, 2, 1, 1, -3, 1), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, -2, 2, -2, 2, 1, -3, 1, 1), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, 1, -3, 1, 1, -2, 2, -2, 2), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, 1, 1, -3, 1, -2, -2, 2, 2), target) < -3:
        return False
    if np.dot((-2, 2, 2, -2, 2, -2, -2, 2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-1, -5, 3, 3, -5, 3, 3, -1, 1, 5, -3, -3), target) < -5:
        return False
    if np.dot((-1, -5, 3, 3, 1, 5, -3, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-1, -3, 1, 3, -3, 3, 1, -1, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((-1, -3, 1, 3, 3, 1, -1, -3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-1, -3, 3, 1, -3, 3, 1, -1, 1, 3, -1, -3), target) < -3:
        return False
    if np.dot((-1, -3, 3, 1, -3, 3, 1, -1, 3, 1, -3, -1), target) < -3:
        return False
    if np.dot((-1, -3, 3, 1, 1, 3, -1, -3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-1, -3, 3, 1, 3, 1, -3, -1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-1, -1, -1, 3, -3, 1, 1, 1, 1, 1, 1, -3), target) < -3:
        return False
    if np.dot((-1, -1, -1, 3, 0, 0, 0, 0, 0, 0, 0, 0), target) < -1:
        return False
    if np.dot((-1, -1, -1, 3, 1, 1, 1, -3, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-1, -1, 3, -1, -3, 1, 1, 1, 1, 1, -3, 1), target) < -3:
        return False
    if np.dot((-1, -1, 3, -1, 1, 1, -3, 1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-1, 0, 0, 1, -1, 1, 0, 0, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 0, 1, 0, 0, -1, 1, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 0, 1, 1, 0, 0, -1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((-1, 0, 0, 1, 1, 0, 0, -1, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, -1, 0, 1, 0, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, -1, 1, 0, 0, 1, 0, -1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, -1, 0, 1, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, -1, 1, 0, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, -1, 1, 0, 1, 0, -1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, 0, -1, 1, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, 1, 0, -1, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 0, 1, 0, -1, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 1, 0, -1, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 1, 0, -1, 0, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 1, 0, 0, -1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((-1, 0, 1, 0, 1, 0, 0, -1, 0, -1, 0, 1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, -1, 0, 0, 1, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, -1, 0, 1, 0, 1, 0, -1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, -1, 0, 1, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((-1, 1, 0, 0, 1, 0, 0, -1, -1, 0, 0, 1), target) < -1:
        return False
    if np.dot((-1, 3, -5, 3, -5, 3, 3, -1, 1, -3, 5, -3), target) < -5:
        return False
    if np.dot((-1, 3, -5, 3, 1, -3, 5, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((-1, 3, -1, -1, -3, 1, 1, 1, 1, -3, 1, 1), target) < -3:
        return False
    if np.dot((-1, 3, -1, -1, 1, -3, 1, 1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((-1, 3, 1, -3, -3, 3, 1, -1, 1, -1, -3, 3), target) < -3:
        return False
    if np.dot((-1, 3, 1, -3, 1, -1, -3, 3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((-1, 3, 3, -5, -5, 3, 3, -1, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((-1, 3, 3, -5, 1, -3, -3, 5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((0, -1, 0, 1, -1, 0, 1, 0, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((0, -1, 0, 1, -1, 1, 0, 0, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((0, -1, 0, 1, 0, 1, 0, -1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, -1, 0, 1, 1, 0, 0, -1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, -1, 0, 1, 0, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, -1, 1, 0, 0, 0, 1, -1, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, 0, 1, -1, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, 0, 1, 0, -1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((0, -1, 1, 0, 1, 0, -1, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, -1, 0, 0, 1, 1, 0, 0, -1), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, -1, 0, 1, 0, 0, 1, 0, -1), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, -1, 1, 0, 0, 0, 0, 1, -1), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, 0, 0, 1, -1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, 0, 1, 0, -1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((0, 0, -1, 1, 1, 0, 0, -1, -1, 0, 0, 1), target) < -1:
        return False
    if np.dot((0, 0, 0, 0, -1, -1, -1, 3, 0, 0, 0, 0), target) < -1:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 3), target) < -1:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 0, 1, -1, -1, 1, 0, 0, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((0, 0, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 1, -1, 0, -1, 1, 0, 0, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((0, 1, -1, 0, 0, -1, 1, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((0, 1, 0, -1, -1, 0, 1, 0, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((0, 1, 0, -1, -1, 0, 1, 0, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((0, 1, 0, -1, -1, 1, 0, 0, 0, -1, 0, 1), target) < -1:
        return False
    if np.dot((0, 1, 0, -1, 0, -1, 0, 1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((0, 1, 0, -1, 0, -1, 1, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((0, 1, 0, -1, 0, 0, -1, 1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((1, -3, -3, 5, -5, -1, 3, 3, 3, 3, -1, -5), target) < -5:
        return False
    if np.dot((1, -3, -3, 5, -5, 3, -1, 3, 3, -1, 3, -5), target) < -5:
        return False
    if np.dot((1, -3, -3, 5, -5, 3, 3, -1, -1, 3, 3, -5), target) < -5:
        return False
    if np.dot((1, -3, -3, 5, -1, 3, 3, -5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((1, -3, -3, 5, 3, -1, 3, -5, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((1, -3, -3, 5, 3, 3, -1, -5, -5, -1, 3, 3), target) < -5:
        return False
    if np.dot((1, -3, 1, 1, -3, 1, 1, 1, -1, 3, -1, -1), target) < -3:
        return False
    if np.dot((1, -3, 1, 1, -2, 2, -2, 2, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((1, -3, 1, 1, -2, 2, 2, -2, -2, 2, -2, 2), target) < -3:
        return False
    if np.dot((1, -3, 1, 1, -1, 3, -1, -1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((1, -3, 5, -3, -5, 3, -1, 3, 3, -1, -5, 3), target) < -5:
        return False
    if np.dot((1, -3, 5, -3, -5, 3, 3, -1, -1, 3, -5, 3), target) < -5:
        return False
    if np.dot((1, -3, 5, -3, -1, 3, -5, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((1, -3, 5, -3, 3, -1, -5, 3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((1, -1, -3, 3, -3, -1, 3, 1, 3, 1, -1, -3), target) < -3:
        return False
    if np.dot((1, -1, -3, 3, -3, 3, 1, -1, -1, 3, 1, -3), target) < -3:
        return False
    if np.dot((1, -1, -3, 3, -1, 3, 1, -3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((1, -1, -3, 3, 3, 1, -1, -3, -3, -1, 3, 1), target) < -3:
        return False
    if np.dot((1, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), target) < 0:
        return False
    if np.dot((1, 0, -1, 0, -1, 0, 1, 0, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((1, 0, -1, 0, -1, 0, 1, 0, 0, -1, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, -1, 0, -1, 1, 0, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, -1, 0, 0, -1, 1, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, -1, 0, 0, 1, -1, 1, 0, 0), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, -1, 0, 0, 1, 0, 0, -1, 1), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, -1, 0, 1, 0, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, -1, 0, 1, 0, 0, -1, 0, 1), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 0, 1), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, 0, -1, 0, 1, -1, 0, 1, 0), target) < -1:
        return False
    if np.dot((1, 0, 0, -1, 0, 0, -1, 1, -1, 0, 0, 1), target) < -1:
        return False
    if np.dot((1, 1, -3, 1, -3, 1, 1, 1, -1, -1, 3, -1), target) < -3:
        return False
    if np.dot((1, 1, -3, 1, -2, -2, 2, 2, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((1, 1, -3, 1, -2, 2, 2, -2, -2, -2, 2, 2), target) < -3:
        return False
    if np.dot((1, 1, -3, 1, -1, -1, 3, -1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((1, 1, 1, -3, -3, 1, 1, 1, -1, -1, -1, 3), target) < -3:
        return False
    if np.dot((1, 1, 1, -3, -1, -1, -1, 3, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((1, 3, -1, -3, -3, 3, 1, -1, -1, -3, 3, 1), target) < -3:
        return False
    if np.dot((1, 3, -1, -3, -1, -3, 3, 1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((1, 5, -3, -3, -5, 3, 3, -1, -1, -5, 3, 3), target) < -5:
        return False
    if np.dot((1, 5, -3, -3, -1, -5, 3, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((2, -2, -2, 2, -3, 1, 1, 1, -2, 2, 2, -2), target) < -3:
        return False
    if np.dot((2, -2, -2, 2, -2, 2, 2, -2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((2, -2, 2, -2, -3, 1, 1, 1, -2, 2, -2, 2), target) < -3:
        return False
    if np.dot((2, -2, 2, -2, -2, 2, -2, 2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((2, 2, -2, -2, -3, 1, 1, 1, -2, -2, 2, 2), target) < -3:
        return False
    if np.dot((2, 2, -2, -2, -2, -2, 2, 2, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((3, -5, -1, 3, -5, 3, 3, -1, -3, 5, 1, -3), target) < -5:
        return False
    if np.dot((3, -5, -1, 3, -3, 5, 1, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, -5, 3, -1, -5, 3, -1, 3, -3, 5, 1, -3), target) < -5:
        return False
    if np.dot((3, -5, 3, -1, -5, 3, 3, -1, -3, 5, -3, 1), target) < -5:
        return False
    if np.dot((3, -5, 3, -1, -3, 5, -3, 1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, -5, 3, -1, -3, 5, 1, -3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((3, -1, -5, 3, -5, 3, -1, 3, 1, -3, 5, -3), target) < -5:
        return False
    if np.dot((3, -1, -5, 3, -5, 3, 3, -1, -3, 1, 5, -3), target) < -5:
        return False
    if np.dot((3, -1, -5, 3, -3, 1, 5, -3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, -1, -5, 3, 1, -3, 5, -3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((3, -1, -3, 1, -3, 3, 1, -1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((3, -1, -1, -1, -3, 1, 1, 1, -3, 1, 1, 1), target) < -3:
        return False
    if np.dot((3, -1, 3, -5, -5, 3, -1, 3, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((3, -1, 3, -5, -5, 3, 3, -1, -3, 1, -3, 5), target) < -5:
        return False
    if np.dot((3, -1, 3, -5, -3, 1, -3, 5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, -1, 3, -5, 1, -3, -3, 5, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((3, 1, -3, -1, -3, 3, 1, -1, -1, -3, 3, 1), target) < -3:
        return False
    if np.dot((3, 1, -3, -1, -1, -3, 3, 1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, -3, -1, 3, 1, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, -3, -1, 3, 1, 1, -1, -3, 3), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, -3, 3, 1, -1, -3, -1, 3, 1), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, -3, 3, 1, -1, -1, -3, 1, 3), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, -1, -3, 1, 3, -3, 3, 1, -1), target) < -3:
        return False
    if np.dot((3, 1, -1, -3, 1, -1, -3, 3, -3, -1, 3, 1), target) < -3:
        return False
    if np.dot((3, 3, -5, -1, -5, 3, 3, -1, -3, -3, 5, 1), target) < -5:
        return False
    if np.dot((3, 3, -5, -1, -3, -3, 5, 1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, -5, -1, 3, 3, 1, -3, -3, 5), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, -5, 3, -1, 3, -3, 1, -3, 5), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, -5, 3, 3, -1, -3, -3, 1, 5), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, -3, -3, 1, 5, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, -3, 1, -3, 5, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((3, 3, -1, -5, 1, -3, -3, 5, -5, -1, 3, 3), target) < -5:
        return False
    if np.dot((5, -3, -3, 1, -5, 3, 3, -1, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((5, -3, 1, -3, -5, 3, -1, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((5, -3, 1, -3, -5, 3, 3, -1, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((5, 1, -3, -3, -5, -1, 3, 3, -5, 3, 3, -1), target) < -5:
        return False
    if np.dot((5, 1, -3, -3, -5, 3, -1, 3, -5, 3, -1, 3), target) < -5:
        return False
    if np.dot((5, 1, -3, -3, -5, 3, 3, -1, -5, -1, 3, 3), target) < -5:
        return False
    return True


@pytest.mark.parametrize(
    "oracle,psi",
    [
        (oracle_222, unit_tensor(2, 3)),
        (oracle_222, None),
        (oracle_222_w, w_tensor),
        (oracle_223, None),
        (oracle_223_interesting, tensor_223_interesting),
        (oracle_224, None),
        (oracle_333, None),
        (oracle_444, None),
    ],
)
def test_polytope(oracle, psi, eps=1e-2, trials=100):
    shape = oracle.shape

    # random starting tensor?
    if psi is None:
        psi = random_tensor(shape)

    for trial in range(trials):
        # choose a random target and determine whether in entanglement polytope
        targets = random_targets(shape)
        is_member = oracle(targets)

        # try to scale to target spectra
        res = scale(psi, targets, eps, max_iterations=None if is_member else 100)

        # if in polytope, scaling should always succeed since we set max_iterations to None
        if is_member:
            assert res

        # otherwise approximate scaling might still succeed => perform some sanity checks
        if res:
            marginals = [marginal(res.psi, k) for k in range(len(shape))]
            specs = [sorted(np.linalg.eigvalsh(rho), reverse=True) for rho in marginals]
            assert oracle(specs), "scalings should stay in polytope"
