from tensorscaling import unit_tensor, random_tensor, random_targets, marginal, scale
import numpy as np
import pytest


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
def test_polytope_with_oracle(oracle, psi, eps=1e-2, trials=100):
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


# 2x2x2 Kronecker polytope
@oracle([2, 2, 2])
def points_222():
    positive = [
        ((1 / 2, 1 / 2), (1 / 2, 1 / 2), (1 / 2, 1 / 2)),
        ((1 / 2, 1 / 2), (1 / 2, 1 / 2), (1, 0)),
        ((1 / 2, 1 / 2), (1, 0), (1 / 2, 1 / 2)),
        ((1, 0), (1 / 2, 1 / 2), (1 / 2, 1 / 2)),
        ((1, 0), (1, 0), (1, 0)),
    ]
    negative = [
        ((1 / 2, 1 / 2), (1, 0), (1, 0)),
        ((1, 0), (1 / 2, 1 / 2), (1, 0)),
        ((1, 0), (1, 0), (1 / 2, 1 / 2)),
    ]
    return positive, negative


# 2x2x2 entanglement polytope of W state |100> + |010> + |001>
@oracle([2, 2, 2])
def points_222_w():
    positive = [
        ((1 / 2, 1 / 2), (1 / 2, 1 / 2), (1, 0)),
        ((1 / 2, 1 / 2), (1, 0), (1 / 2, 1 / 2)),
        ((1, 0), (1 / 2, 1 / 2), (1 / 2, 1 / 2)),
        ((1, 0), (1, 0), (1, 0)),
    ]
    negative = [
        ((1 / 2, 1 / 2), (1 / 2, 1 / 2), (1 / 2, 1 / 2)),
        ((1 / 2, 1 / 2), (1, 0), (1, 0)),
        ((1, 0), (1 / 2, 1 / 2), (1, 0)),
        ((1, 0), (1, 0), (1 / 2, 1 / 2)),
    ]
    return positive, negative


# 3x3x3 Kronecker polytope
@oracle([3, 3, 3])
def points_333():
    positive = [
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1, 0, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 4, 1 / 4), (3 / 4, 1 / 4, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 2, 1 / 2, 0), (1, 0, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 2, 1 / 2, 0), (3 / 4, 1 / 4, 0), (1 / 2, 1 / 4, 1 / 4)),
        ((1 / 2, 1 / 4, 1 / 4), (1 / 2, 1 / 2, 0), (3 / 4, 1 / 4, 0)),
        ((1 / 2, 1 / 4, 1 / 4), (3 / 4, 1 / 4, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1, 0, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1, 0, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 3, 0), (2 / 3, 1 / 3, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0)),
        ((1, 0, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1, 0, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
        ((1, 0, 0), (1, 0, 0), (1, 0, 0)),
        ((2 / 3, 1 / 3, 0), (1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 3, 0)),
        ((2 / 3, 1 / 3, 0), (2 / 3, 1 / 3, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((2 / 3, 1 / 6, 1 / 6), (2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0)),
        ((3 / 4, 1 / 4, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 4, 1 / 4)),
        ((3 / 4, 1 / 4, 0), (1 / 2, 1 / 4, 1 / 4), (1 / 2, 1 / 2, 0)),
    ]
    negative = []
    return positive, negative


tensor_333_T_10 = np.array(
    [
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]
)
tensor_333_T_10 = tensor_333_T_10 / np.linalg.norm(tensor_333_T_10)


@oracle([3, 3, 3])
def points_333_T_10():
    positive = [
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0), (1, 0, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 2, 1 / 4, 1 / 4), (3 / 4, 1 / 4, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 2, 1 / 2, 0), (1, 0, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 2, 1 / 2, 0), (3 / 4, 1 / 4, 0), (1 / 2, 1 / 4, 1 / 4)),
        ((1 / 2, 1 / 4, 1 / 4), (1 / 2, 1 / 2, 0), (3 / 4, 1 / 4, 0)),
        ((1 / 2, 1 / 4, 1 / 4), (3 / 4, 1 / 4, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1, 0, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1, 0, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 3, 0), (2 / 3, 1 / 3, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0)),
        ((1, 0, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 2, 0)),
        ((1, 0, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
        ((1, 0, 0), (1, 0, 0), (1, 0, 0)),
        ((2 / 3, 1 / 3, 0), (1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 3, 0)),
        ((2 / 3, 1 / 3, 0), (2 / 3, 1 / 3, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0), (2 / 3, 1 / 6, 1 / 6)),
        ((2 / 3, 1 / 6, 1 / 6), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((2 / 3, 1 / 6, 1 / 6), (2 / 3, 1 / 6, 1 / 6), (1 / 2, 1 / 2, 0)),
        ((3 / 4, 1 / 4, 0), (1 / 2, 1 / 2, 0), (1 / 2, 1 / 4, 1 / 4)),
        ((3 / 4, 1 / 4, 0), (1 / 2, 1 / 4, 1 / 4), (1 / 2, 1 / 2, 0)),
    ]
    negative = [
        ((1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0), (1 / 3, 1 / 3, 1 / 3)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)),
        ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3)),
    ]
    return positive, negative


# 4x4x4 Kronecker polytope
@oracle([4, 4, 4])
def points_444():
    positive = [
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1, 0, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 4, 1 / 12, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 8, 3 / 8, 1 / 4, 0),
            (5 / 8, 3 / 8, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 8, 3 / 8, 1 / 4, 0),
            (3 / 4, 1 / 8, 1 / 8, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (7 / 10, 3 / 20, 3 / 20, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 12, 5 / 12, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
            (5 / 8, 3 / 8, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 4, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 3 / 8, 1 / 8, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 3 / 8, 1 / 8, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 3 / 8, 0, 0),
            (3 / 8, 3 / 8, 1 / 4, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 3 / 8, 0, 0),
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 6, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 4, 1 / 12, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (7 / 10, 3 / 20, 3 / 20, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 8, 1 / 8, 0),
            (3 / 8, 3 / 8, 1 / 4, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1, 0, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (4 / 7, 3 / 7, 0, 0),
        ),
        (
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
            (4 / 7, 3 / 7, 0, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
        ),
        (
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 8, 1 / 8, 0),
        ),
        (
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
            (3 / 4, 1 / 8, 1 / 8, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (4 / 5, 1 / 10, 1 / 10, 0),
        ),
        (
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
        ),
        (
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
        ),
        (
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (4 / 9, 4 / 9, 1 / 9, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (4 / 9, 4 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 6, 1 / 6, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (5 / 6, 1 / 6, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 4, 1 / 12, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
            (3 / 4, 1 / 8, 1 / 8, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (5 / 6, 1 / 6, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1, 0, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 4, 1 / 12, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 8, 1 / 8, 0),
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 6, 1 / 6, 0, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1, 0, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
            (3 / 7, 2 / 7, 2 / 7, 0),
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
        ),
        (
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
            (3 / 7, 2 / 7, 2 / 7, 0),
        ),
        (
            (4 / 11, 4 / 11, 3 / 11, 0),
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
        ),
        (
            (4 / 11, 4 / 11, 3 / 11, 0),
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
        ),
        (
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (3 / 8, 3 / 8, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 3 / 8, 0, 0),
        ),
        (
            (3 / 8, 3 / 8, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 8, 1 / 8, 0),
        ),
        (
            (3 / 8, 3 / 8, 1 / 4, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (3 / 8, 3 / 8, 1 / 4, 0),
            (5 / 8, 3 / 8, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (3 / 8, 3 / 8, 1 / 4, 0),
            (3 / 4, 1 / 8, 1 / 8, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (2 / 5, 2 / 5, 1 / 5, 0),
            (4 / 5, 1 / 5, 0, 0),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (4 / 5, 1 / 5, 0, 0),
            (2 / 5, 2 / 5, 1 / 5, 0),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (7 / 10, 3 / 20, 3 / 20, 0),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
            (4 / 5, 1 / 10, 1 / 10, 0),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (4 / 5, 1 / 10, 1 / 10, 0),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (7 / 10, 3 / 20, 3 / 20, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (4 / 5, 1 / 10, 1 / 10, 0),
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
        ),
        (
            (2 / 5, 3 / 10, 3 / 10, 0),
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
        ),
        (
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (4 / 5, 1 / 10, 1 / 10, 0),
        ),
        (
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 5, 0),
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
        ),
        (
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 5, 0),
        ),
        (
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (2 / 5, 2 / 5, 1 / 5, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (4 / 5, 1 / 5, 0, 0),
        ),
        (
            (2 / 5, 2 / 5, 1 / 5, 0),
            (4 / 5, 1 / 5, 0, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (5 / 12, 5 / 12, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
        ),
        (
            (5 / 12, 5 / 12, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (3 / 7, 2 / 7, 2 / 7, 0),
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
        ),
        (
            (3 / 7, 2 / 7, 2 / 7, 0),
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
        ),
        (
            (3 / 7, 3 / 7, 1 / 7, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (5 / 7, 1 / 7, 1 / 7, 0),
        ),
        (
            (3 / 7, 3 / 7, 1 / 7, 0),
            (5 / 7, 1 / 7, 1 / 7, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
        ),
        (
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 9, 1 / 9, 1 / 9, 0),
        ),
        (
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (4 / 9, 4 / 9, 1 / 9, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (4 / 9, 4 / 9, 1 / 9, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
            (4 / 11, 4 / 11, 3 / 11, 0),
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
        ),
        (
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
            (4 / 11, 4 / 11, 3 / 11, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 3 / 8, 0, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
            (5 / 8, 3 / 8, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 4, 1 / 4, 0, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 4, 1 / 4, 0, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 4, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 3 / 8, 1 / 8, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 2, 3 / 8, 1 / 8, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (3 / 4, 1 / 4, 0, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1, 0, 0, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (1 / 2, 1 / 2, 0, 0),
            (1, 0, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (11 / 20, 3 / 20, 3 / 20, 3 / 20),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
        ),
        (
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
        ),
        (
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
            (4 / 7, 3 / 7, 0, 0),
        ),
        (
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (3 / 7, 3 / 7, 1 / 7, 0),
            (5 / 7, 1 / 7, 1 / 7, 0),
        ),
        (
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (4 / 7, 3 / 7, 0, 0),
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
        ),
        (
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (5 / 7, 1 / 7, 1 / 7, 0),
            (3 / 7, 3 / 7, 1 / 7, 0),
        ),
        (
            (4 / 7, 3 / 7, 0, 0),
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
        ),
        (
            (4 / 7, 3 / 7, 0, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (2 / 7, 2 / 7, 2 / 7, 1 / 7),
        ),
        (
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (7 / 12, 1 / 4, 1 / 12, 1 / 12),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 10, 3 / 10, 3 / 10, 1 / 10),
        ),
        (
            (3 / 5, 1 / 5, 1 / 10, 1 / 10),
            (1 / 2, 1 / 2, 0, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (3 / 5, 1 / 5, 1 / 5, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
        ),
        (
            (3 / 5, 1 / 5, 1 / 5, 0),
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 3 / 8, 1 / 8, 0),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 3 / 8, 1 / 4, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 3 / 8, 1 / 8, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
            (3 / 8, 1 / 4, 1 / 4, 1 / 8),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 3 / 8, 1 / 4, 0),
        ),
        (
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (5 / 8, 1 / 8, 1 / 8, 1 / 8),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (5 / 8, 3 / 8, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 8, 3 / 8, 1 / 4, 0),
        ),
        (
            (5 / 8, 3 / 8, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
        ),
        (
            (5 / 8, 3 / 8, 0, 0),
            (3 / 8, 3 / 8, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (5 / 8, 3 / 8, 0, 0),
            (1 / 2, 1 / 4, 1 / 8, 1 / 8),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (4 / 9, 4 / 9, 1 / 9, 0),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (4 / 9, 4 / 9, 1 / 9, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 9, 1 / 9, 1 / 9),
            (5 / 9, 2 / 9, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (5 / 12, 5 / 12, 1 / 6, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (2 / 3, 1 / 6, 1 / 6, 0),
            (2 / 3, 1 / 6, 1 / 6, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (2 / 3, 1 / 4, 1 / 12, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 4, 1 / 12, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 3, 1 / 3, 0, 0),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (1 / 2, 1 / 6, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (2 / 3, 1 / 3, 0, 0),
            (2 / 3, 1 / 3, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 5, 0),
        ),
        (
            (7 / 10, 1 / 10, 1 / 10, 1 / 10),
            (3 / 5, 1 / 5, 1 / 5, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
        ),
        (
            (7 / 10, 3 / 20, 3 / 20, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (7 / 10, 3 / 20, 3 / 20, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (5 / 7, 1 / 7, 1 / 7, 0),
            (3 / 7, 3 / 7, 1 / 7, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
        ),
        (
            (5 / 7, 1 / 7, 1 / 7, 0),
            (4 / 7, 1 / 7, 1 / 7, 1 / 7),
            (3 / 7, 3 / 7, 1 / 7, 0),
        ),
        (
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
            (4 / 11, 4 / 11, 3 / 11, 0),
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
        ),
        (
            (8 / 11, 1 / 11, 1 / 11, 1 / 11),
            (5 / 11, 2 / 11, 2 / 11, 2 / 11),
            (4 / 11, 4 / 11, 3 / 11, 0),
        ),
        (
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (11 / 15, 2 / 15, 1 / 15, 1 / 15),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 1 / 4, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (3 / 4, 1 / 12, 1 / 12, 1 / 12),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (5 / 12, 5 / 12, 1 / 12, 1 / 12),
        ),
        (
            (3 / 4, 1 / 8, 1 / 8, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (3 / 8, 3 / 8, 1 / 4, 0),
        ),
        (
            (3 / 4, 1 / 8, 1 / 8, 0),
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (3 / 4, 1 / 8, 1 / 8, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (7 / 24, 7 / 24, 5 / 24, 5 / 24),
        ),
        (
            (3 / 4, 1 / 8, 1 / 8, 0),
            (3 / 8, 3 / 8, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (3 / 4, 1 / 4, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (3 / 4, 1 / 4, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 4, 1 / 4, 0),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 2 / 9, 2 / 9, 2 / 9),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
        ),
        (
            (7 / 9, 1 / 9, 1 / 9, 0),
            (4 / 9, 1 / 3, 1 / 9, 1 / 9),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
            (3 / 7, 2 / 7, 2 / 7, 0),
        ),
        (
            (11 / 14, 1 / 14, 1 / 14, 1 / 14),
            (3 / 7, 2 / 7, 2 / 7, 0),
            (5 / 14, 5 / 14, 1 / 7, 1 / 7),
        ),
        (
            (4 / 5, 1 / 10, 1 / 10, 0),
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (3 / 10, 3 / 10, 1 / 5, 1 / 5),
        ),
        (
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 3 / 10, 3 / 10, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
        ),
        (
            (4 / 5, 1 / 10, 1 / 10, 0),
            (2 / 5, 2 / 5, 1 / 10, 1 / 10),
            (2 / 5, 3 / 10, 3 / 10, 0),
        ),
        (
            (4 / 5, 1 / 5, 0, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
            (2 / 5, 2 / 5, 1 / 5, 0),
        ),
        (
            (4 / 5, 1 / 5, 0, 0),
            (2 / 5, 2 / 5, 1 / 5, 0),
            (2 / 5, 1 / 5, 1 / 5, 1 / 5),
        ),
        (
            (5 / 6, 1 / 6, 0, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (5 / 6, 1 / 6, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 6, 1 / 6),
        ),
        (
            (1, 0, 0, 0),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
            (1 / 4, 1 / 4, 1 / 4, 1 / 4),
        ),
        (
            (1, 0, 0, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
            (1 / 3, 1 / 3, 1 / 3, 0),
        ),
        (
            (1, 0, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
            (1 / 2, 1 / 2, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
        ),
    ]
    negative = []
    return positive, negative


@pytest.mark.parametrize(
    "points,psi",
    [
        (points_222, None),
        (points_222, unit_tensor(2, 3)),
        (points_222_w, w_tensor),
        (points_333, None),
        (points_333, unit_tensor(3, 3)),
        (points_333_T_10, tensor_333_T_10),
        pytest.param(points_444, None, marks=pytest.mark.slow),
        pytest.param(points_444, unit_tensor(4, 4), marks=pytest.mark.slow),
    ],
)
def test_polytope_inner(points, psi, eps=1e-2):
    shape = points.shape
    positive, negative = points()

    # random starting tensor?
    if psi is None:
        psi = random_tensor(shape)

    # try to scale to "positive" examples (should always succeed since we set max_iterations to None)
    for targets in positive:
        assert scale(psi, targets, eps, method="gradient", max_iterations=None)

    # try to scale to "negative" examples (should never succeed)
    for targets in negative:
        assert not scale(psi, targets, eps, method="gradient")
