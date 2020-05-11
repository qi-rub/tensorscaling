from tensorscaling import *
import numpy as np


def test_unit_tensor():
    psi = unit_tensor(2, 4)

    target = [0.5, 0.5]
    assert scale_symmetric(psi, target, 1e-4)

    target = [0.6, 0.4]
    assert scale_symmetric(psi, target, 1e-4)

    target = [0.7, 0.3]
    assert scale_symmetric(psi, target, 1e-4)


def symmetric_qubit_polytope(psi, eps, max_trials=5):
    lb = 0.5
    ub = 1
    while ub - lb > eps:
        mid = 0.5 * (lb + ub)
        target = [mid, 1 - mid]
        # print(f"target: {mid}")
        for _ in range(max_trials):
            if scale_symmetric(psi, target, eps / 2):
                ub = mid
                break
        else:
            lb = mid
    return lb, ub


def dicke_0011():
    psi = np.zeros(shape=[2, 2, 2, 2])
    psi[0, 0, 1, 1] = 1
    psi[0, 1, 0, 1] = 1
    psi[0, 1, 1, 0] = 1
    psi[1, 0, 0, 1] = 1
    psi[1, 0, 1, 0] = 1
    psi[1, 1, 0, 0] = 1
    psi = psi / np.linalg.norm(psi)
    return psi


def dicke_0001():
    psi = np.zeros(shape=[2, 2, 2, 2])
    psi[0, 0, 0, 1] = 1
    psi[0, 0, 1, 0] = 1
    psi[0, 1, 0, 0] = 1
    psi[1, 0, 0, 0] = 1
    psi = psi / np.linalg.norm(psi)
    return psi


def test_dicke_0011():
    psi = dicke_0011()

    target = [0.5, 0.5]
    assert scale_symmetric(psi, target, 1e-4)

    target = [0.6, 0.4]
    assert scale_symmetric(psi, target, 1e-4)

    target = [0.7, 0.3]
    assert scale_symmetric(psi, target, 1e-4)

    lb, ub = symmetric_qubit_polytope(psi, eps=1e-3)
    assert lb <= 0.5 <= ub


def test_dicke_0001():
    psi = dicke_0001()

    lb, ub = symmetric_qubit_polytope(psi, eps=1e-3)
    assert lb <= 0.75 <= ub
