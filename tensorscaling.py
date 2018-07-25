import numpy as np
import scipy.linalg
import string, operator
from typing import Optional

__all__ = [
    "unit_tensor",
    "random_tensor",
    "random_unitary",
    "marginal",
    "scale_one",
    "scale_many",
    "marginal_distances",
    "parse_targets",
    "Result",
    "scale",
]


def unit_tensor(n, d):
    """Return n x ... x n unit tensor with d tensor factors."""
    psi = np.zeros(shape=[n] * d)
    for i in range(n):
        psi[(i,) * d] = 1
    psi /= np.sqrt(n)
    return psi


def random_tensor(shape):
    """Return random tensor chosen from the unitarily-invariant probability measure on the unit sphere."""
    psi = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    psi = psi / np.linalg.norm(psi)
    return psi


def random_unitary(n):
    """Return Haar-random n by n unitary matrix."""
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = scipy.linalg.qr(H)
    return Q


def marginal(psi, k):
    """Return k-th quantum marginal (reduced density matrix)."""
    shape = psi.shape
    psi = np.moveaxis(psi, k, 0)
    psi = np.reshape(psi, (shape[k], np.prod(shape) // shape[k]))
    return np.dot(psi, psi.T.conj())


def scale_one(g, k, psi):
    """Return result of applying g to the k-th tensor factor of psi."""
    assert 0 <= k <= len(psi.shape)
    assert g.shape == (psi.shape[k], psi.shape[k])

    # build an einsum rule such as "bB,ABC->AbC"
    factors_old = string.ascii_uppercase[: len(psi.shape)]
    g_old = factors_old[k]
    g_new = factors_old[k].lower()
    factors_new = factors_old[:k] + g_new + factors_old[k + 1 :]
    rule = f"{g_new}{g_old},{factors_old}->{factors_new}"

    # contract
    return np.einsum(rule, g, psi)


def scale_many(gs, psi):
    """Return result of applying a group elements to several tensor factors of psi."""
    for k, g in gs.items():
        psi = scale_one(g, k, psi)
    return psi


def marginal_distances(psi, targets):
    """
    Return dictionary of distances to target marginals in Frobenius norm. Each target
    marginal is a diagonal matrix containing the target spectrum.
    """
    return {
        k: np.linalg.norm(marginal(psi, k) - np.diag(spec))
        for k, spec in targets.items()
    }


class Result:
    def __init__(self, success, iterations):
        self.success = success
        self.iterations = iterations
        self.gs = None
        self.psi = None

    def __repr__(self):
        return f"Result(success={self.success}, iterations={self.iterations}, ...)"

    def __bool__(self):
        return self.success


def parse_targets(targets, shape):
    if isinstance(targets, (list, tuple)):
        assert len(targets) <= len(shape), "more target spectra than tensor factors"
        shift = len(shape) - len(targets)
        targets = {shift + k: spec for k, spec in enumerate(targets)}

    targets = {k: np.array(spec) for k, spec in targets.items()}
    assert targets, "no target spectra provided"
    assert all(
        len(spec) == shape[k] for k, spec in targets.items()
    ), "target dimension mismatch"
    assert all(
        np.isclose(np.sum(spec), 1) for spec in targets.values()
    ), "target spectra should sum to one"
    assert all(
        all(spec[:-1] >= spec[1:]) for spec in targets.values()
    ), "target spectra should be ordered non-increasingly"
    return targets


def scale(psi, targets, eps, max_iterations=100, randomize=True, verbose=False):
    """
    Scale tensor psi to a tensor whose marginals are eps-close in Frobenius norm to
    diagonal matrices with the given eigenvalues ("target spectra").

    The parameter targets can be a list or tuple, or a dictionary mapping subsystem
    indices to spectra. In the former case, if fewer spectra are provided than there are
    tensor factors, then those spectra will apply to the *last* marginals of the tensor.

    NOTE: There are several differences when compared to the rigorous tensor scaling
    algorithm in https://arxiv.org/abs/1804.04739. First, and most importantly, the
    maximal number of iterations is *not* chosen such that the algorithm provides any
    rigorous guarantees. Second, we use the Frobenius norm instead of the trace norm
    to quantify the distance to the targe marginals. Third, the randomization is done
    by *Haar-random unitaries* rather than by random integer matrices. Finally, our
    algorithm scales by *lower-triangular* matrices to diagonal matrices whose diagonal
    entries are *non-increasing*.

    TODO: Scaling to singular marginals is not implemented yet.
    """
    assert np.isclose(np.linalg.norm(psi), 1), "expect unit vectors"

    # convert targets to dictionary of arrays
    targets = parse_targets(targets, psi.shape)

    if verbose:
        print(f"scaling tensor of shape {psi.shape}")
        print("target spectra:")
        for k, spec in targets.items():
            print(f"  {k}: {tuple(spec)}")

    # randomize by local unitaries
    gs = {k: np.eye(psi.shape[k]) for k in targets}
    if randomize:
        for k in targets:
            U = random_unitary(psi.shape[k])
            psi = scale_one(U, k, psi)
            gs[k] = U

    # TODO: should truncate tensor and spectrum when the latter is singular
    if any(np.isclose(spec[-1], 0) for spec in targets.values()):
        raise NotImplementedError("singular target marginals")

    for it in range(max_iterations):
        # compute distances and check if we are done
        dists = marginal_distances(psi, targets)
        sys, max_dist = max(dists.items(), key=operator.itemgetter(1))
        if verbose:
            print(f"#{it:03d}: max_dist = {max_dist} (system {sys})")
        if max_dist <= eps:
            if verbose:
                print("success!")
            res = Result(True, it)
            res.gs = gs
            res.psi = psi
            return res

        # scale worst marginal using Cholesky decomposition
        rho = marginal(psi, sys)
        L = scipy.linalg.cholesky(rho, lower=True)
        L_inv = scipy.linalg.inv(L)
        g = np.diag(targets[sys] ** (1 / 2)) @ L_inv

        psi = scale_one(g, sys, psi)
        assert np.isclose(np.linalg.norm(psi), 1)
        gs[sys] = g @ gs[sys]

    if verbose:
        print("did not converge!")
    return Result(False, it)
