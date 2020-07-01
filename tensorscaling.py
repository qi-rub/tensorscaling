import numpy as np
from numpy.linalg import norm
import scipy.linalg
import string, operator
from typing import Optional

__all__ = [
    "unit_tensor",
    "dicke_tensor",
    "random_tensor",
    "random_unitary",
    "random_orthogonal",
    "random_spectrum",
    "random_targets",
    "marginal",
    "scale_one",
    "scale_many",
    "marginal_distances",
    "is_spectrum",
    "parse_targets",
    "Result",
    "scale",
    "scale_symmetric",
]


def unit_tensor(n, d):
    """Return n x ... x n unit tensor with d tensor factors."""
    psi = np.zeros(shape=[n] * d)
    for i in range(n):
        psi[(i,) * d] = 1
    psi /= np.sqrt(n)
    return psi


def dicke_tensor(k, n):
    """
    Return n-qubit Dicke state with k ones.

    TODO: Generalize to qudits.
    """
    psi = np.zeros(shape=(2,) * n)
    for idx in np.ndindex(psi.shape):
        if np.sum(idx) == k:
            psi[idx] = 1
    psi = psi / norm(psi)
    return psi


def random_tensor(shape):
    """Return random tensor chosen from the unitarily-invariant probability measure on the unit sphere."""
    psi = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    psi = psi / norm(psi)
    return psi


def random_unitary(n):
    """Return Haar-random n by n unitary matrix."""
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = scipy.linalg.qr(H)
    return Q


def random_orthogonal(n):
    """Return Haar-random n by n orthogonal matrix."""
    H = np.random.randn(n, n)
    Q, R = scipy.linalg.qr(H)
    return Q


def random_spectrum(n):
    """
    Return random non-increasing probability distribution.

    TODO: Currently this only produces non-singular spectra.
    """
    while True:
        # x = np.random.random(n - 1)
        x = np.random.randint(100, size=n - 1) / 100
        x = np.array([0] + sorted(x) + [1])
        x = sorted(x[1:] - x[:-1], reverse=True)
        if x[-1]:
            break

    return np.array(x)


def random_targets(shape):
    return [random_spectrum(n) for n in shape]


def ql_decomposition(g):
    """Return QL decomposition of given invertible matrix."""
    n = g.shape[0]
    assert g.shape == (n, n)
    x = np.eye(n)[::-1]
    q, r = np.linalg.qr(x @ g @ x, "complete")
    return x @ q @ x, x @ r @ x


def marginal(psi, k):
    """Return k-th quantum marginal (reduced density matrix)."""
    shape = psi.shape
    psi = np.moveaxis(psi, k, 0)
    psi = np.reshape(psi, (shape[k], np.prod(shape) // shape[k]))
    return psi @ psi.T.conj()


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
    Return dictionary of distances to target marginals in Frobenius norm.
    We recall that each target marginal is the diagonal matrix with entries the target spectrum.
    """
    return {k: norm(marginal(psi, k) - np.diag(spec)) for k, spec in targets.items()}

def marginal_spectral_radius(psi, targets):
    """
    Return dictionary of distances to target marginals in Frobenius norm.
    We recall that each target marginal is the diagonal matrix with entries the target spectrum.
    """
    return {k: max(np.linalg.eigvalsh(marginal(psi, k) - np.diag(spec))) for k, spec in targets.items()}

def marginal_distances(psi, targets):
    """
    Return dictionary of distances to target marginals in Frobenius norm.
    We recall that each target marginal is the diagonal matrix with entries the target spectrum.
    """
    return {k: norm(marginal(psi, k) - np.diag(spec)) for k, spec in targets.items()}


def is_spectrum(spec):
    return np.isclose(np.sum(spec), 1) and all(spec[:-1] >= spec[1:])


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
        is_spectrum(spec) for spec in targets.values()
    ), "target spectra should sum to one"
    assert all(
        all(spec[:-1] >= spec[1:]) for spec in targets.values()
    ), "target spectra should be ordered non-increasingly"
    return targets


class Result:
    def __init__(self, success, iterations, max_dist, gs, Us, psi, log_cap):
        self.success = success
        self.iterations = iterations
        self.max_dist = max_dist
        self.gs = gs
        self.Us = Us
        self.psi = psi
        self.log_cap = log_cap  # estimate of Borel capacity of Us @ psi (!)

    def __repr__(self):
        return f"Result(success={self.success}, iterations={self.iterations}, max_dist={self.max_dist}, ..., log_cap={self.log_cap})"

    def __bool__(self):
        return self.success


def scale(
    psi,
    targets,
    eps,
    max_iterations=2000,
    randomize=True,
    verbose=False,
    method="sinkhorn",
):
    """
    Scale tensor psi to a tensor whose marginals are eps-close in Frobenius norm to
    diagonal matrices with the given eigenvalues ("target spectra").

    The parameter targets can be a list or tuple, or a dictionary mapping subsystem
    indices to spectra. In the former case, if fewer spectra are provided than there are
    tensor factors, then those spectra will apply to the *last* marginals of the tensor.

    The parameter method can be eitehr "sinkhorn" or "gradient". In the former case,
    we use the tensor scaling algorithm from https://arxiv.org/abs/1804.04739. The latter
    algorithm is the geodesic gradient method from https://arxiv.org/abs/1910.12375.

    NOTE: There are several differences when compared to the rigorous tensor scaling
    algorithm in https://arxiv.org/abs/1804.04739. First, and most importantly, the
    maximal number of iterations is *not* chosen such that the algorithm provides any
    rigorous guarantees. Second, we use the Frobenius norm instead of the trace norm
    to quantify the distance to the targe marginals. Third, the randomization is done
    by *Haar-random unitaries* rather than by random integer matrices. Finally, our
    algorithm scales by *lower-triangular* matrices to diagonal matrices whose diagonal
    entries are *non-increasing*.

    TODO: Scaling to singular marginals is not implemented yet for Sinkhorn.
    """
    assert np.isclose(norm(psi), 1), "expect unit vectors"

    # convert targets to dictionary of arrays
    shape = psi.shape
    targets = parse_targets(targets, shape)
    targets_dual = {k: -target[::-1] for k, target in targets.items()}

    if verbose:
        print(f"scaling tensor of shape {shape} and type {psi.dtype}")
        print("target spectra:")
        for k, spec in targets.items():
            print(f"  {k}: {tuple(spec)}")

    # randomize by local unitaries
    if randomize:
        Us = {k: random_unitary(shape[k]) for k in targets}
    else:
        Us = {k: np.eye(shape[k]) for k in targets}

    # TODO: should truncate tensor and spectrum and apply algorithm
    if method == "sinkhorn":
        if any(np.isclose(spec[-1], 0) for spec in targets.values()):
            raise NotImplementedError("singular target marginals")

    # scaling methods
    def sinkhorn_step():
        # scale worst marginal using Cholesky decomposition
        rho = marginal(psi, sys)
        L = scipy.linalg.cholesky(rho, lower=True)
        L_inv = scipy.linalg.inv(L)
        g = np.diag(targets[sys] ** (1 / 2)) @ L_inv
        gs[sys] = g @ gs[sys]

        # keep track of log capacity
        # nonlocal log_cap
        # log_cap -= targets[sys] @ np.log(np.abs(np.diag(g)))

    def gradient_step():
        # TODO: check step size
        target_norm = norm([norm(target) ** 2 for target in targets])
        N_sqr = len(shape) + target_norm
        eta = 1 / (2 * N_sqr)

        # gradient step in each direction
        for k in targets:
            rho = marginal(psi, k)
            q, l = ql_decomposition(gs[k])
            H = q.conj().transpose() @ rho @ q - np.diag(targets[k])
            gs[k] = scipy.linalg.expm(-eta * H) @ l

    if method == "sinkhorn":
        step = sinkhorn_step
    elif method == "gradient":
        step = gradient_step
    else:
        raise Exception(f"Unknown method: {method}")

    # iterate
    psi_randomized = scale_many(Us, psi)
    gs = {k: np.eye(shape[k]) for k in targets}
    it = 0

    #make empty list for the spectral norms
    spectral_norms = []
    frobenius_norms = []

    while True:
        # compute current tensor and distances
        psi = scale_many(gs, psi_randomized)
        psi /= norm(psi)
        dists = marginal_distances(psi, targets)
        sys, max_dist = max(dists.items(), key=operator.itemgetter(1))
        #add the spectral norm
        spectral_norms.append(marginal_spectral_radius(psi, targets)[0])
        frobenius_norms.append(marginal_distances(psi,targets)[0])

        if verbose:
            print(f"#{it:03d}: max_dist = {max_dist:.8f} @ sys = {sys}")

        # check if we are done
        if max_dist <= eps:
            if verbose:
                print("success!")

            # fix up scaling matrices so that result of scaling is a unit vector (TODO: not needed for Sinkhorn)
            gs[sys] /= norm(scale_many(gs, psi_randomized))

            # compute capacity
            log_cap = 0
            for k in targets:
                _, l = ql_decomposition(gs[k])
                log_cap -= targets[k] @ np.log(np.abs(np.diag(l)))
            return Result(True, it, max_dist, gs, Us, psi, log_cap), spectral_norms, frobenius_norms

        if max_iterations and it == max_iterations:
            break

        # iteration step
        step()

        it += 1

    if verbose:
        print("did not converge!")
    return Result(False, it, max_dist, gs, Us, psi, log_cap=None), spectral_norms, frobenius_norms


def scale_symmetric(
    psi, target, eps, max_iterations=1000, randomize=True, verbose=False
):
    """
    Scale tensor psi to a tensor whose marginals are eps-close in Frobenius norm to
    diagonal matrices with the given eigenvalues ("target spectra") *in reverse*.

    NOTE: This algorithm follows https://arxiv.org/abs/1910.12375.
    """
    assert np.isclose(norm(psi), 1), "expect unit vectors"
    assert all(
        np.allclose(np.swapaxes(psi, 0, k), psi) for k in range(len(psi.shape))
    ), "expect symmetric tensor"

    # convert targets to dictionary of arrays
    shape = psi.shape
    _, target = parse_targets([target], shape).popitem()

    # compute step size
    N_sqr = len(shape) ** 2 + norm(target)
    eta = 1 / (2 * N_sqr)

    if verbose:
        print(f"scaling symmetric tensor of shape {shape} and type {psi.dtype}")
        print(f"target spectrum: {tuple(target)}")
        print(f"step size: {eta}")

    # randomize by local unitaries
    if randomize:
        U = random_unitary(shape[0])
    else:
        U = np.eye(shape[0])

    it = 0
    psi_initial = psi
    g = np.eye(shape[0])
    while True:
        # compute current tensor and distances
        gs = {k: g @ U for k in range(len(shape))}
        psi = scale_many(gs, psi_initial)
        psi /= norm(psi)
        dist = norm(marginal(psi, 0) - np.diag(target))
        spec_dist = norm(np.linalg.eigvalsh(marginal(psi, 0))[::-1] - target)
        if verbose:
            print(f"#{it:03d}: dist = {dist:.8f}, spec_dist = {spec_dist:.8f}")

        # check if we are done
        if spec_dist <= eps:
            if verbose:
                print("success!")

            # fix up scaling matrices so that result of scaling is a unit vector
            g /= norm(scale_many(gs, psi_initial)) ** (1 / len(shape))

            # compute capacity
            _, l = ql_decomposition(g)
            log_cap = -len(shape) * target @ np.log(np.abs(np.diag(l)))

            return Result(True, it, dist, g, U, psi, log_cap)

        if max_iterations and it == max_iterations:
            break

        # scaling step
        rho = marginal(psi, 0)
        q, l = ql_decomposition(g)
        H = q.conj().transpose() @ rho @ q - np.diag(target)
        g = scipy.linalg.expm(-eta * H) @ l

        it += 1

    if verbose:
        print("did not converge!")
    return Result(False, it, dist, g, U, psi, log_cap=None)
