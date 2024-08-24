from tensorscaling import (
    compose,
    is_spectrum,
    marginal,
    parse_targets,
    random_spectrum,
    random_targets,
    random_tensor,
    random_unitary,
    scale,
    scale_many,
    scale_one,
    unit_tensor,
)
import numpy as np
import scipy.linalg
import pytest


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_unit_tensor(n, d):
    psi = unit_tensor(n, d)
    assert np.isclose(np.linalg.norm(psi), 1)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 4, 5)])
def test_random_tensor(shape):
    psi = random_tensor(shape)
    assert psi.shape == shape
    assert np.isclose(np.linalg.norm(psi), 1)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_random_unitary(n):
    U = random_unitary(n)
    assert U.shape == (n, n)
    assert np.allclose(U @ U.T.conj(), np.eye(n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_random_spectrum(n):
    spec = random_spectrum(n)
    assert is_spectrum(spec)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 4, 5)])
def test_random_targets(shape):
    targets = random_targets(shape)
    assert all(is_spectrum(spec) for spec in targets)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 4, 5)])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_marginal_of_random_tensor(shape, k):
    psi = random_tensor(shape)
    rho = marginal(psi, k)
    assert np.allclose(rho.trace(), 1)
    assert np.allclose(rho, rho.T.conj())
    spec = np.linalg.eigvalsh(rho)
    assert 0 <= np.min(spec) and np.max(spec) <= 1


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_unit_tensor_marginals(n, d):
    psi = unit_tensor(n, d)
    for k in range(d):
        rho = marginal(psi, k)
        assert np.allclose(rho, np.eye(n) / n)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 4, 5), (3, 4, 5, 6)])
def test_scale_one(shape):
    # scale the k-th marginal of a random tensor to the identity matrix
    for k in range(len(shape)):
        psi = random_tensor(shape)
        rho = marginal(psi, k)
        g = scipy.linalg.inv(scipy.linalg.sqrtm(rho))
        psi = scale_one(g, k, psi)
        assert np.allclose(marginal(psi, k), np.eye(shape[k]))


@pytest.mark.parametrize(
    "shape,targets",
    [
        (
            [3, 4, 5],
            [
                (0.5, 0.25, 0.25),
                (1 / 3, 0.25, 0.25, 0.16666666666666666),
                (1 / 3, 0.2, 0.18333333333333335, 0.15, 0.13333333333333333),
            ],
        ),
        (
            [3, 4, 5],
            [
                (1 / 3, 0.25, 0.25, 0.16666666666666666),
                (1 / 3, 0.2, 0.18333333333333335, 0.15, 0.13333333333333333),
            ],
        ),
        (
            [3, 4, 5],
            [
                (1 / 3, 0.2, 0.18333333333333335, 0.15, 0.13333333333333333),
            ],
        ),
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
def test_scale_success(shape, targets, eps):
    # check that we scale successfully
    psi = random_tensor(shape)
    res = scale(psi, targets, eps)
    assert res and res.success

    # check that state has the right marginals
    targets = parse_targets(targets, shape)
    for k, spec in targets.items():
        rho = marginal(res.psi, k)
        assert np.linalg.norm(rho - np.diag(spec)) <= eps

    # check that the state can be obtained by applying the scaling matrices
    psi_expected = scale_many(compose(res.gs, res.Us), psi)
    assert np.allclose(res.psi, psi_expected)


@pytest.mark.parametrize(
    "shape,targets",
    [
        (
            [3, 4, 5],
            [
                (0.9, 0.05, 0.05),
                (1 / 3, 0.25, 0.25, 0.16666666666666666),
                (1 / 3, 0.2, 0.18333333333333335, 0.15, 0.13333333333333333),
            ],
        )
    ],
)
@pytest.mark.parametrize("eps", [1e-10])
def test_scale_failure(shape, targets, eps):
    # check that we fail to scale
    psi = random_tensor(shape)
    res = scale(psi, targets, eps)
    assert not res and not res.success


def test_scale_without_randomization():
    psi = unit_tensor(2, 3)

    # for this spectrum, it doesn't matter whether we randomize or not
    targets = ([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    res = scale(psi, targets, 1e-4)
    assert res and np.isclose(res.log_cap, 0)
    res = scale(psi, targets, 1e-4, randomize=False)
    assert res and np.isclose(res.log_cap, 0)

    # but for this one it matters
    targets = ([0.6, 0.4], [0.5, 0.5], [0.5, 0.5])
    assert scale(psi, targets, 1e-4)
    assert not scale(psi, targets, 1e-4, randomize=False)


@pytest.mark.parametrize(
    "shape,targets,is_uniform",
    [
        (
            [3, 3, 3],
            [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]],
            True,
        ),
        (
            [3, 3, 3],
            [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
            False,
        ),
    ],
)
def test_capacity(shape, targets, is_uniform):
    psi = random_tensor(shape)
    res = scale(psi, targets, 1e-4, method="sinkhorn", randomize=False)
    assert res

    # in the uniform case, computing the capacity should not depend on the randomization step
    assert is_uniform == np.isclose(
        scale(psi, targets, 1e-4, method="sinkhorn", randomize=True).log_cap,
        res.log_cap,
    )

    # computing the capacity should not depend on the method
    assert np.isclose(
        scale(psi, targets, 1e-4, method="gradient", randomize=False).log_cap,
        res.log_cap,
    )
