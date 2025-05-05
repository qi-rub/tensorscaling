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
from ipm import (
    scale_ipm,
)
import numpy as np
import scipy.linalg
import pytest

# @pytest.mark.parametrize("shape,targets", [((2,2,2),{0: (1/2,1/2), 1: (1/2,1/2), 2: (1/2,1/2)})])
# @pytest.mark.parametrize("shape,targets", [((3,4,5),{0: (1/3,1/3,1/3), 1: (1/4,1/4,1/4,1/4), 2: (1/5,1/5,1/5,1/5,1/5)})])
@pytest.mark.parametrize("shape,targets", [
    #((2,2),{0: (1/2,1/2), 1: (1/2,1/2)}),
    #((2,2,2),{0: (1/2,1/2), 1: (1/2,1/2), 2: (1/2,1/2)}),
    #((2,3,4),{0: (1/2,1/2), 1: (1/3,1/3,1/3), 2: (1/4,1/4,1/4,1/4)}),
    #((3,3,4),{0: (1/3,1/3,1/3), 1: (1/3,1/3,1/3), 2: (1/4,1/4,1/4,1/4)}),
    #((2,2),{0: (2/3,1/3), 1: (2/3,1/3)}),
    #((2,3,4),{0: (2/3,1/3), 1: (1/3,1/3,1/3), 2: (1/4,1/4,1/4,1/4)}),
    # ((2,3,4),{0: (1,0), 1: (1/3,1/3,1/3), 2: (1/4,1/4,1/4,1/4)}),
    #((2,3,4),{0: (1,0), 1: (1/3,1/3,1/3), 2: (1/3,1/3,1/3,0)}),
    ((2,3,4),{0: (1/2,1/2), 1: (1/3,1/3,1/3), 2: (1/2,1/6,1/6,1/6)})
    ])
def test_scale_damped_newton(shape,targets):
    psi = random_tensor(shape)
    result, max_dists = scale_ipm(psi, targets, 0.01, long_step=False, method="damped_newton", verbose=False)
    assert result.success

@pytest.mark.parametrize("shape,targets,should_succeed", [
    ((2,2),{0: (1/2,1/2), 1: (1/2,1/2)}, True),
    ((2,2,2),{0: (1/2,1/2), 1: (1/2,1/2), 2: (1/2,1/2)}, True),
    ((2,3,4),{0: (1/2,1/2), 1: (1/3,1/3,1/3), 2: (1/4,1/4,1/4,1/4)}, True),
    ((2,3,4),{0: (1/2,1/2), 1: (1/3,1/3,1/3), 2: (1/2,1/6,1/6,1/6)}, True)
    ])
def test_scale_longstep_ipm(shape,targets,should_succeed):
    psi = random_tensor(shape)
    result, max_dists = scale_ipm(psi, targets, 0.001, long_step=True, method="ipm", verbose=False)
    assert (result.success == should_succeed)

def test_scale_ipm_W():
    psi = np.array([[[0,2],[1,0]],[[1,0],[0,0]]])
    targets = { k: (1/2,1/2) for k in range(3) }
    result = scale_ipm(psi, targets, 0.01, long_step=True, method="ipm", verbose=False, max_outer_iterations=3)
    assert not result.success
