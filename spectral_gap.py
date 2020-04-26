import numpy as np
from tensorscaling import *
import scipy.linalg
import string, operator
import sys
#import gflags
import json

#FLAGS = gflags.FLAGS

#gflags.DEFINE_string('marginals', '[[1,1,1], [2,3,4], [3,4,5]]', 'Desired marginals to test in moment polytope.')


def covariance_matrix(marginals):
  """Returns covariance matrix in the tensor space whose marginals are given by marginals
     marginals is an array of vectors, each being the diagonal of the marginal distribution
     of its corresponding particle."""
  cov_matrix = 1
  for i in range(len(marginals)):
  	marginals[i] = sorted(marginals[i], reverse = True)
  	marginal_matrix = np.diag(marginals[i])
  	cov_matrix = np.kron(cov_matrix, marginal_matrix)
  return cov_matrix


def random_normal_tensor(marginals):
  cov_matrix = covariance_matrix(marginals)
  mean = np.zeros(cov_matrix.shape[0])
  return normalize([np.random.multivariate_normal(mean, cov_matrix)], 2)[0]


def normalize(vectors, ell):
  norm_vectors = []
  for v in vectors:
  	for i in range(len(v)):
  	  v[i] = float(v[i])
  	norm = np.linalg.norm(v, ell)
  	norm_vectors.append(np.tensordot(norm**(-1), v, axes=0))
  return norm_vectors


def main():
  config_object = json.load(open("config.json"))
  targets = config_object["marginals"]
  targets = normalize(targets, 1)
  epsilon = config_object["precision"]
  for i in range(config_object["number_of_runs"]):
    psi = random_normal_tensor(targets).reshape(3,3,3)
    scale(psi, targets, epsilon, 50, verbose = True)

def main_cole(n,targets, epsilon, number_of_runs,max_iterations=100):
  targets = normalize(targets, 1)
  for i in range(number_of_runs):
    psi = random_normal_tensor(targets).reshape(n,n,n)
    return scale(psi, targets, epsilon, max_iterations, verbose = False)

def scaling_distances(psi, targets, eps, max_iterations=200, randomize=True, verbose=False):
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

    dist_list = []

    # convert targets to dictionary of arrays
    shape = psi.shape
    targets = parse_targets(targets, shape)

    if verbose:
        print(f"scaling tensor of shape {shape} and type {psi.dtype}")
        print("target spectra:")
        for k, spec in targets.items():
            print(f"  {k}: {tuple(spec)}")

    # randomize by local unitaries
    if randomize:
        gs = {k: random_unitary(shape[k]) for k in targets}
    else:
        gs = {k: np.eye(shape[k]) for k in targets}

    # TODO: should truncate tensor and spectrum and apply algorithm
    if any(np.isclose(spec[-1], 0) for spec in targets.values()):
        raise NotImplementedError("singular target marginals")

    it = 0
    psi_initial = psi
    while True:
        # compute current tensor and distances
        psi = scale_many(gs, psi_initial)
        psi /= np.linalg.norm(psi)
        dists = marginal_distances(psi, targets)
        sys, max_dist = max(dists.items(), key=operator.itemgetter(1))
        dist_list.append(scipy.log(max_dist))
        if verbose:
            print(f"#{it:03d}: max_dist = {np.log(max_dist):.8f} @ sys = {sys}")

        # check if we are done
        if max_dist <= eps:
            if verbose:
                print("success!")

            # fix up scaling matrices so that result of scaling is a unit vector
            return dist_list

        if max_iterations and it == max_iterations:
            break

        # scale worst marginal using Cholesky decomposition
        rho = marginal(psi, sys)
        L = scipy.linalg.cholesky(rho, lower=True)
        L_inv = scipy.linalg.inv(L)
        g = np.diag(targets[sys] ** (1 / 2)) @ L_inv
        gs[sys] = g @ gs[sys]

        it += 1

    if verbose:
        print("did not converge!")
    return dist_list



if __name__ == "__main__":
  main()