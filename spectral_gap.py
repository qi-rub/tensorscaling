import numpy as np
from tensorscaling import *
import scipy.linalg
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

if __name__ == "__main__":
  main()