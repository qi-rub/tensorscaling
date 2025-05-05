import numpy as np
from numpy.linalg import norm, eig, eigh
import scipy.linalg
from scipy.special import exprel
import string
import operator
import functools
from tensorscaling import (
    marginal,
    parse_targets,
    ql_decomposition,
    random_spectrum,
    random_unitary,
    Result,
    scale_many,
    scale_one,
)

def scale_ipm(
    psi,
    targets,
    eps,
    max_outer_iterations=10,
    initial_radius=10,
    verbose=False,
    long_step=True,
    method="ipm",
    ):

    shape = psi.shape

    if method not in ["ipm", "damped_newton"]:
        raise NotImplementedError(f"unknown method {method}")

    Npisq = len(shape) + np.sqrt(sum(norm(targets[k])**2 for k in targets))
    sqrtalphainv = np.sqrt(4 * Npisq**2 / (4 * Npisq - 1))
    if verbose:
        print("1/sqrt(alpha): ", sqrtalphainv)

    outer_it = 0
    inner_it = 0
    total_inner_it = 0
    Ps = {k: np.eye(shape[k]) for k in targets}

    # current upper bound on the radius
    if method == "ipm":
        R = initial_radius
        theta = 1 + R**2 / 2
        lambda1 = 1/4
        lambda2 = 1/9
        t_factor = compute_t_factor(theta, sqrtalphainv, lambda1, lambda2)
        if verbose:
            print(f"barrier parameter: {theta}  t_factor: {t_factor}")

    # initialized later dynamically
    t_curr = 0

    # keeping track of long-step step-counts
    nsteps = []
    max_dists = []

    while True:
        # some pre-processing
        Ps_eigs = { k : eigh(Ps[k]) for k in targets }
        Ps_eigvals = { k : Ps_eigs[k][0] for k in targets }
        Ps_eigvecs = { k : Ps_eigs[k][1] for k in targets }

        
        P_eigvals = functools.reduce(np.kron, Ps_eigvals)
        P_eigvecs = functools.reduce(np.kron, Ps_eigvecs)
        P_eig = (P_eigvals, P_eigvecs)

        Ps_sqrt = {k: vecs @ np.sqrt(np.diag(vals)) @ vecs.conjugate().transpose() for k, (vals, vecs) in Ps_eigs.items()}
        qs = dict()
        ls = dict()
        for k in targets:
            q, l = ql_decomposition(Ps_sqrt[k])
            qs[k] = q
            ls[k] = l

        psi_scaled = scale_many(Ps_sqrt, psi)
        psi_scaled /= norm(psi_scaled)

        dists = marginal_spec_distances(psi_scaled, targets)
        sys, max_dist = max(dists.items(), key=operator.itemgetter(1))
        max_dists.append(max_dist)
        stop_condition = (max_dist <= eps)
        if stop_condition:
            # desired precision achieved
            if verbose:
                print(f"psi_scaled: {psi_scaled}")
                print(f"marginal spectra: {[np.linalg.eigvalsh(marginal(psi_scaled, k)) for k in targets]})")
            log_cap = 0
            psi_scaled_unnormalized = scale_many(Ps_sqrt, psi)
            log_cap -= np.log(np.linalg.norm(psi_scaled_unnormalized))
            for k in targets:
                _, l = ql_decomposition(Ps_sqrt[k])
                log_cap -= targets[k] @ np.log(np.abs(np.diag(l)))
            return Result(max_dist <= eps, inner_it, max_dist, Ps_sqrt, {k:np.eye(shape[k]) for k in targets}, psi_scaled, log_cap), max_dists

        if verbose:
            print(f"marginal spectra: {[np.linalg.eigvalsh(marginal(psi_scaled, k)) for k in targets]})")

        # Compute gradient of target-shifted Kempf--Ness function
        objective_gradient_as_dict = dict()
        # gradient of shifted Kempf--Ness
        for k in targets:
            rho = marginal(psi_scaled, k)
            H = rho - qs[k] @ np.diag(targets[k]) @ qs[k].conj().transpose()
            objective_gradient_as_dict[k] = H

        # Evaluation function of Hessian of target-shifted Kempf--Ness function
        def hessian_eval(H):
            # Kempf--Ness function has Hessian:
            # <v|Pi(H)^2|v> - (<v|Pi(H)|v>)^2 = norm( Pi(H) |v> )^2 -(<v|Pi(H)|v>)^2
            # eval Pi(H)|v>
            sum_of_shifts = np.zeros(psi.shape, dtype=np.complex128)
            for k in targets:
                psi_scaled_k = scale_one(H[k], k, psi_scaled)
                sum_of_shifts += psi_scaled_k
            ev = (norm(sum_of_shifts)**2 - np.vdot(psi_scaled, sum_of_shifts)**2)

            # term coming from shifting trick: tr[lambda_k [R_k, R_k*]]
            for k in targets:
                R_k = np.triu(qs[k].conj().transpose() @ H[k] @ qs[k], 1)
                ev += np.vdot(np.diag(targets[k]), (R_k @ R_k.conjugate().transpose() - R_k.conjugate().transpose() @ R_k))

            return ev

        input_indexing = np.hstack(([0], np.cumsum(np.square(shape))))

        objective_hessian_hermitian_as_dict = dict()
        for k in targets:
            for l in targets:
                for i_k in range(shape[k]):
                    for j_k in range(shape[k]):
                        H_k = elementary_hermitian_matrix(shape[k],i_k,j_k)
                        for i_l in range(shape[l]):
                            for j_l in range(shape[l]):
                                H_l = elementary_hermitian_matrix(shape[l],i_l,j_l)

                                Hplus = { s : np.zeros((shape[s], shape[s]), dtype=np.complex128) for s in targets }
                                Hmin = { s : np.zeros((shape[s], shape[s]), dtype=np.complex128) for s in targets }
                                Hplus[k] += H_k
                                Hplus[l] += H_l
                                Hmin[k] += H_k
                                Hmin[l] -= H_l
                                Hplus_eval = hessian_eval(Hplus)
                                Hmin_eval = hessian_eval(Hmin)
                                entry = (Hplus_eval - Hmin_eval) / 4
                                objective_hessian_hermitian_as_dict[(k,l,i_k,j_k,i_l,j_l)] = np.real(entry)

        objective_hessian_hermitian = np.zeros((np.sum(np.square(shape)), np.sum(np.square(shape))))
        for (k,l,i_k,j_k,i_l,j_l), entry in objective_hessian_hermitian_as_dict.items():
            objective_hessian_hermitian[input_indexing[k]+i_k*shape[k]+j_k, input_indexing[l]+i_l*shape[l]+j_l] = np.real(entry)

        objective_gradient_hermitian = np.zeros(np.sum(np.square(shape)))
        for k in targets:
            for i_k in range(shape[k]):
                for j_k in range(shape[k]):
                    objective_gradient_hermitian[input_indexing[k]+i_k*shape[k]+j_k] = np.real(np.vdot(elementary_hermitian_matrix(shape[k],i_k,j_k), objective_gradient_as_dict[k]))

        if verbose:
            print("objective gradient hermitian: ", objective_gradient_hermitian)
            print("objective hessian hermitian: ", objective_hessian_hermitian)
            print("objective hessian hermitian eigenvalues: ", np.linalg.eigvalsh(objective_hessian_hermitian))

        if method == "damped_newton":
            og = objective_gradient_hermitian
            oh = objective_hessian_hermitian
            systemsol = - np.linalg.pinv(oh) @ og
            # newton_step = { k : np.array(systemsol[input_indexing[k]:input_indexing[k+1]]).reshape((shape[k],shape[k])) for k in targets }
            newton_step = { k : np.zeros((shape[k],shape[k]),dtype=np.complex128) for k in targets }
            for k in targets:
                for i in range(shape[k]):
                    for j in range(shape[k]):
                        newton_step[k] += systemsol[input_indexing[k]+i*shape[k]+j] * elementary_hermitian_matrix(shape[k],i,j)

            if verbose:
                print("newton_step: ", newton_step)
            # print("gradient size: ", {k : norm(objective_gradient_as_dict[k]) for k in targets})
                print("flat step size: ", {k : norm(newton_step[k]) for k in targets})
                print("local norm step size: ", np.sqrt(np.vdot(og, np.linalg.pinv(oh) @ og)))
                print(f"iteration count: {inner_it}")
            distsq_evals = { k : np.linalg.norm(np.log(Ps_eigvals[k]))**2 for k in targets }
            distsq_eval = sum(distsq_evals[k] for k in targets)
            if verbose:
                print(f"distance to start: {distsq_eval}")
                print('-'*80)
            # damping; necessary to avoid numerical issues
            newton_step = { k : newton_step[k] * min(1,1/norm(newton_step[k])) for k in targets }

            Ps = { k : Ps_sqrt[k] @ scipy.linalg.expm(newton_step[k]) @ Ps_sqrt[k] for k in targets }

        elif method == "ipm":
            # compute derivative terms coming from barrier

            distsq_evals = { k : np.linalg.norm(np.log(Ps_eigvals[k]))**2 for k in targets }
            distsq_eval = sum(distsq_evals[k] for k in targets)

            distsq_grads = { k : distsq_grad_transported(Ps_eigs[k]) for k in targets }

            barrier_gradient_hermitian = np.zeros(np.sum(np.square(shape)))
            for k in targets:
                for i_k in range(shape[k]):
                    for j_k in range(shape[k]):
                        # should be real-valued!
                        barrier_gradient_hermitian[input_indexing[k]+i_k*shape[k]+j_k] = np.real(np.vdot(elementary_hermitian_matrix(shape[k],i_k,j_k), distsq_grads[k]))

            barrier_hessian_hermitian = np.outer(barrier_gradient_hermitian, barrier_gradient_hermitian) / (2 * (R**2 / 2 - distsq_eval / 2))**2
            barrier_gradient_hermitian *= 1 / 2 * (1 + 1 / (R**2/2 - distsq_eval/2))

            distsq_hesss_herm = { k : distsq_hessian_herm(Ps_eigs[k]) for k in targets }
            for k in targets:
                barrier_hessian_hermitian[input_indexing[k]:input_indexing[k+1],input_indexing[k]:input_indexing[k+1]] += distsq_hesss_herm[k] / 2 * (1 + 1/(R**2/2 - distsq_eval/2))

            if t_curr == 0:
                # initialize
                t_curr = lambda1 / (2 * np.sqrt(np.real(np.vdot(objective_gradient_hermitian, scipy.linalg.pinvh(barrier_hessian_hermitian) @ objective_gradient_hermitian))) * sqrtalphainv)
                if verbose:
                    print(f"initializing t_curr to {t_curr}")

            def compute_newton(t):
                systemA = barrier_hessian_hermitian + t * objective_hessian_hermitian
                systemb = barrier_gradient_hermitian + t * objective_gradient_hermitian
                systemAinv = scipy.linalg.pinvh(systemA)
                systemsol = -systemAinv @ systemb
                newton_decrement = np.sqrt(np.real(np.vdot(systemb, -systemsol)))
                newton_decrement_sqrtalphainv = newton_decrement * sqrtalphainv
                return systemsol, newton_decrement_sqrtalphainv

            systemsol, newton_decrement_sqrtalphainv = compute_newton(t_curr)
            if long_step:
                ctr = -1
                while True:
                    t_curr = t_curr * t_factor
                    systemsol, newton_decrement_sqrtalphainv = compute_newton(t_curr)
                    ctr += 1
                    if newton_decrement_sqrtalphainv > lambda1:
                        break
                nsteps.append(ctr)
                if verbose:
                    print(f"long step: increased t {ctr} times")
                    print(nsteps)
                systemsol, newton_decrement_sqrtalphainv = compute_newton(t_curr / t_factor)
            t_curr = t_curr * t_factor
            if verbose:
                print("increasing t to ", t_curr, " at inner_it ", inner_it)
                print("Newton decrement times 1/sqrt(alpha): ", newton_decrement_sqrtalphainv, " next expected: ", 2*(newton_decrement_sqrtalphainv)**2)
                print("R - dist", R - np.sqrt(distsq_eval))
            if (np.sqrt(distsq_eval) > 0.95 * R):
                if outer_it >= max_outer_iterations:
                    return Result(max_dist <= eps, inner_it, max_dist, Ps_sqrt, {k:np.eye(shape[k]) for k in targets}, psi_scaled, float('NaN')), max_dists

                R *= 2
                theta = 1 + R**2 / 2
                t_factor = compute_t_factor(theta, sqrtalphainv, lambda1, lambda2)
                t_curr = 0
                outer_it += 1
                total_inner_it += inner_it
                inner_it = 0
                max_dists = []
                Ps = {k: np.eye(shape[k]) for k in targets}
                print(f"too close to boundary, increasing R to {R}")
                continue

            newton_step = { k : np.zeros((shape[k],shape[k]),dtype=np.complex128) for k in targets }
            for k in targets:
                for i in range(shape[k]):
                    for j in range(shape[k]):
                        newton_step[k] += systemsol[input_indexing[k]+i*shape[k]+j] * elementary_hermitian_matrix(shape[k],i,j)

            Ps = { k : hermitian_part(Ps_sqrt[k] @ scipy.linalg.expm(newton_step[k]) @ Ps_sqrt[k]) for k in targets }
            if verbose:
                print(f"current R: {R} t: {t_curr}")
                print('-'*80)

        inner_it += 1


def hermitian_part(X):
    return (X + X.conjugate().transpose())/2


def compute_t_factor(theta, sqrtalphainv, lambda1, lambda2):
    # the factor 0.9 makes it slightly conservative
    return np.exp(0.9 * (lambda1-lambda2)/(lambda1+np.sqrt(theta)*sqrtalphainv))


def elementary_hermitian_matrix(n, i, j):
    if i == j:
        return elementary_square_matrix(n, i, j)
    elif i < j:
        return (elementary_square_matrix(n, i, j) + elementary_square_matrix(n, j, i)) / np.sqrt(2)
    else:
        return 1j * (elementary_square_matrix(n, i, j) - elementary_square_matrix(n, j, i)) / np.sqrt(2)


def marginal_spec_distances(psi, targets):
    """
    Return dictionary of distances to target marginals in Frobenius norm.
    We recall that each target marginal is the diagonal matrix with entries the target spectrum.
    """
    return {k: norm(np.linalg.eigvalsh(marginal(psi, k)) - np.sort(spec)) for k, spec in targets.items()}


def elementary_square_matrix(n, i, j):
    E_ij = np.zeros((n,n))
    E_ij[i,j]=1
    return E_ij


def distsq_grad_transported(P_eig):
    eigvals, eigvecs = P_eig
    newvals = np.log(eigvals)
    grad_h = eigvecs @ np.diag(newvals) @ eigvecs.conj().transpose()
    return 2 * grad_h


def _H(x,y):
    # evaluates (x+y) log(x/y) / (x-y)
    ratio = x/y
    if abs(ratio - 1) < 0.1:
        # numerically stable near np.log(ratio) ~ 0
        return (ratio + 1) / exprel(np.log(ratio))
    else:
        return (ratio + 1) * np.log(ratio) / (ratio - 1)


def distsq_hessian_herm(P_eig):
    eigvals, eigvecs = P_eig
    eigvecs_inv = eigvecs.conj().transpose()
    n = eigvals.shape[0]

    H = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            factor = _H(eigvals[i], eigvals[j])
            H[i,j] = factor
            H[j,i] = factor

    hessian = np.zeros((n**2, n**2))
    for k in range(n):
        for l in range(n):
            H_kl = elementary_hermitian_matrix(n,k,l)
            output = eigvecs @ np.multiply(H, eigvecs_inv @ H_kl @ eigvecs) @ eigvecs_inv
            for k_ in range(n):
                for l_ in range(n):
                    inner_prod = np.vdot(elementary_hermitian_matrix(n,k_,l_), output)
                    hessian[k_*n+l_,k*n+l] = np.real(inner_prod)

    return hessian
