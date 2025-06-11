#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
VANILLA INFERENCE ALGORITHM (Fixed Point Iteration)

This module implements the VANILLA algorithm for Active Inference - a fast, efficient
method for updating beliefs about hidden states based on observations. VANILLA uses
Fixed Point Iteration (FPI) to solve the Bayesian inference problem.

ALGORITHM OVERVIEW:
==================

The VANILLA algorithm answers the question: "Given what I just observed, what hidden
states am I most likely in?" It uses variational Bayes with mean-field approximation
to efficiently compute posterior beliefs.

KEY CONCEPTS:
=============

1. MEAN-FIELD APPROXIMATION:
   - Assumes independence between different state factors
   - Factorizes joint posterior: q(s1,s2,...) ≈ q(s1) × q(s2) × ...
   - Enables efficient computation for high-dimensional state spaces

2. FIXED POINT ITERATION:
   - Iteratively updates beliefs until convergence
   - Each factor's beliefs updated using current beliefs about other factors  
   - Converges to optimal factorized posterior

3. VARIATIONAL FREE ENERGY:
   - Measures quality of the approximation
   - Convergence criterion: stop when free energy change < threshold
   - Balances accuracy (fit to data) vs complexity (divergence from prior)

MATHEMATICAL FOUNDATION:
=======================

For each state factor f:
q(s_f) ∝ exp(log_likelihood_f + log_prior_f)

Where log_likelihood_f comes from marginalizing the joint likelihood over other factors:
log_likelihood_f = Σ q(s_-f) × log P(obs | s_f, s_-f)

The algorithm iterates these updates until the posterior converges.

WHEN TO USE VANILLA:
===================

- Reactive behavior (single timestep inference)
- Fast response required
- Simple environments without complex temporal dependencies
- When computational efficiency is prioritized over planning capability

COMPARISON TO MMP:
- VANILLA: Fast, reactive, single timestep
- MMP: Slower, planning, multiple timesteps with temporal dependencies
"""

import numpy as np
from pymdp.maths import spm_dot, dot_likelihood, get_joint_likelihood, softmax, calc_free_energy, spm_log_single, spm_log_obj_array
from pymdp.utils import to_obj_array, obj_array, obj_array_uniform
from itertools import chain
from copy import deepcopy

def run_vanilla_fpi(A, obs, num_obs, num_states, prior=None, num_iter=10, dF=1.0, dF_tol=0.001, compute_vfe=True):
    """
    VANILLA INFERENCE: Fixed Point Iteration for Bayesian State Estimation
    
    This is the core implementation of the VANILLA algorithm for Active Inference.
    It uses Fixed Point Iteration to compute posterior beliefs about hidden states
    given observations, efficiently handling multi-factor state spaces.
    
    THE ALGORITHM IN STEPS:
    1. Compute joint likelihood from all observation modalities
    2. Initialize uniform posterior beliefs for all state factors
    3. Iteratively update each factor's beliefs using current beliefs about other factors
    4. Continue until convergence (free energy change below threshold)
    5. Return converged posterior beliefs
    
    THE BASIC IDEA:
    Given observation: "I see bright light and hear silence"
    Prior beliefs: "I was probably in the kitchen"
    Observation model: "Bright light is common in bright rooms, silence is common everywhere"
    
    Algorithm computes: "Given this observation, I'm most likely in a bright room"
    
    MATHEMATICAL FOUNDATION:
    For each iteration and each state factor f:
    q_new(s_f) ∝ exp(Σ q(s_other) × log P(obs | s_f, s_other) + log P(s_f))
    
    Where:
    - q(s_f) are beliefs about state factor f
    - P(obs | s_f, s_other) comes from the A matrix
    - P(s_f) is the prior for factor f
    
    CONVERGENCE:
    The algorithm monitors variational free energy:
    F = Accuracy + Complexity
    
    And stops when |F_new - F_old| < threshold, indicating convergence.
    
    WHEN TO USE:
    - Single timestep inference (reactive behavior)
    - Fast response needed
    - Current observation → current state beliefs
    - No planning or temporal dependencies required

    Parameters
    ----------
    A : numpy.ndarray of dtype object
        Observation model: A[m][obs, state1, state2, ...] = P(observation | states)
        Maps from hidden states to observations for each modality.
    obs : numpy.ndarray or object array
        Current observation from the environment.
        Single modality: 1D one-hot vector
        Multi-modality: object array of 1D one-hot vectors
    num_obs : list of int
        Number of possible observations for each modality.
        Example: [4, 3] = 4 visual observations, 3 auditory observations
    num_states : list of int
        Number of possible states for each hidden state factor.
        Example: [5, 2] = 5 rooms, 2 switch positions
    prior : numpy.ndarray of dtype object, optional
        Prior beliefs about hidden states. If None, uses uniform priors.
        Usually comes from D vector (initial beliefs) or previous timestep.
    num_iter : int, default 10
        Maximum number of fixed-point iterations before stopping.
        Higher values allow more time for convergence but cost more computation.
    dF : float, default 1.0
        Initial free energy gradient (used internally for convergence detection).
    dF_tol : float, default 0.001
        Convergence threshold: stop when free energy change < dF_tol.
        Smaller values = more precise convergence, more iterations.
    compute_vfe : bool, default True
        Whether to compute variational free energy for convergence checking.
        If False, runs for exactly num_iter iterations.
  
    Returns
    ----------
    qs : numpy.ndarray or object array
        Posterior beliefs about hidden states after convergence.
        qs[f] = probability distribution over states for factor f.
        These represent the agent's updated beliefs about current hidden states.
    """

    # get model dimensions
    n_modalities = len(num_obs)
    n_factors = len(num_states)

    """
    =========== Step 1 ===========
        Loop over the observation modalities and use assumption of independence 
        among observation modalitiesto multiply each modality-specific likelihood 
        onto a single joint likelihood over hidden factors [size num_states]
    """

    likelihood = get_joint_likelihood(A, obs, num_states)

    likelihood = spm_log_single(likelihood)

    """
    =========== Step 2 ===========
        Create a flat posterior (and prior if necessary)
    """

    qs = obj_array_uniform(num_states)

    """
    If prior is not provided, initialise prior to be identical to posterior 
    (namely, a flat categorical distribution). Take the logarithm of it (required for 
    FPI algorithm below).
    """
    if prior is None:
        prior = obj_array_uniform(num_states)
        
    prior = spm_log_obj_array(prior) # log the prior


    """
    =========== Step 3 ===========
        Initialize initial free energy
    """
    if compute_vfe:
        prev_vfe = calc_free_energy(qs, prior, n_factors)

    """
    =========== Step 4 ===========
        If we have a single factor, we can just add prior and likelihood because there is a unique FE minimum that can reached instantaneously,
        otherwise we run fixed point iteration
    """

    if n_factors == 1:

        qL = spm_dot(likelihood, qs, [0])

        return to_obj_array(softmax(qL + prior[0]))

    else:
        """
        =========== Step 5 ===========
        Run the FPI scheme
        """

        # change stop condition for fixed point iterations based on whether we are computing the variational free energy or not
        condition_check_both = lambda curr_iter, dF: curr_iter < num_iter and dF >= dF_tol
        condition_check_just_numiter = lambda curr_iter, dF: curr_iter < num_iter
        check_stop_condition = condition_check_both if compute_vfe else condition_check_just_numiter

        curr_iter = 0

        while check_stop_condition(curr_iter, dF):
            # Initialise variational free energy
            vfe = 0

            # arg_list = [likelihood, list(range(n_factors))]
            # arg_list = arg_list + list(chain(*([qs_i,[i]] for i, qs_i in enumerate(qs)))) + [list(range(n_factors))]
            # LL_tensor = np.einsum(*arg_list)

            qs_all = qs[0]
            for factor in range(n_factors-1):
                qs_all = qs_all[...,None]*qs[factor+1]
            LL_tensor = likelihood * qs_all

            for factor, qs_i in enumerate(qs):
                # qL = np.einsum(LL_tensor, list(range(n_factors)), 1.0/qs_i, [factor], [factor])
                qL = np.einsum(LL_tensor, list(range(n_factors)), [factor])/qs_i
                qs[factor] = softmax(qL + prior[factor])

            # print(f'Posteriors at iteration {curr_iter}:\n')
            # print(qs[0])
            # print(qs[1])
            # List of orders in which marginal posteriors are sequentially multiplied into the joint likelihood:
            # First order loops over factors starting at index = 0, second order goes in reverse
            # factor_orders = [range(n_factors), range((n_factors - 1), -1, -1)]

            # iteratively marginalize out each posterior marginal from the joint log-likelihood
            # except for the one associated with a given factor
            # for factor_order in factor_orders:
            #     for factor in factor_order:
            #         qL = spm_dot(likelihood, qs, [factor])
            #         qs[factor] = softmax(qL + prior[factor])

            if compute_vfe:
                # calculate new free energy
                vfe = calc_free_energy(qs, prior, n_factors, likelihood)

                # print(f'VFE at iteration {curr_iter}: {vfe}\n')
                # stopping condition - time derivative of free energy
                dF = np.abs(prev_vfe - vfe)
                prev_vfe = vfe

            curr_iter += 1

        return qs

def run_vanilla_fpi_factorized(A, obs, num_obs, num_states, mb_dict, prior=None, num_iter=10, dF=1.0, dF_tol=0.001, compute_vfe=True):
    """
    Update marginal posterior beliefs over hidden states using mean-field variational inference, via
    fixed point iteration. 

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``np.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs: numpy 1D array or numpy ndarray of dtype object
        The observation (generated by the environment). If single modality, this should be a 1D ``np.ndarray``
        (one-hot vector representation). If multi-modality, this should be ``np.ndarray`` of dtype object whose entries are 1D one-hot vectors.
    num_obs: ``list`` of ints
        List of dimensionalities of each observation modality
    num_states: ``list`` of ints
        List of dimensionalities of each hidden state factor
    mb_dict: ``Dict``
        Dictionary with two keys (``A_factor_list`` and ``A_modality_list``), that stores the factor indices that influence each modality (``A_factor_list``)
        and the modality indices influenced by each factor (``A_modality_list``).
    prior: numpy ndarray of dtype object, default None
        Prior over hidden states. If absent, prior is set to be the log uniform distribution over hidden states (identical to the 
        initialisation of the posterior)
    num_iter: int, default 10
        Number of variational fixed-point iterations to run until convergence.
    dF: float, default 1.0
        Initial free energy gradient (dF/dt) before updating in the course of gradient descent.
    dF_tol: float, default 0.001
        Threshold value of the time derivative of the variational free energy (dF/dt), to be checked at 
        each iteration. If dF <= dF_tol, the iterations are halted pre-emptively and the final 
        marginal posterior belief(s) is(are) returned
    compute_vfe: bool, default True
        Whether to compute the variational free energy at each iteration. If False, the function runs through 
        all variational iterations.
  
    Returns
    ----------
    qs: numpy 1D array, numpy ndarray of dtype object, optional
        Marginal posterior beliefs over hidden states at current timepoint
    """

    # get model dimensions
    n_modalities = len(num_obs)
    n_factors = len(num_states)

    """
    =========== Step 1 ===========
        Generate modality-specific log-likelihood tensors (will be tensors of different-shapes,
        where `likelihood[m].ndim` will be equal to  `len(mb_dict['A_factor_list'][m])`
    """

    likelihood = obj_array(n_modalities)
    obs = to_obj_array(obs)
    for (m, A_m) in enumerate(A):
        likelihood[m] = dot_likelihood(A_m, obs[m])

    log_likelihood = spm_log_obj_array(likelihood)

    """
    =========== Step 2 ===========
        Create a flat posterior (and prior if necessary)
    """

    qs = obj_array_uniform(num_states)

    """
    If prior is not provided, initialise prior to be identical to posterior 
    (namely, a flat categorical distribution). Take the logarithm of it (required for 
    FPI algorithm below).
    """
    if prior is None:
        prior = obj_array_uniform(num_states)
        
    prior = spm_log_obj_array(prior) # log the prior


    """
    =========== Step 3 ===========
        Initialize initial free energy
    """
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    """
    =========== Step 4 ===========
        If we have a single factor, we can just add prior and likelihood because there is a unique FE minimum that can reached instantaneously,
        otherwise we run fixed point iteration
    """

    if n_factors == 1:

        joint_loglikelihood = np.zeros(tuple(num_states))
        for m in range(n_modalities):
            joint_loglikelihood += log_likelihood[m] # add up all the log-likelihoods, since we know they will all have the same dimension in the case of a single hidden state factor
        qL = spm_dot(joint_loglikelihood, qs, [0])

        qs = to_obj_array(softmax(qL + prior[0]))

    else:
        """
        =========== Step 5 ===========
        Run the factorized FPI scheme
        """

        A_factor_list, A_modality_list = mb_dict['A_factor_list'], mb_dict['A_modality_list']

        if compute_vfe:
            joint_loglikelihood = np.zeros(tuple(num_states))
            for m in range(n_modalities):
                reshape_dims = n_factors*[1]
                for _f_id in A_factor_list[m]:
                    reshape_dims[_f_id] = num_states[_f_id]

                joint_loglikelihood += log_likelihood[m].reshape(reshape_dims) # add up all the log-likelihoods after reshaping them to the global common dimensions of all hidden state factors

        curr_iter = 0

        # change stop condition for fixed point iterations based on whether we are computing the variational free energy or not
        condition_check_both = lambda curr_iter, dF: curr_iter < num_iter and dF >= dF_tol
        condition_check_just_numiter = lambda curr_iter, dF: curr_iter < num_iter
        check_stop_condition = condition_check_both if compute_vfe else condition_check_just_numiter

        while check_stop_condition(curr_iter, dF):
            
            # vfe = 0 

            qs_new = obj_array(n_factors)
            for f in range(n_factors):
            
                '''
                Sum the expected log likelihoods E_q(s_i/f)[ln P(o=obs[m]|s)] for independent modalities together,
                since they may have differing dimension. This obtains a marginal log-likelihood for the current factor index `factor`,
                which includes the evidence for that particular factor afforded by the different modalities. 
                '''
            
                qL = np.zeros(num_states[f])

                for ii, m in enumerate(A_modality_list[f]):
                
                    qL += spm_dot(log_likelihood[m], qs[A_factor_list[m]], [A_factor_list[m].index(f)])

                qs_new[f] = softmax(qL + prior[f])

                # vfe -= qL.sum() # accuracy part of vfe, sum of factor-level expected energies E_q(s_i/f)[ln P(o=obs|s)]
            
            qs = deepcopy(qs_new)
            # print(f'Posteriors at iteration {curr_iter}:\n')
            # print(qs[0])
            # print(qs[1])
            # calculate new free energy, leaving out the accuracy term
            # vfe += calc_free_energy(qs, prior, n_factors)

            if compute_vfe:
                vfe = calc_free_energy(qs, prior, n_factors, likelihood=joint_loglikelihood)

                # print(f'VFE at iteration {curr_iter}: {vfe}\n')
                # stopping condition - time derivative of free energy
                dF = np.abs(prev_vfe - vfe)
                prev_vfe = vfe

            curr_iter += 1
            
    return qs


def _run_vanilla_fpi_faster(A, obs, n_observations, n_states, prior=None, num_iter=10, dF=1.0, dF_tol=0.001):
    """
    Update marginal posterior beliefs about hidden states
    using a new version of variational fixed point iteration (FPI). 
    @NOTE (Conor, 26.02.2020):
    This method uses a faster algorithm than the traditional 'spm_dot' approach. Instead of
    separately computing a conditional joint log likelihood of an outcome, under the
    posterior probabilities of a certain marginal, instead all marginals are multiplied into one 
    joint tensor that gives the joint likelihood of an observation under all hidden states, 
    that is then sequentially (and *parallelizably*) marginalized out to get each marginal posterior. 
    This method is less RAM-intensive, admits heavy parallelization, and runs (about 2x) faster.
    @NOTE (Conor, 28.02.2020):
    After further testing, discovered interesting differences between this version and the 
    original version. It appears that the
    original version (simple 'run_vanilla_fpi') shows mean-field biases or 'explaining away' 
    effects, whereas this version spreads probabilities more 'fairly' among possibilities.
    To summarize: it actually matters what order you do the summing across the joint likelihood tensor. 
    In this verison, all marginals are multiplied into the likelihood tensor before summing out, 
    whereas in the previous version, marginals are recursively multiplied and summed out.
    @NOTE (Conor, 24.04.2020): I would expect that the factor_order approach used above would help 
    ameliorate the effects of the mean-field bias. I would also expect that the use of a factor_order 
    below is unnnecessary, since the marginalisation w.r.t. each factor is done only after all marginals 
    are multiplied into the larger tensor.

    Parameters
    ----------
    - 'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, 
        given the observation
    - 'obs' [numpy 1D array or array of arrays (with 1D numpy array entries)]:
        The observation (generated by the environment). If single modality, this can be a 1D array 
        (one-hot vector representation). If multi-modality, this can be an array of arrays 
        (whose entries are 1D one-hot vectors).
    - 'n_observations' [int or list of ints]
    - 'n_states' [int or list of ints]
    - 'prior' [numpy 1D array, array of arrays (with 1D numpy array entries) or None]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior. 
        If absent, prior is set to be a uniform distribution over hidden states 
        (identical to the initialisation of the posterior)
    -'num_iter' [int]:
        Number of variational fixed-point iterations to run.
    -'dF' [float]:
        Starting free energy gradient (dF/dt) before updating in the course of gradient descent.
    -'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dt), 
        to be checked at each iteration. If dF <= dF_tol, the iterations are halted pre-emptively 
        and the final marginal posterior belief(s) is(are) returned
    Returns
    ----------
    -'qs' [numpy 1D array or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved 
        via variational fixed point iteration (mean-field)
    """

    # get model dimensions
    n_modalities = len(n_observations)
    n_factors = len(n_states)

    """
    =========== Step 1 ===========
        Loop over the observation modalities and use assumption of independence 
        among observation modalities to multiply each modality-specific likelihood 
        onto a single joint likelihood over hidden factors [size n_states]
    """

    # likelihood = np.ones(tuple(n_states))

    # if n_modalities is 1:
    #     likelihood *= spm_dot(A, obs, obs_mode=True)
    # else:
    #     for modality in range(n_modalities):
    #         likelihood *= spm_dot(A[modality], obs[modality], obs_mode=True)
    likelihood = get_joint_likelihood(A, obs, n_states)
    likelihood = np.log(likelihood + 1e-16)

    """
    =========== Step 2 ===========
        Create a flat posterior (and prior if necessary)
    """

    qs = np.empty(n_factors, dtype=object)
    for factor in range(n_factors):
        qs[factor] = np.ones(n_states[factor]) / n_states[factor]

    """
    If prior is not provided, initialise prior to be identical to posterior 
    (namely, a flat categorical distribution). Take the logarithm of it 
    (required for FPI algorithm below).
    """
    if prior is None:
        prior = np.empty(n_factors, dtype=object)
        for factor in range(n_factors):
            prior[factor] = np.log(np.ones(n_states[factor]) / n_states[factor] + 1e-16)

    """
    =========== Step 3 ===========
        Initialize initial free energy
    """
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    """
    =========== Step 4 ===========
        If we have a single factor, we can just add prior and likelihood,
        otherwise we run FPI
    """

    if n_factors == 1:
        qL = spm_dot(likelihood, qs, [0])
        return softmax(qL + prior[0])

    else:
        """
        =========== Step 5 ===========
        Run the revised fixed-point iteration scheme
        """

        curr_iter = 0

        while curr_iter < num_iter and dF >= dF_tol:
            # Initialise variational free energy
            vfe = 0

            # List of orders in which marginal posteriors are sequentially 
            # multiplied into the joint likelihood: First order loops over 
            # factors starting at index = 0, second order goes in reverse
            factor_orders = [range(n_factors), range((n_factors - 1), -1, -1)]

            for factor_order in factor_orders:
                # reset the log likelihood
                L = likelihood.copy()

                # multiply each marginal onto a growing single joint distribution
                for factor in factor_order:
                    s = np.ones(np.ndim(L), dtype=int)
                    s[factor] = len(qs[factor])
                    L *= qs[factor].reshape(tuple(s))

                # now loop over factors again, and this time divide out the 
                # appropriate marginal before summing out.
                # !!! KEY DIFFERENCE BETWEEN THIS AND 'VANILLA' FPI, 
                # WHERE THE ORDER OF THE MARGINALIZATION MATTERS !!!
                for f in factor_order:
                    s = np.ones(np.ndim(L), dtype=int)
                    s[factor] = len(qs[factor])  # type: ignore

                    # divide out the factor we multiplied into X already
                    temp = L * (1.0 / qs[factor]).reshape(tuple(s))  # type: ignore
                    dims2sum = tuple(np.where(np.arange(n_factors) != f)[0])
                    qL = np.sum(temp, dims2sum)

                    temp = L * (1.0 / qs[factor]).reshape(tuple(s))  # type: ignore
                    qs[factor] = softmax(qL + prior[factor])  # type: ignore

            # calculate new free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # stopping condition - time derivative of free energy
            dF = np.abs(prev_vfe - vfe)
            prev_vfe = vfe

            curr_iter += 1

        return qs
