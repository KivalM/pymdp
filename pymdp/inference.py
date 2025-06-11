#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
ACTIVE INFERENCE STATE INFERENCE MODULE

This module contains the core state inference algorithms for Active Inference agents.
These functions implement the "perception" part of the perception-action loop, where
agents update their beliefs about hidden world states based on observations.

KEY CONCEPTS:
=============

1. STATE INFERENCE:
   - Hidden states are what's really happening in the world (true state)
   - Observations are what the agent can sense (noisy/partial information)
   - Inference uses Bayes' rule to estimate hidden states from observations

2. INFERENCE ALGORITHMS:
   - VANILLA: Simple, fast inference for reactive behavior
   - MMP (Marginal Message Passing): Complex temporal inference for planning
   - VMP (Variational Message Passing): Advanced variational methods
   
3. FACTORIZATION:
   - Computational optimization for complex state spaces
   - Breaks down large state spaces into manageable factors
   - Enables efficient inference in high-dimensional problems

MATHEMATICAL FOUNDATION:
=======================

Bayes' Rule for State Inference:
P(hidden_state | observation) ∝ P(observation | hidden_state) × P(hidden_state)

Where:
- P(observation | hidden_state) comes from the A matrix (observation model)
- P(hidden_state) is the prior belief (from D vector or previous timestep)
- P(hidden_state | observation) is what we want to compute (posterior belief)

For temporal sequences:
P(states_sequence | observations_sequence) using message passing algorithms

The agent seeks to minimize variational free energy:
F = Accuracy + Complexity
Where Accuracy = how well beliefs explain observations
And Complexity = how much beliefs deviate from priors
"""

import numpy as np

from pymdp import utils
from pymdp.maths import get_joint_likelihood_seq, get_joint_likelihood_seq_by_modality
from pymdp.algos import run_vanilla_fpi, run_vanilla_fpi_factorized, run_mmp, run_mmp_factorized, _run_mmp_testing

VANILLA = "VANILLA"
VMP = "VMP"
MMP = "MMP"
BP = "BP"
EP = "EP"
CV = "CV"

def update_posterior_states_full(
    A,
    B,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior = True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing
    Posterior P(s_t | o_1:t) = P(o_1:t | s_t) P(s_t) / P(o_1:t)
    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    prev_obs: ``list``
        List of observations over time. Each observation in the list can be an ``int``, a ``list`` of ints, a ``tuple`` of ints, a one-hot vector or an object array of one-hot vectors.
    policies: ``list`` of 2D ``numpy.ndarray``
        List that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    prior: ``numpy.ndarray`` of dtype object, default ``None``
        If provided, this a ``numpy.ndarray`` of dtype object, with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    policy_sep_prior: ``Bool``, default ``True``
        Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs: keyword arguments
        Optional keyword arguments for the function ``algos.mmp.run_mmp``

    Returns
    ---------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    F: 1D ``numpy.ndarray``
        Vector of variational free energies for each policy
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    
    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
   
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    for p_idx, policy in enumerate(policies):

            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx] = run_mmp(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior= prior[p_idx] if policy_sep_prior else prior, 
                **kwargs
            )

    return qs_seq_pi, F

def update_posterior_states_full_factorized(
    A,
    mb_dict,
    B,
    B_factor_list,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior = True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    mb_dict: ``Dict``
        Dictionary with two keys (``A_factor_list`` and ``A_modality_list``), that stores the factor indices that influence each modality (``A_factor_list``)
        and the modality indices influenced by each factor (``A_modality_list``).
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    B_factor_list: ``list`` of ``list`` of ``int``
        List of lists of hidden state factors each hidden state factor depends on. Each element ``B_factor_list[i]`` is a list of the factor indices that factor i's dynamics depend on.
    prev_obs: ``list``
        List of observations over time. Each observation in the list can be an ``int``, a ``list`` of ints, a ``tuple`` of ints, a one-hot vector or an object array of one-hot vectors.
    policies: ``list`` of 2D ``numpy.ndarray``
        List that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    prior: ``numpy.ndarray`` of dtype object, default ``None``
        If provided, this a ``numpy.ndarray`` of dtype object, with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    policy_sep_prior: ``Bool``, default ``True``
        Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs: keyword arguments
        Optional keyword arguments for the function ``algos.mmp.run_mmp``

    Returns
    ---------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    F: 1D ``numpy.ndarray``
        Vector of variational free energies for each policy
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    
    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
   
    lh_seq = get_joint_likelihood_seq_by_modality(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    for p_idx, policy in enumerate(policies):

            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx] = run_mmp_factorized(
                lh_seq,
                mb_dict,
                B,
                B_factor_list,
                policy,
                prev_actions=prev_actions,
                prior= prior[p_idx] if policy_sep_prior else prior, 
                **kwargs
            )

    return qs_seq_pi, F

def _update_posterior_states_full_test(
    A,
    B,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior = True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing (TEST VERSION, with extra returns for benchmarking).

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``np.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    prev_obs: list
        List of observations over time. Each observation in the list can be an ``int``, a ``list`` of ints, a ``tuple`` of ints, a one-hot vector or an object array of one-hot vectors.
    prior: ``numpy.ndarray`` of dtype object, default None
        If provided, this a ``numpy.ndarray`` of dtype object, with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    policy_sep_prior: Bool, default True
        Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs: keyword arguments
        Optional keyword arguments for the function ``algos.mmp.run_mmp``

    Returns
    --------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    F: 1D ``numpy.ndarray``
        Vector of variational free energies for each policy
    xn_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy, for each iteration of marginal message passing.
        Nesting structure is policy, iteration, factor, so ``xn_seq_p[p][itr][f]`` stores the ``num_states x infer_len`` 
        array of beliefs about hidden states at different time points of inference horizon.
    vn_seq_pi: `numpy.ndarray`` of dtype object
        Prediction errors over hidden states for each policy, for each iteration of marginal message passing.
        Nesting structure is policy, iteration, factor, so ``vn_seq_p[p][itr][f]`` stores the ``num_states x infer_len`` 
        array of beliefs about hidden states at different time points of inference horizon.
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)

    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
    
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    xn_seq_pi = utils.obj_array(len(policies))
    vn_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    for p_idx, policy in enumerate(policies):

            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx], xn_seq_pi[p_idx], vn_seq_pi[p_idx] = _run_mmp_testing(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior[p_idx] if policy_sep_prior else prior, 
                **kwargs
            )

    return qs_seq_pi, F, xn_seq_pi, vn_seq_pi

def average_states_over_policies(qs_pi, q_pi):
    """
    BAYESIAN MODEL AVERAGING: Integrating State Beliefs Across Policies
    
    This function combines state beliefs from different policies into a single unified belief.
    It's used when the agent has uncertainty about which policy is best and wants to
    compute overall state beliefs that account for this policy uncertainty.
    
    THE BASIC IDEA:
    When the agent is considering multiple policies, each policy leads to different
    beliefs about hidden states. Instead of picking one policy and ignoring the others,
    Bayesian Model Averaging (BMA) combines all the state beliefs, weighted by how
    much the agent believes in each policy.
    
    MATHEMATICAL FOUNDATION:
    For each state factor f:
    qs_bma[f] = Σ P(policy π) × qs[π][f]
    
    Where:
    - P(policy π) is the agent's belief that policy π is optimal
    - qs[π][f] is the belief about state factor f under policy π
    - The sum is over all policies
    
    WHEN TO USE:
    - MMP algorithm (planning with multiple policies)
    - When agent has uncertainty about which policy is best
    - For computing policy-independent state beliefs
    - At boundaries of inference horizons
    
    EXAMPLE:
    Policy preferences: [0.6, 0.4] for ["go-right", "go-left"]
    State beliefs under go-right: "70% chance in room A"
    State beliefs under go-left: "30% chance in room A"
    BMA result: 0.6×0.7 + 0.4×0.3 = 54% chance in room A
    (Weighted average that accounts for policy uncertainty)

    Parameters
    ----------
    qs_pi: ``numpy.ndarray`` of dtype object
        Policy-conditioned state beliefs: qs_pi[p][f] = belief about factor f under policy p.
        This is a nested array where each policy has its own set of state beliefs.
    q_pi: ``numpy.ndarray`` of dtype object
        Policy preferences: q_pi[p] = probability that policy p is optimal.
        These weights determine how much each policy contributes to the average.
        Should sum to 1 (proper probability distribution).

    Returns
    ---------
    qs_bma: ``numpy.ndarray`` of dtype object
        Bayesian model averaged state beliefs: qs_bma[f] = integrated belief about factor f.
        This represents the agent's overall state beliefs after accounting for policy uncertainty.
        Same structure as individual policy beliefs but averaged across all policies.
    """

    # Get model dimensions from the first policy's beliefs
    num_factors = len(qs_pi[0])  # Number of hidden state factors
    num_states = [qs_f.shape[0] for qs_f in qs_pi[0]]  # States per factor

    # Initialize Bayesian model averaged beliefs
    qs_bma = utils.obj_array(num_factors)
    for f in range(num_factors):
        qs_bma[f] = np.zeros(num_states[f])

    ### Compute Weighted Average Across Policies ###
    for p_idx, policy_weight in enumerate(q_pi):
        for f in range(num_factors):
            # Add this policy's contribution, weighted by its probability
            qs_bma[f] += qs_pi[p_idx][f] * policy_weight

    return qs_bma

def update_posterior_states(A, obs, prior=None, **kwargs):
    """
    VANILLA STATE INFERENCE: Simple Bayesian Perception
    
    This is the core "perception" function for Active Inference agents using the VANILLA algorithm.
    It implements Bayes' rule to update beliefs about hidden states based on a single observation.
    This is like asking: "Given what I just observed, what's most likely happening in the world?"
    
    THE BASIC IDEA:
    The agent doesn't directly know the true hidden states (like room location, object positions).
    It only gets observations (what it can see, hear, feel). This function uses Bayes' rule
    to combine:
    1. Prior beliefs (what the agent expected before seeing anything)
    2. Observation likelihood (how likely this observation is given different states)
    → Posterior beliefs (updated beliefs about what's actually happening)
    
    MATHEMATICAL FOUNDATION:
    Bayes' rule: P(state | observation) ∝ P(observation | state) × P(state)
    
    Where:
    - P(observation | state) comes from the A matrix (observation model)
    - P(state) is the prior belief (from D vector or previous inference)
    - P(state | observation) is the posterior we want to compute
    
    ALGORITHM: Fixed Point Iteration (FPI)
    Uses mean-field variational inference with iterative updates until convergence.
    This efficiently handles cases where state factors interact.
    
    WHEN TO USE:
    - VANILLA algorithm (simple, reactive behavior)
    - Single timestep inference (not planning ahead)
    - When you have current observation and want current state beliefs
    
    EXAMPLE:
    - Agent observes: "bright light" 
    - Prior belief: "probably in dark room"
    - A matrix says: "bright rooms usually produce bright light observations"
    - Posterior: "probably in bright room" (belief updated by observation)

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Observation model: A[m][obs, state1, state2, ...] = P(observation | states)
        Tells the agent what observations to expect in different hidden states.
    obs: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, int or tuple
        The observation that was just received from the environment.
        Can be:
        - Single modality: int (observation index) or 1D array (one-hot vector)
        - Multi-modality: tuple of ints or object array of one-hot vectors
        Example: [0, 2] = "observation 0 for vision, observation 2 for sound"
    prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object, default None
        Prior beliefs about hidden states before seeing the observation.
        If None, uses uniform (flat) priors (all states equally likely).
        Usually comes from D vector (initial beliefs) or previous timestep.
    **kwargs: keyword arguments 
        Optional parameters for the fixed-point iteration algorithm:
        - num_iter: maximum number of iterations
        - dF: threshold for convergence
        - dF_tol: tolerance for free energy changes

    Returns
    ----------
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Posterior beliefs about hidden states after seeing the observation.
        qs[f] = probability distribution over states for factor f.
        These represent the agent's updated beliefs about what's happening.
    """

    # Get model dimensions
    num_obs, num_states, num_modalities, _ = utils.get_model_dimensions(A = A)
    
    # Process observation into standard format
    obs = utils.process_observation(obs, num_modalities, num_obs)

    # Process prior beliefs
    if prior is not None:
        prior = utils.to_obj_array(prior)

    # Run the VANILLA inference algorithm (Fixed Point Iteration)
    return run_vanilla_fpi(A, obs, num_obs, num_states, prior, **kwargs)

def update_posterior_states_factorized(A, obs, num_obs, num_states, mb_dict, prior=None, **kwargs):
    """
    VANILLA STATE INFERENCE: Efficient Factorized Version
    
    This is the computationally optimized version of the core state inference function.
    It uses factorization to efficiently handle complex state spaces by identifying
    which state factors actually affect which observations (Markov blankets).
    
    THE BASIC IDEA:
    Same as regular VANILLA inference, but optimized for computational efficiency.
    Instead of considering all possible state factor combinations, it identifies
    which state factors actually matter for each observation modality.
    
    FACTORIZATION OPTIMIZATION:
    In complex environments, not all state factors affect all observations:
    - Room location might affect visual observations but not internal body state
    - Hunger level might affect interoceptive observations but not visual observations
    
    By identifying these dependencies (Markov blankets), we can:
    - Reduce computational complexity from exponential to linear
    - Update only relevant state factors for each observation
    - Enable efficient inference in high-dimensional state spaces
    
    MARKOV BLANKETS:
    - A_factor_list[m] = which state factors affect observation modality m
    - A_modality_list[f] = which observation modalities are affected by state factor f
    
    WHEN TO USE:
    - VANILLA algorithm with complex state spaces
    - When different observations depend on different state factors
    - When computational efficiency is important
    
    EXAMPLE:
    State factors: [room_location, hunger_level, light_switch]
    Observations: [vision, interoception]
    Dependencies:
    - vision depends on [room_location, light_switch] 
    - interoception depends on [hunger_level]
    → Only update relevant factors for each observation type

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Observation model: A[m][obs, state1, state2, ...] = P(observation | states)
        Tells the agent what observations to expect in different hidden states.
    obs: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, int or tuple
        The observation that was just received from the environment.
        Can be single or multi-modality as described in update_posterior_states.
    num_obs: ``list`` of ``int``
        Number of possible observations for each modality.
        Example: [4, 3] = 4 visual observations, 3 auditory observations.
    num_states: ``list`` of ``int``
        Number of possible states for each hidden state factor.
        Example: [5, 3, 2] = 5 rooms, 3 hunger levels, 2 switch positions.
    mb_dict: ``Dict``
        Markov blanket dictionary with keys:
        - 'A_factor_list': which state factors affect which observation modalities
        - 'A_modality_list': which observation modalities are affected by which state factors
        This enables computational efficiency by avoiding unnecessary calculations.
    prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object, default None
        Prior beliefs about hidden states before seeing the observation.
        If None, uses uniform (flat) priors (all states equally likely).
    **kwargs: keyword arguments 
        Optional parameters for the factorized fixed-point iteration algorithm:
        - num_iter: maximum number of iterations
        - dF: threshold for convergence
        - dF_tol: tolerance for free energy changes

    Returns
    ----------
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Posterior beliefs about hidden states after seeing the observation.
        qs[f] = probability distribution over states for factor f.
        Same output as regular inference but computed more efficiently.
    """
    
    # Get number of modalities
    num_modalities = len(num_obs)
    
    # Process observation into standard format
    obs = utils.process_observation(obs, num_modalities, num_obs)

    # Process prior beliefs
    if prior is not None:
        prior = utils.to_obj_array(prior)

    # Run the factorized VANILLA inference algorithm
    return run_vanilla_fpi_factorized(A, obs, num_obs, num_states, mb_dict, prior, **kwargs)
