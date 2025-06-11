#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMP PLANNING ALGORITHM (Marginal Message Passing)

This module implements the MMP algorithm for Active Inference - a sophisticated
temporal inference method that enables planning by reasoning about sequences of
hidden states over time. MMP uses message passing to efficiently handle temporal
dependencies and policy-conditioned beliefs.

ALGORITHM OVERVIEW:
==================

The MMP algorithm answers the question: "Given my observations and this policy,
what sequence of hidden states am I most likely to experience over time?"
It combines past observations with future policy predictions to compute
temporal state sequences.

KEY CONCEPTS:
=============

1. TEMPORAL INFERENCE:
   - Reasons about state sequences over multiple timesteps
   - Combines past evidence (observations) with future predictions (policy)
   - Enables planning by evaluating consequences of action sequences

2. MESSAGE PASSING:
   - Forward messages: influence of past states on current states
   - Backward messages: influence of future states on current states  
   - Bidirectional propagation ensures temporal consistency

3. POLICY-CONDITIONED BELIEFS:
   - Computes beliefs under specific policy: "If I follow this plan..."
   - Each policy produces different predicted state sequences
   - Enables comparison of different plans

4. INFERENCE WINDOW:
   - Past window: timesteps with actual observations
   - Future window: timesteps with predicted actions
   - Combined window enables coherent temporal reasoning

MATHEMATICAL FOUNDATION:
=======================

For each timestep t and state factor f:
q(s_t^f) ∝ exp(log_likelihood_t^f + log_past_message^f + log_future_message^f)

Where:
- log_likelihood_t^f comes from observations (if available)
- log_past_message^f = influence from previous timesteps
- log_future_message^f = influence from future timesteps

Messages are computed using transition matrices:
- Forward: B[f][:, :, action] × q(s_{t-1}^f)
- Backward: B[f]^T[:, :, action] × q(s_{t+1}^f)

WHEN TO USE MMP:
===============

- Planning behavior (multi-timestep inference)
- Temporal dependencies matter
- Policy evaluation and comparison
- Complex environments requiring lookahead
- When computational cost is acceptable for better planning

COMPARISON TO VANILLA:
- MMP: Planning, multi-timestep, temporal dependencies, slower
- VANILLA: Reactive, single timestep, faster, no temporal reasoning
"""

import numpy as np

from pymdp.utils import to_obj_array, get_model_dimensions, obj_array, obj_array_zeros, obj_array_uniform
from pymdp.maths import spm_dot, spm_norm, softmax, calc_free_energy, spm_log_single, factor_dot_flex
import copy

def run_mmp(
    lh_seq, B, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=True, tau=0.25, last_timestep = False):
    """
    MMP PLANNING: Temporal State Inference for Policy Evaluation
    
    This is the core implementation of the MMP algorithm for Active Inference.
    It computes policy-conditioned beliefs about hidden state sequences over time,
    enabling the agent to evaluate the consequences of different action plans.
    
    THE ALGORITHM IN STEPS:
    1. Set up inference window (past observations + future policy)
    2. Initialize uniform beliefs for all timesteps and factors
    3. For each iteration and timestep:
       - Compute likelihood message (from observations, if available)
       - Compute past message (influence from previous timestep)  
       - Compute future message (influence from next timestep)
       - Update beliefs by combining all messages
    4. Return temporal sequence of beliefs and free energy
    
    THE BASIC IDEA:
    Given: "I observed X yesterday, Y today, and plan to do actions [A, B, C] tomorrow"
    Question: "What sequence of hidden states explains my observations and predicts my policy outcomes?"
    
    Algorithm computes: "If I follow this policy, I'll likely be in states [S1, S2, S3, S4] over time"
    
    TEMPORAL MESSAGE PASSING:
    For each timestep t and factor f:
    
    Past Message: "What states at t-1 could have led to my beliefs at t?"
    Future Message: "What states at t+1 do my beliefs at t predict?"
    Likelihood: "What states at t best explain observation at t?"
    
    Belief Update: q(s_t) ∝ exp(likelihood + past_msg + future_msg)
    
    WHY MESSAGE PASSING:
    - Forward-backward propagation ensures temporal consistency
    - Past constrains present based on observations
    - Future constrains present based on planned actions
    - Bidirectional flow creates coherent temporal sequences
    
    WHEN TO USE:
    - Evaluating specific policies: "What happens if I follow this plan?"
    - Planning algorithms that compare different action sequences
    - Environments where temporal dependencies matter
    - When you need state predictions for future timesteps

    Parameters
    ----------
    lh_seq : numpy.ndarray of dtype object
        Sequence of log-likelihoods for observations over time.
        lh_seq[t] = log P(observation_t | states) for timestep t.
        Only available for past timesteps (where observations exist).
    B : numpy.ndarray of dtype object
        Transition model: B[f][next_state, prev_state, action] = P(next_state | prev_state, action)
        Describes how actions change states over time for each factor f.
    policy : numpy.ndarray, shape (policy_len, num_control_factors)
        Action sequence to evaluate: policy[t, f] = action for factor f at timestep t.
        This defines the "plan" whose consequences we want to predict.
    prev_actions : numpy.ndarray, optional
        Previous actions already taken, shape (past_len, num_control_factors).
        Used to connect past observations with future policy.
    prior : numpy.ndarray of dtype object, optional
        Prior beliefs about initial states. If None, uses uniform priors.
        Usually comes from D vector or previous inference results.
    num_iter : int, default 10
        Number of message passing iterations for convergence.
        More iterations = better convergence but higher computational cost.
    grad_descent : bool, default True
        Whether to use gradient descent updates (True) or fixed-point updates (False).
        Gradient descent often more stable but slower convergence.
    tau : float, default 0.25
        Learning rate for gradient descent updates (if grad_descent=True).
        Controls step size: smaller values = more stable, larger = faster.
    last_timestep : bool, default False
        Whether this is the final timestep of planning.
        Affects inference window construction for computational efficiency.
        
    Returns
    ---------
    qs_seq : numpy.ndarray of dtype object
        Policy-conditioned state beliefs over time.
        qs_seq[t][f] = beliefs about factor f at timestep t under this policy.
        This represents the predicted temporal sequence of hidden states.
    F : float
        Variational free energy of the policy.
        Lower F = better policy (more likely state sequence given observations).
        Used for policy comparison and selection.
    """

    # window
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len
    
    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
        
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f],0,1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    for itr in range(num_iter):
        F = 0.0 # reset variational free energy (accumulated over time and factors, but reset per iteration)
        for t in range(infer_len):
            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))
                    print(f'Enumerated version: lnA at time {t}: {lnA}')    
                else:
                    lnA = np.zeros(num_states[f])
                
                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log_single(future_msg)
                
                # inference
                if grad_descent:
                    sx = qs_seq[t][f] # save this as a separate variable so that it can be used in VFE computation
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    lnqs = lnqs + tau * (err - err.mean()) # for numerical stability, before passing into the softmax
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += sx.dot(0.5*err)
                    else:
                        F += sx.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log_single(lh_seq[t]) )
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)

    return qs_seq, F

def run_mmp_factorized(
    lh_seq, mb_dict, B, B_factor_list, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=True, tau=0.25, last_timestep = False):
    """
    Marginal message passing scheme for updating marginal posterior beliefs about hidden states over time, 
    conditioned on a particular policy.

    Parameters
    ----------
    lh_seq: ``numpy.ndarray`` of dtype object
        Log likelihoods of hidden states under a sequence of observations over time. This is assumed to already be log-transformed. Each ``lh_seq[t]`` contains
        the log likelihood of hidden states for a particular observation at time ``t``
    mb_dict: ``Dict``
        Dictionary with two keys (``A_factor_list`` and ``A_modality_list``), that stores the factor indices that influence each modality (``A_factor_list``)
        and the modality indices influenced by each factor (``A_modality_list``).
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    B_factor_list: ``list`` of ``list`` of ``int``
        List of lists of hidden state factors each hidden state factor depends on. Each element ``B_factor_list[i]`` is a list of the factor indices that factor i's dynamics depend on.
    policy: 2D ``numpy.ndarray``
        Matrix of shape ``(policy_len, num_control_factors)`` that indicates the indices of each action (control state index) upon timestep ``t`` and control_factor ``f` in the element ``policy[t,f]`` for a given policy.
    prev_actions: ``numpy.ndarray``, default None
        If provided, should be a matrix of previous actions of shape ``(infer_len, num_control_factors)`` that indicates the indices of each action (control state index) taken in the past (up until the current timestep).
    prior: ``numpy.ndarray`` of dtype object, default None
        If provided, the prior beliefs about initial states (at t = 0, relative to ``infer_len``). If ``None``, this defaults
        to a flat (uninformative) prior over hidden states.
    numiter: int, default 10
        Number of variational iterations.
    grad_descent: Bool, default True
        Flag for whether to use gradient descent (free energy gradient updates) instead of fixed point solution to the posterior beliefs
    tau: float, default 0.25
        Decay constant for use in ``grad_descent`` version. Tunes the size of the gradient descent updates to the posterior.
    last_timestep: Bool, default False
        Flag for whether we are at the last timestep of belief updating
        
    Returns
    ---------
    qs_seq: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states under the policy. Nesting structure is timepoints, factors,
        where e.g. ``qs_seq[t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under the policy in question.
    F: float
        Variational free energy of the policy.
    """

    # window
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len
    
    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
        
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f],0,1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    A_factor_list, A_modality_list = mb_dict['A_factor_list'], mb_dict['A_modality_list']

    joint_lh_seq = obj_array(len(lh_seq))
    num_modalities = len(A_factor_list)
    for t in range(len(lh_seq)):
        joint_loglikelihood = np.zeros(tuple(num_states))
        for m in range(num_modalities):
            reshape_dims = num_factors*[1]
            for _f_id in A_factor_list[m]:
                reshape_dims[_f_id] = num_states[_f_id]
            joint_loglikelihood += lh_seq[t][m].reshape(reshape_dims) # add up all the log-likelihoods after reshaping them to the global common dimensions of all hidden state factors
        joint_lh_seq[t] = joint_loglikelihood

    # compute inverse B dependencies, which is a list that for each hidden state factor, lists the indices of the other hidden state factors that it 'drives' or is a parent of in the HMM graphical model
    inv_B_deps = [[i for i, d in enumerate(B_factor_list) if f in d] for f in range(num_factors)]
    for itr in range(num_iter):
        F = 0.0 # reset variational free energy (accumulated over time and factors, but reset per iteration)
        for t in range(infer_len):
            for f in range(num_factors):
                # likelihood
                lnA = np.zeros(num_states[f])
                if t < past_len:
                    for m in A_modality_list[f]:
                        lnA += spm_log_single(spm_dot(lh_seq[t][m], qs_seq[t][A_factor_list[m]], [A_factor_list[m].index(f)]))  
                    print(f'Factorized version: lnA at time {t}: {lnA}')                
                
                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    past_msg = spm_dot(B[f][...,int(policy[t - 1, f])], qs_seq[t-1][B_factor_list[f]])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    # list of future_msgs, one for each of the factors that factor f is driving

                    B_marg_list = [] # list of the marginalized B matrices, that correspond to mapping between the factor of interest `f` and each of its children factors `i`
                    for i in inv_B_deps[f]: #loop over all the hidden state factors that are driven by f
                        b = B[i][...,int(policy[t,i])]
                        keep_dims = (0,1+B_factor_list[i].index(f))
                        dims = []
                        idxs = []
                        for j, d in enumerate(B_factor_list[i]): # loop over the list of factors that drive each child `i` of factor-of-interest `f` (i.e. the co-parents of `f`, with respect to child `i`)
                            if f != d:
                                dims.append((1 + j,))
                                idxs.append(d)
                        xs = [qs_seq[t+1][f_i] for f_i in idxs]
                        B_marg_list.append( factor_dot_flex(b, xs, tuple(dims), keep_dims=keep_dims) ) # marginalize out all parents of `i` besides `f`

                    lnB_future = np.zeros(num_states[f])
                    for i, b in enumerate(B_marg_list):
                        b_norm_T = spm_norm(b.T)
                        lnB_future += spm_log_single(b_norm_T.dot(qs_seq[t + 1][inv_B_deps[f][i]]))
                    
                    
                    lnB_future *= 0.5
                
                # inference
                if grad_descent:
                    sx = qs_seq[t][f] # save this as a separate variable so that it can be used in VFE computation
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    lnqs = lnqs + tau * (err - err.mean())
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += sx.dot(0.5*err)
                    else:
                        F += sx.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log_single(joint_lh_seq[t]) )
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)

    return qs_seq, F

def _run_mmp_testing(
    lh_seq, B, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=True, tau=0.25, last_timestep = False):
    """
    Marginal message passing scheme for updating marginal posterior beliefs about hidden states over time, 
    conditioned on a particular policy.

    Parameters
    ----------
    lh_seq: ``numpy.ndarray`` of dtype object
        Log likelihoods of hidden states under a sequence of observations over time. This is assumed to already be log-transformed. Each ``lh_seq[t]`` contains
        the log likelihood of hidden states for a particular observation at time ``t``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    policy: 2D ``numpy.ndarray``
        Matrix of shape ``(policy_len, num_control_factors)`` that indicates the indices of each action (control state index) upon timestep ``t`` and control_factor ``f` in the element ``policy[t,f]`` for a given policy.
    prev_actions: ``numpy.ndarray``, default None
        If provided, should be a matrix of previous actions of shape ``(infer_len, num_control_factors)`` that indicates the indices of each action (control state index) taken in the past (up until the current timestep).
    prior: ``numpy.ndarray`` of dtype object, default None
        If provided, the prior beliefs about initial states (at t = 0, relative to ``infer_len``). If ``None``, this defaults
        to a flat (uninformative) prior over hidden states.
    numiter: int, default 10
        Number of variational iterations.
    grad_descent: Bool, default True
        Flag for whether to use gradient descent (free energy gradient updates) instead of fixed point solution to the posterior beliefs
    tau: float, default 0.25
        Decay constant for use in ``grad_descent`` version. Tunes the size of the gradient descent updates to the posterior.
    last_timestep: Bool, default False
        Flag for whether we are at the last timestep of belief updating
        
    Returns
    ---------
    qs_seq: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states under the policy. Nesting structure is timepoints, factors,
        where e.g. ``qs_seq[t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under the policy in question.
    F: float
        Variational free energy of the policy.
    xn: list
        The sequence of beliefs as they're computed across iterations of marginal message passing (used for benchmarking). Nesting structure is iteration, factor, so ``xn[itr][f]`` 
        stores the ``num_states x infer_len`` array of beliefs about hidden states at different time points of inference horizon.
    vn: list
        The sequence of prediction errors as they're computed across iterations of marginal message passing (used for benchmarking). Nesting structure is iteration, factor, so ``vn[itr][f]`` 
        stores the ``num_states x infer_len`` array of prediction errors for hidden states at different time points of inference horizon.
    """

    # window
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len
    
    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
        
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f],0,1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    xn = [] # list for storing beliefs across iterations
    vn = [] # list for storing prediction errors across iterations

    shape_list = [ [num_states[f], infer_len] for f in range(num_factors) ]
    
    for itr in range(num_iter):

        xn_itr_all_factors = obj_array_zeros(shape_list) # temporary cache for storing beliefs across different hidden state factors, for a fixed iteration of the belief updating scheme
        vn_itr_all_factors = obj_array_zeros(shape_list) # temporary cache for storing prediction errors across different hidden state factors, for a fixed iteration of the belief updating scheme

        F = 0.0 # reset variational free energy (accumulated over time and factors, but reset per iteration)
        for t in range(infer_len):

            if t == (infer_len - 1):
                debug_flag = True

            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    # if itr == 0:
                    #     print(f'obs from timestep {t}\n')
                    lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))
                else:
                    lnA = np.zeros(num_states[f])
                
                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log_single(future_msg)

                # inference
                if grad_descent:
                    sx = qs_seq[t][f] # save this as a separate variable so that it can be used in VFE computation
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    vn_tmp = err - err.mean()
                    lnqs = lnqs + tau * vn_tmp
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += sx.dot(0.5*err)
                    else:
                        F += sx.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                    
                    xn_itr_all_factors[f][:,t] = np.copy(qs_seq[t][f])
                    vn_itr_all_factors[f][:,t] = np.copy(vn_tmp)

                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log_single(lh_seq[t]) )
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)
        xn.append(xn_itr_all_factors)
        vn.append(vn_itr_all_factors)

    return qs_seq, F, xn, vn
