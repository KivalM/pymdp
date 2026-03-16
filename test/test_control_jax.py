#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import unittest
import pytest

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import pymdp.jax.control as ctl_jax
import pymdp.control as ctl_np

from pymdp.jax.maths import factor_dot
from pymdp import utils

cfg = {"source_key": 0, "num_models": 4}

def generate_model_params():
    """
    Generate random model dimensions
    """
    rng_keys = jr.split(jr.PRNGKey(cfg["source_key"]), cfg["num_models"])
    num_factors_list = [ jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys ]
    num_states_list = [ jr.randint(key, (nf,), 1, 5).tolist() for nf, key in zip(num_factors_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], cfg["num_models"])
    num_modalities_list = [ jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys ]
    num_obs_list = [ jr.randint(key, (nm,), 1, 5).tolist() for nm, key in zip(num_modalities_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], cfg["num_models"])
    A_deps_list = []
    for nf, nm, model_key in zip(num_factors_list, num_modalities_list, rng_keys):
        keys_model_i = jr.split(model_key, nm)
        A_deps_model_i = [jr.randint(key, (nm,), 0, nf).tolist() for key in keys_model_i]
        A_deps_list.append(A_deps_model_i)
    
    return {'nf_list': num_factors_list, 
            'ns_list': num_states_list, 
            'nm_list': num_modalities_list, 
            'no_list': num_obs_list, 
            'A_deps_list': A_deps_list}

def make_simple_policy_model():
    A = [jnp.array([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32)]
    B = [jnp.stack((jnp.eye(2, dtype=jnp.float32), jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)), axis=-1)]
    C = [jnp.array([[0.2, 0.8], [0.6, 0.4]], dtype=jnp.float32)]
    pA = [1.0 + 4.0 * A[0]]
    pB = [1.0 + 4.0 * B[0]]
    qs_init = [jnp.array([0.55, 0.45], dtype=jnp.float32)]
    policies = ctl_jax.construct_policies([2], [2], policy_len=2)
    E = jnp.ones((policies.shape[0],), dtype=jnp.float32) / policies.shape[0]
    return {
        "A": A,
        "B": B,
        "C": C,
        "E": E,
        "pA": pA,
        "pB": pB,
        "qs_init": qs_init,
        "policies": policies,
        "A_dependencies": [[0]],
        "B_dependencies": [[0]],
        "I": [jnp.zeros((2, 1), dtype=jnp.float32)],
    }

class TestControlJax(unittest.TestCase):

    def test_get_expected_obs_factorized(self):
        """
        Tests the jax-ified version of computations of expected observations under some hidden states and policy
        """
        gm_params = generate_model_params()
        num_factors_list, num_states_list, num_modalities_list, num_obs_list, A_deps_list = gm_params['nf_list'], gm_params['ns_list'], gm_params['nm_list'], gm_params['no_list'], gm_params['A_deps_list']
        for (num_states, num_obs, A_deps) in zip(num_states_list, num_obs_list, A_deps_list):
            
            qs_numpy = utils.random_single_categorical(num_states)
            qs_jax = list(qs_numpy)

            A_np = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_deps)
            A_jax = jtu.tree_map(lambda x: jnp.array(x), list(A_np))   

            qo_test = ctl_jax.compute_expected_obs(qs_jax, A_jax, A_deps) 
            qo_validation = ctl_np.get_expected_obs_factorized([qs_numpy], A_np, A_deps) # need to wrap `qs` in list because `get_expected_obs_factorized` expects a list of `qs` (representing multiple timesteps)

            for qo_m, qo_val_m in zip(qo_test, qo_validation[0]): # need to extract first index of `qo_validation` because `get_expected_obs_factorized` returns a list of `qo` (representing multiple timesteps)
                self.assertTrue(np.allclose(qo_m, qo_val_m))

    def test_info_gain_factorized(self):
        """ 
        Unit test the `calc_states_info_gain_factorized` function by qualitatively checking that in the T-Maze (contextual bandit)
        example, the state info gain is higher for the policy that leads to visiting the cue, which is higher than state info gain
        for visiting the bandit arm, which in turn is higher than the state info gain for the policy that leads to staying in the start state.
        """

        num_states = [2, 3]  
        num_obs = [3, 3, 3]

        A_dependencies = [[0, 1], [0, 1], [1]] 
        A = []
        for m, obs in enumerate(num_obs):
            lagging_dimensions = [ns for i, ns in enumerate(num_states) if i in A_dependencies[m]]
            modality_shape = [obs] + lagging_dimensions
            A.append(np.zeros(modality_shape))
            if m == 0:
                A[m][:, :, 0] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 1] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 2] = np.array([[0.9, 0.1], [0.0, 0.0], [0.1, 0.9]]) # cue statistics
            if m == 1:
                A[m][2, :, 0] = np.ones(num_states[0])
                A[m][0:2, :, 1] = np.array([[0.6, 0.4], [0.6, 0.4]]) # bandit statistics (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
                A[m][2, :, 2] = np.ones(num_states[0])
            if m == 2:
                A[m] = np.eye(obs)

        qs_start = list(utils.obj_array_uniform(num_states))
        qs_start[1] = np.array([1., 0., 0.]) # agent believes it's in the start state

        A = [jnp.array(A_m) for A_m in A]
        qs_start = [jnp.array(qs) for qs in qs_start]
        qo_start = ctl_jax.compute_expected_obs(qs_start, A, A_dependencies)
        
        start_info_gain = ctl_jax.compute_info_gain(qs_start, qo_start, A, A_dependencies)

        qs_arm = list(utils.obj_array_uniform(num_states))
        qs_arm[1] = np.array([0., 1., 0.]) # agent believes it's in the arm-visiting state
        qs_arm = [jnp.array(qs) for qs in qs_arm]
        qo_arm = ctl_jax.compute_expected_obs(qs_arm, A, A_dependencies)
        
        arm_info_gain = ctl_jax.compute_info_gain(qs_arm, qo_arm, A, A_dependencies)
        
        qs_cue = utils.obj_array_uniform(num_states)
        qs_cue[1] = np.array([0., 0., 1.]) # agent believes it's in the cue-visiting state
        qs_cue = [jnp.array(qs) for qs in qs_cue]
        
        qo_cue = ctl_jax.compute_expected_obs(qs_cue, A, A_dependencies)
        cue_info_gain = ctl_jax.compute_info_gain(qs_cue, qo_cue, A, A_dependencies)
        
        self.assertGreater(arm_info_gain, start_info_gain)
        self.assertGreater(cue_info_gain, arm_info_gain)

        gm_params = generate_model_params()
        num_factors_list, num_states_list, num_modalities_list, num_obs_list, A_deps_list = gm_params['nf_list'], gm_params['ns_list'], gm_params['nm_list'], gm_params['no_list'], gm_params['A_deps_list']
        for (num_states, num_obs, A_deps) in zip(num_states_list, num_obs_list, A_deps_list):

            qs_numpy = utils.random_single_categorical(num_states)
            qs_jax = list(qs_numpy)

            A_np = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_deps)
            A_jax = jtu.tree_map(lambda x: jnp.array(x), list(A_np))   

            qo = ctl_jax.compute_expected_obs(qs_jax, A_jax, A_deps)

            info_gain = ctl_jax.compute_info_gain(qs_jax, qo, A_jax, A_deps)
            info_gain_validation = ctl_np.calc_states_info_gain_factorized(A_np, [qs_numpy],  A_deps)

            self.assertTrue(np.allclose(info_gain, info_gain_validation, atol=1e-5))

    def test_update_posterior_policies_inductive_diagnostics(self):
        model = make_simple_policy_model()

        q_pi, G = ctl_jax.update_posterior_policies_inductive(
            model["policies"],
            model["qs_init"],
            model["A"],
            model["B"],
            model["C"],
            model["E"],
            model["pA"],
            model["pB"],
            model["A_dependencies"],
            model["B_dependencies"],
            model["I"],
            use_param_info_gain=True,
            use_inductive=False,
        )

        q_pi_diag, G_diag, diagnostics = ctl_jax.update_posterior_policies_inductive(
            model["policies"],
            model["qs_init"],
            model["A"],
            model["B"],
            model["C"],
            model["E"],
            model["pA"],
            model["pB"],
            model["A_dependencies"],
            model["B_dependencies"],
            model["I"],
            use_param_info_gain=True,
            use_inductive=False,
            return_diagnostics=True,
        )

        reconstructed = (
            diagnostics["info_gain"]
            + diagnostics["utility"]
            - diagnostics["param_info_gain_a"]
            - diagnostics["param_info_gain_b"]
            + diagnostics["inductive_value"]
        )

        self.assertTrue(np.allclose(q_pi_diag, q_pi))
        self.assertTrue(np.allclose(G_diag, G))
        self.assertTrue(np.allclose(reconstructed, diagnostics["step_neg_G"]))
        self.assertTrue(np.allclose(G_diag, diagnostics["step_neg_G"].sum(-1)))
        self.assertEqual(diagnostics["info_gain"].shape, (model["policies"].shape[0], model["policies"].shape[1]))
        self.assertEqual(diagnostics["utility"].shape, (model["policies"].shape[0], model["policies"].shape[1]))
        self.assertEqual(diagnostics["param_info_gain_a"].shape, (model["policies"].shape[0], model["policies"].shape[1]))
        self.assertEqual(diagnostics["param_info_gain_b"].shape, (model["policies"].shape[0], model["policies"].shape[1]))
        self.assertEqual(
            diagnostics["expected_states"][0].shape,
            (model["policies"].shape[0], model["policies"].shape[1], model["qs_init"][0].shape[0]),
        )
        self.assertEqual(
            diagnostics["expected_observations"][0].shape,
            (model["policies"].shape[0], model["policies"].shape[1], model["A"][0].shape[0]),
        )
    

if __name__ == "__main__":
    unittest.main()