#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import unittest

import numpy as np
import jax.numpy as jnp
from jax import vmap, nn, random
import jax.tree_util as jtu

from pymdp.jax.agent import Agent
from pymdp.jax.maths import compute_log_likelihood_single_modality
from pymdp.jax.utils import norm_dist
from equinox import Module
from typing import Any, List

def make_simple_jax_agent(batch_size=2, policy_len=2):
    A_single = jnp.array([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32)
    B_single = jnp.stack((jnp.eye(2, dtype=jnp.float32), jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)), axis=-1)
    C_single = jnp.array([[0.2, 0.8], [0.6, 0.4]], dtype=jnp.float32)
    D_single = jnp.array([0.6, 0.4], dtype=jnp.float32)

    agent = Agent(
        A=[jnp.broadcast_to(A_single, (batch_size,) + A_single.shape)],
        B=[jnp.broadcast_to(B_single, (batch_size,) + B_single.shape)],
        C=[jnp.broadcast_to(C_single, (batch_size,) + C_single.shape)],
        D=[jnp.broadcast_to(D_single, (batch_size,) + D_single.shape)],
        E=None,
        pA=[jnp.broadcast_to(1.0 + 4.0 * A_single, (batch_size,) + A_single.shape)],
        pB=[jnp.broadcast_to(1.0 + 4.0 * B_single, (batch_size,) + B_single.shape)],
        A_dependencies=[[0]],
        B_dependencies=[[0]],
        policy_len=policy_len,
        use_param_info_gain=True,
        use_inductive=False,
    )

    qs = [
        jnp.array(
            [
                [[0.7, 0.3], [0.55, 0.45]],
                [[0.4, 0.6], [0.6, 0.4]],
            ],
            dtype=jnp.float32,
        )
    ]

    return agent, qs

class TestAgentJax(unittest.TestCase):

    def test_vmappable_agent_methods(self):

        dim, N = 5, 10
        sampling_key = random.PRNGKey(1)

        class BasicAgent(Module):
            A: jnp.ndarray
            B: jnp.ndarray 
            qs: jnp.ndarray

            def __init__(self, A, B, qs=None):
                self.A = A
                self.B = B
                self.qs = jnp.ones((N, dim))/dim if qs is None else qs
            
            @vmap
            def infer_states(self, obs):
                qs = nn.softmax(compute_log_likelihood_single_modality(obs, self.A))
                return qs, BasicAgent(self.A, self.B, qs=qs)

        A_key, B_key, obs_key, test_key = random.split(sampling_key, 4)

        all_A = vmap(norm_dist)(random.uniform(A_key, shape = (N, dim, dim)))
        all_B = vmap(norm_dist)(random.uniform(B_key, shape = (N, dim, dim)))
        all_obs = vmap(nn.one_hot, (0, None))(random.choice(obs_key, dim, shape = (N,)), dim)

        my_agent = BasicAgent(all_A, all_B)

        all_qs, my_agent = my_agent.infer_states(all_obs)

        assert all_qs.shape == my_agent.qs.shape
        self.assertTrue(jnp.allclose(all_qs, my_agent.qs))

        # validate that the method broadcasted properly
        for id_to_check in range(N):
            validation_qs = nn.softmax(compute_log_likelihood_single_modality(all_obs[id_to_check], all_A[id_to_check]))
            self.assertTrue(jnp.allclose(validation_qs, all_qs[id_to_check]))

    def test_infer_policies_return_diagnostics(self):
        agent, qs = make_simple_jax_agent()

        policy_outputs = agent.infer_policies(qs)
        diagnostics_outputs = agent.infer_policies(qs, return_diagnostics=True)

        q_pi, G = policy_outputs
        q_pi_diag, G_diag, diagnostics = diagnostics_outputs

        self.assertEqual(len(policy_outputs), 2)
        self.assertEqual(len(diagnostics_outputs), 3)
        self.assertTrue(np.allclose(q_pi_diag, q_pi))
        self.assertTrue(np.allclose(G_diag, G))
        self.assertTrue(np.allclose(G_diag, diagnostics["step_neg_G"].sum(-1)))
        self.assertEqual(diagnostics["info_gain"].shape, (agent.batch_size, len(agent.policies), agent.policy_len))
        self.assertEqual(diagnostics["utility"].shape, (agent.batch_size, len(agent.policies), agent.policy_len))
        self.assertEqual(diagnostics["param_info_gain_a"].shape, (agent.batch_size, len(agent.policies), agent.policy_len))
        self.assertEqual(diagnostics["param_info_gain_b"].shape, (agent.batch_size, len(agent.policies), agent.policy_len))
        self.assertEqual(
            diagnostics["expected_states"][0].shape,
            (agent.batch_size, len(agent.policies), agent.policy_len, agent.num_states[0]),
        )
        self.assertEqual(
            diagnostics["expected_observations"][0].shape,
            (agent.batch_size, len(agent.policies), agent.policy_len, agent.num_obs[0]),
        )

if __name__ == "__main__":
    unittest.main()       








    
