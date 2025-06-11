#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import warnings
import numpy as np
from pymdp import inference, control, learning
from pymdp import utils, maths
import copy

class Agent(object):
    """ 
    Active Inference Agent Class
    
    This is the main class that implements an Active Inference agent. Active Inference is a theory
    from neuroscience and AI that explains how intelligent agents (like animals or robots) can
    act in their environment by minimizing "surprise" or uncertainty about what they observe.
    
    Key Active Inference Concepts:
    - The agent has beliefs about hidden states in the world (what's actually happening)
    - The agent receives observations that give partial information about these hidden states
    - The agent takes actions to both gather information and achieve goals
    - The agent learns by updating its beliefs based on what it observes
    
    THE COMPLETE ACTIVE INFERENCE CYCLE:
    ===================================
    
    1. INITIALIZATION (__init__):
       - Setup observation model A: "What will I see in different world states?"
       - Setup transition model B: "How do my actions change the world?" 
       - Setup preferences C: "What observations do I want to see?"
       - Setup initial beliefs D: "What state is the world likely to start in?"
       - Setup policies: "What action sequences are possible?"
    
    2. PERCEPTION (infer_states):
       - Receive observation from environment
       - Update beliefs about current world state using Bayes' rule
       - Combine: prior beliefs + observation likelihood → posterior beliefs
       - Formula: P(state|obs) ∝ P(obs|state) × P(state)
    
    3. PLANNING (infer_policies): 
       - Evaluate each possible action sequence (policy)
       - Predict: "If I follow this policy, what will I observe?"
       - Evaluate: "Will I like those observations? Will I learn something?"
       - Compute Expected Free Energy for each policy
       - Choose policy probabilities using softmax
    
    4. ACTION (sample_action):
       - Convert policy preferences into specific action
       - Either pick most popular action across policies (marginal)
       - Or commit to one policy and follow it (full)
       - Execute action in environment
    
    5. LEARNING (update_A, update_B, update_D):
       - Update model parameters based on experience
       - Learn better observation model: "I was wrong about what I'd see"
       - Learn better transition model: "I was wrong about what my actions do"
       - Learn better initial beliefs: "I was wrong about likely starting states"
    
    6. REPEAT: Go back to step 2 with new observation
    
    MATHEMATICAL FOUNDATION:
    =======================
    
    The agent minimizes "Free Energy" F, which upper-bounds surprise:
    F = Accuracy + Complexity
    
    Where:
    - Accuracy = How well beliefs explain observations
    - Complexity = How much beliefs differ from priors
    
    This drives the agent to:
    - Seek observations that match its predictions (avoid surprise)
    - Take actions that lead to preferred observations (achieve goals)
    - Seek information to improve its model (learn and explore)
    
    The basic usage is as follows:

    >>> my_agent = Agent(A = A, B = B, <more_params>)
    >>> observation = env.step(initial_action)
    >>> qs = my_agent.infer_states(observation)
    >>> q_pi, G = my_agent.infer_policies()
    >>> next_action = my_agent.sample_action()
    >>> next_observation = env.step(next_action)

    This represents one timestep of an active inference process. Wrapping this step in a loop with an ``Env()`` class that returns
    observations and takes actions as inputs, would entail a dynamic agent-environment interaction.
    """

    def __init__(
        self,
        A,          # Observation model: How hidden states generate observations
        B,          # Transition model: How actions change hidden states over time
        C=None,     # Preferences: What observations the agent "likes" or wants to see
        D=None,     # Initial state prior: Agent's initial beliefs about hidden states
        E=None,     # Policy prior: Agent's initial preferences for different action sequences
        H=None,     # Goal states for backwards induction (advanced feature)
        pA=None,    # Prior parameters for learning the observation model
        pB=None,    # Prior parameters for learning the transition model  
        pD=None,    # Prior parameters for learning initial state beliefs
        num_controls=None,      # Number of possible actions for each controllable factor
        policy_len=1,           # How many steps ahead the agent plans (1 = reactive, >1 = planning)
        inference_horizon=1,    # How many past observations to consider when inferring states
        control_fac_idx=None,   # Which hidden state factors can be controlled by actions
        policies=None,          # Specific action sequences to consider (if None, all combinations)
        gamma=16.0,            # Policy precision: How confident the agent is in its policy choices (higher = more confident)
        alpha=16.0,            # Action precision: How deterministically the agent chooses actions (higher = more deterministic)
        use_utility=True,       # Whether to consider preferences (goals) when choosing actions
        use_states_info_gain=True,    # Whether to seek information about hidden states
        use_param_info_gain=False,    # Whether to seek information to learn model parameters
        action_selection="deterministic",  # How to choose actions: "deterministic" or "stochastic" 
        sampling_mode = "marginal",   # "marginal": choose best action, "full": choose best policy then follow it
        inference_algo="VANILLA",     # Algorithm for state inference: "VANILLA" (simple) or "MMP" (complex planning)
        inference_params=None,        # Additional parameters for the inference algorithm
        modalities_to_learn="all",    # Which observation types to learn about
        lr_pA=1.0,                   # Learning rate for observation model (how fast to update beliefs)
        factors_to_learn="all",       # Which hidden state factors to learn about  
        lr_pB=1.0,                   # Learning rate for transition model
        lr_pD=1.0,                   # Learning rate for initial state beliefs
        use_BMA=True,                # Use Bayesian Model Averaging (advanced belief updating)
        policy_sep_prior=False,      # Use policy-separated priors (advanced feature)
        save_belief_hist=False,      # Whether to save history of beliefs over time
        A_factor_list=None,          # Which hidden factors affect which observations (for efficiency)
        B_factor_list=None,          # Which hidden factors affect which transitions (for efficiency)
        sophisticated=False,         # Use sophisticated inference (tree search for planning)
        si_horizon=3,                # Planning horizon for sophisticated inference
        si_policy_prune_threshold=1/16,   # Threshold for pruning unlikely policies
        si_state_prune_threshold=1/16,    # Threshold for pruning unlikely states
        si_prune_penalty=512,        # Penalty for exploring pruned branches
        ii_depth=10,                 # Depth for backwards induction
        ii_threshold=1/16,           # Threshold for backwards induction
    ):
        """
        Initialize an Active Inference Agent
        
        This method sets up all the components the agent needs to perceive, think, and act.
        Think of it like setting up a brain with:
        - Knowledge about how the world works (A and B matrices)
        - Goals and preferences (C vector) 
        - Initial beliefs about the current situation (D vector)
        - Learning capabilities (pA, pB, pD parameters)
        - Decision-making strategies (gamma, alpha, policies)
        """

        ### STEP 1: Store Policy and Decision-Making Parameters ###
        
        # How many steps ahead the agent plans
        # policy_len = 1: Agent is reactive, only thinks about immediate next action
        # policy_len > 1: Agent plans multiple steps ahead
        self.policy_len = policy_len
        
        # Policy precision (gamma): Controls how confident agent is in policy selection
        # Higher gamma = agent strongly prefers the best policy
        # Lower gamma = agent considers multiple policies more equally
        # Formula: P(policy) ∝ exp(gamma * expected_value_of_policy)
        self.gamma = gamma
        
        # Action precision (alpha): Controls randomness in action selection
        # Higher alpha = more deterministic action choices
        # Lower alpha = more random/exploratory action choices  
        self.alpha = alpha
        
        # How the agent chooses actions: "deterministic" picks best, "stochastic" samples
        self.action_selection = action_selection
        
        # Whether to sample from marginal action distribution or full policy distribution
        # "marginal": Consider all policies but pick best single action
        # "full": Pick a complete policy and follow it
        self.sampling_mode = sampling_mode
        
        # What drives the agent's behavior:
        # use_utility: Seek preferred/rewarding observations (goal-seeking)
        # use_states_info_gain: Seek information about what's happening (curiosity about states)  
        # use_param_info_gain: Seek information to improve world model (curiosity about learning)
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain

        ### STEP 2: Store Learning Parameters ###
        
        # Learning rates control how fast the agent updates its beliefs
        # Higher learning rate = faster adaptation to new information
        # Lower learning rate = more conservative, slower learning
        
        # Which types of observations to learn about ("all" or list of indices)
        self.modalities_to_learn = modalities_to_learn
        # Learning rate for observation model: how observations relate to hidden states
        self.lr_pA = lr_pA
        
        # Which hidden state factors to learn about ("all" or list of indices)  
        self.factors_to_learn = factors_to_learn
        # Learning rate for transition model: how actions change states
        self.lr_pB = lr_pB
        # Learning rate for initial state beliefs
        self.lr_pD = lr_pD

        ### STEP 3: Setup Sophisticated Inference (Advanced Planning) ###
        
        # Sophisticated inference uses tree search for deeper planning
        self.sophisticated = sophisticated
        if self.sophisticated:
            # Sophisticated inference currently only works with single-step policies
            assert self.policy_len == 1, "Sophisticated inference only works with policy_len = 1"
        
        # Parameters for tree search planning:
        self.si_horizon = si_horizon  # How far ahead to search
        self.si_policy_prune_threshold = si_policy_prune_threshold  # Ignore very unlikely policies
        self.si_state_prune_threshold = si_state_prune_threshold    # Ignore very unlikely states
        self.si_prune_penalty = si_prune_penalty  # Cost penalty for exploring pruned branches

        ### STEP 4: Setup Observation Model (A matrices) ###
        
        # The A matrix (observation model) encodes the agent's knowledge about:
        # "If the world is in hidden state X, what observation will I see?"
        # 
        # Example: If you're in a dark room with a light switch:
        # - Hidden state: light is ON or OFF  
        # - Observation: room appears BRIGHT or DARK
        # - A matrix: A[bright, light_on] = 0.9, A[dark, light_on] = 0.1
        #            A[bright, light_off] = 0.1, A[dark, light_off] = 0.9
        # This means: if light is on, you'll probably see brightness (but not always due to noise)
        
        if not isinstance(A, np.ndarray):
            raise TypeError(
                'A matrix must be a numpy array'
            )

        # Convert A to object array format (allows multiple observation modalities)
        # Object arrays let us have different-sized matrices for different senses
        # e.g., A[0] = vision matrix, A[1] = hearing matrix, A[2] = touch matrix
        self.A = utils.to_obj_array(A)

        # Verify A matrices are properly normalized (each column sums to 1)
        # This ensures P(observation|hidden_state) is a proper probability distribution
        # Each column represents: "given this hidden state, what's the probability of each observation?"
        assert utils.is_normalized(self.A), "A matrix is not normalized (i.e. A[m].sum(axis = 0) must all equal 1.0 for all modalities)"

        # Extract dimensions of the observation space
        # num_obs[m] = number of possible observations for modality m
        # e.g., if modality 0 is vision with 3 possible observations: [bright, dim, dark]
        # then num_obs[0] = 3
        self.num_obs = [self.A[m].shape[0] for m in range(len(self.A))]
        # Total number of different senses/observation types
        self.num_modalities = len(self.num_obs)

        # Store prior parameters for learning the observation model
        # pA contains Dirichlet parameters that represent the agent's uncertainty about A
        # Higher pA values = more confident in current A matrix
        # Lower pA values = more willing to change A based on new observations
        # If pA is None, the agent won't learn/update its observation model
        self.pA = pA

        ### STEP 5: Setup Transition Model (B matrices) ###
        
        # The B matrix (transition model) encodes the agent's knowledge about:
        # "If I take action A when in hidden state X, what hidden state will I end up in?"
        #
        # Example: Light switch dynamics
        # - Hidden state: light is ON or OFF
        # - Actions: PRESS_SWITCH or DO_NOTHING  
        # - B matrix for PRESS_SWITCH: B[off, on, press] = 0.9, B[on, off, press] = 0.9
        # - B matrix for DO_NOTHING: B[on, on, nothing] = 1.0, B[off, off, nothing] = 1.0
        # This means: pressing switch usually changes state, doing nothing keeps state same
        
        if not isinstance(B, np.ndarray):
            raise TypeError(
                'B matrix must be a numpy array'
            )

        # Convert B to object array format (allows multiple hidden state factors)  
        # Object arrays let us have different transition dynamics for different aspects of the world
        # e.g., B[0] = room lighting dynamics, B[1] = agent location dynamics
        self.B = utils.to_obj_array(B)

        # Verify B matrices are properly normalized (each column sums to 1)
        # This ensures P(next_state|current_state, action) is a proper probability distribution
        # Each column represents: "given this state and action, what's the probability of each next state?"
        assert utils.is_normalized(self.B), "B matrix is not normalized (i.e. B[f].sum(axis = 0) must all equal 1.0 for all factors)"

        # Extract dimensions of the hidden state space
        # num_states[f] = number of possible states for hidden factor f
        # e.g., if factor 0 represents agent location with 4 rooms: [kitchen, bedroom, bathroom, living_room]
        # then num_states[0] = 4
        self.num_states = [self.B[f].shape[0] for f in range(len(self.B))]
        # Total number of different hidden state factors
        self.num_factors = len(self.num_states)

        # Store prior parameters for learning the transition model
        # pB contains Dirichlet parameters that represent the agent's uncertainty about B
        # Higher pB values = more confident in current B matrix (dynamics)
        # Lower pB values = more willing to change B based on new experience
        # If pB is None, the agent won't learn/update its transition model
        self.pB = pB

        # Determine number of possible actions (controls) for each factor
        # This is inferred from the last dimension of each B matrix
        # num_controls[f] = how many different actions can affect factor f
        if num_controls == None:
            # Auto-detect from B matrix dimensions: last dimension = number of actions
            self.num_controls = [self.B[f].shape[-1] for f in range(self.num_factors)]
        else:
            # Verify user-provided num_controls matches B matrix structure
            inferred_num_controls = [self.B[f].shape[-1] for f in range(self.num_factors)]
            assert num_controls == inferred_num_controls, "num_controls must be consistent with the shapes of the input B matrices"
            self.num_controls = num_controls

        ### STEP 6: Setup Factorization for Computational Efficiency ###
        
        # COMPUTATIONAL EFFICIENCY EXPLANATION:
        # In complex environments, computing exact probabilities over all combinations of 
        # hidden states becomes computationally intractable (curse of dimensionality).
        # 
        # For example, if you have 3 factors with 10 states each, that's 10^3 = 1000 combinations!
        # But often, not all observations depend on all hidden state factors.
        # 
        # Example: In a house environment
        # - Room lighting (factor 0): affects visual observations but not sound 
        # - Agent location (factor 1): affects what you can see and hear
        # - Time of day (factor 2): affects outside sounds but not room lighting
        #
        # A_factor_list lets us specify these dependencies for efficiency:
        # A_factor_list[vision] = [0, 1]     # vision depends on lighting and location
        # A_factor_list[sound] = [1, 2]      # sound depends on location and time
        
        # Track whether we're using factorized (efficient) or full (simple) computation
        self.factorized = False
        
        if A_factor_list == None:
            # DEFAULT: Assume all observations depend on all hidden state factors
            # This is computationally expensive but works for simple problems
            self.A_factor_list = self.num_modalities * [list(range(self.num_factors))]
            
            # Verify that A matrix dimensions match the assumed dependencies
            for m in range(self.num_modalities):
                factor_dims = tuple([self.num_states[f] for f in self.A_factor_list[m]])
                assert self.A[m].shape[1:] == factor_dims, f"Please input an `A_factor_list` whose {m}-th indices pick out the hidden state factors that line up with lagging dimensions of A{m}..." 
                if self.pA is not None:
                    assert self.pA[m].shape[1:] == factor_dims, f"Please input an `A_factor_list` whose {m}-th indices pick out the hidden state factors that line up with lagging dimensions of pA{m}..." 
        else:
            # CUSTOM: User specified which factors affect which observations (more efficient)
            self.factorized = True
            for m in range(self.num_modalities):
                assert max(A_factor_list[m]) <= (self.num_factors - 1), f"Check modality {m} of A_factor_list - must be consistent with `num_states` and `num_factors`..."
                factor_dims = tuple([self.num_states[f] for f in A_factor_list[m]])
                assert self.A[m].shape[1:] == factor_dims, f"Check modality {m} of A_factor_list. It must coincide with lagging dimensions of A{m}..." 
                if self.pA is not None:
                    assert self.pA[m].shape[1:] == factor_dims, f"Check modality {m} of A_factor_list. It must coincide with lagging dimensions of pA{m}..."
            self.A_factor_list = A_factor_list

        # Create the reverse mapping: for each factor, which observations depend on it?
        # This is used for efficient belief updating - we only update beliefs about factors
        # that are relevant to the current observation
        A_modality_list = []
        for f in range(self.num_factors):
            A_modality_list.append( [m for m in range(self.num_modalities) if f in self.A_factor_list[m]] )

        # Store factor-modality relationships in a "Markov blanket" dictionary
        # Markov blanket = the minimal set of variables needed to predict something
        self.mb_dict = {
                        'A_factor_list': self.A_factor_list,      # Which factors affect each observation
                        'A_modality_list': A_modality_list        # Which observations are affected by each factor
                        }

        # Setup B_factor_list for transition model factorization
        # This specifies which hidden state factors can influence the transitions of other factors
        if B_factor_list == None:
            # DEFAULT: Each factor only depends on itself (most common case)
            # e.g., room lighting only changes based on current lighting state + actions
            # agent location only changes based on current location + movement actions
            self.B_factor_list = [[f] for f in range(self.num_factors)]
            
            # Verify B matrix dimensions match this assumption
            for f in range(self.num_factors):
                factor_dims = tuple([self.num_states[f] for f in self.B_factor_list[f]])
                assert self.B[f].shape[1:-1] == factor_dims, f"Please input a `B_factor_list` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of B{f}..." 
                if self.pB is not None:
                    assert self.pB[f].shape[1:-1] == factor_dims, f"Please input a `B_factor_list` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of pB{f}..." 
        else:
            # CUSTOM: User specified factor interactions (for complex coupled dynamics)
            # e.g., turning on air conditioning affects both temperature and humidity factors
            self.factorized = True
            for f in range(self.num_factors):
                assert max(B_factor_list[f]) <= (self.num_factors - 1), f"Check factor {f} of B_factor_list - must be consistent with `num_states` and `num_factors`..."
                factor_dims = tuple([self.num_states[f] for f in B_factor_list[f]])
                assert self.B[f].shape[1:-1] == factor_dims, f"Check factor {f} of B_factor_list. It must coincide with all-but-final lagging dimensions of B{f}..." 
                if self.pB is not None:
                    assert self.pB[f].shape[1:-1] == factor_dims, f"Check factor {f} of B_factor_list. It must coincide with all-but-final lagging dimensions of pB{f}..."
            self.B_factor_list = B_factor_list

        ### STEP 7: Setup Controllable Factors ###
        
        # Not all aspects of the world can be directly controlled by the agent
        # For example: agent can control their location and light switches, 
        # but cannot directly control the weather or other agents
        
        if control_fac_idx == None:
            # AUTO-DETECT: A factor is controllable if it has more than 1 action
            # (1 action = "do nothing", >1 actions = "can actually control something")
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:
            # USER-SPECIFIED: Explicitly list which factors the agent can control
            assert max(control_fac_idx) <= (self.num_factors - 1), "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            self.control_fac_idx = control_fac_idx

            # Verify that all specified controllable factors actually have multiple actions
            for factor_idx in self.control_fac_idx:
                assert self.num_controls[factor_idx] > 1, "Control factor (and B matrix) dimensions are not consistent with user-given control_fac_idx"

        ### STEP 8: Setup Policies (Action Sequences) ###
        
        # A policy is a sequence of actions over time
        # policy_len=1: policy is just a single action [action_t]
        # policy_len=3: policy is sequence [action_t, action_t+1, action_t+2]
        
        if policies is None:
            # AUTO-GENERATE: Create all possible action combinations
            # This can be computationally expensive for many actions/factors
            policies = self._construct_policies()
        self.policies = policies

        # Verify policy structure is consistent with our control setup
        assert all([len(self.num_controls) == policy.shape[1] for policy in self.policies]), "Number of control states is not consistent with policy dimensionalities"
        
        # Check that all actions in policies are actually possible
        all_policies = np.vstack(self.policies)
        assert all([n_c >= max_action for (n_c, max_action) in zip(self.num_controls, list(np.max(all_policies, axis =0)+1))]), "Maximum number of actions is not consistent with `num_controls`"

        ### STEP 9: Setup Preferences (C matrices) ###
        
        # The C vector encodes the agent's preferences - what observations it "likes" or "wants"
        # Higher C values = more preferred observations (goals to seek)
        # Lower C values = less preferred observations (things to avoid)
        # Zero C values = neutral observations
        #
        # Example: For a robot that should seek food and avoid obstacles:
        # C[food_detected] = +10    # Strongly prefer seeing food
        # C[obstacle_detected] = -5 # Want to avoid seeing obstacles  
        # C[empty_space] = 0        # Neutral about empty space

        if C is not None:
            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.C = utils.to_obj_array(C)

            # Verify C vector dimensions match observation dimensions
            assert len(self.C) == self.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {self.num_modalities}"

            for modality, c_m in enumerate(self.C):
                assert c_m.shape[0] == self.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {self.num_obs[modality]}"
        else:
            # DEFAULT: No preferences (all observations equally neutral)
            # Agent will be purely driven by curiosity/information-seeking
            self.C = self._construct_C_prior()

        ### STEP 10: Setup Initial State Beliefs (D vectors) ###
        
        # The D vector represents the agent's initial beliefs about hidden states
        # before seeing any observations. This is like the agent's "prior knowledge"
        # about what state the world is likely to start in.
        #
        # Example: For a robot starting in a house:
        # D[kitchen] = 0.1      # Low probability of starting in kitchen
        # D[living_room] = 0.7  # High probability of starting in living room  
        # D[bedroom] = 0.2      # Medium probability of starting in bedroom
    
        if D is not None:
            if not isinstance(D, np.ndarray):
                raise TypeError(
                    'D vector must be a numpy array'
                )
            self.D = utils.to_obj_array(D)

            # Verify D vector dimensions match hidden state dimensions
            assert len(self.D) == self.num_factors, f"Check D vector: number of sub-arrays must be equal to number of hidden state factors: {self.num_factors}"

            for f, d_f in enumerate(self.D):
                assert d_f.shape[0] == self.num_states[f], f"Check D vector: number of entries of D vector for factor {f} should be equal to {self.num_states[f]}"
        else:
            if pD is not None:
                # If learning parameters provided, use them to initialize D
                self.D = utils.norm_dist_obj_arr(pD)
            else:
                # DEFAULT: Uniform distribution over initial states (no prior knowledge)
                self.D = self._construct_D_prior()

        # Verify D is a proper probability distribution (sums to 1)
        assert utils.is_normalized(self.D), "D vector is not normalized (i.e. D[f].sum() must all equal 1.0 for all factors)"

        # Store learning parameters for initial state beliefs
        self.pD = pD

        ### STEP 11: Setup Policy Preferences (E vector) ###
        
        # The E vector represents the agent's a priori preferences over policies
        # This is independent of outcomes - it's like having a bias toward certain
        # types of actions regardless of their consequences
        #
        # Example: An agent might prefer:
        # - Simple policies over complex ones (Occam's razor)
        # - Conservative policies over risky ones
        # - Familiar policies over novel ones
        
        if E is not None:
            if not isinstance(E, np.ndarray):
                raise TypeError(
                    'E vector must be a numpy array'
                )
            self.E = E

            # Verify E vector length matches number of policies
            assert len(self.E) == len(self.policies), f"Check E vector: length of E must be equal to number of policies: {len(self.policies)}"

        else:
            # DEFAULT: Uniform preferences over all policies (no bias)
            self.E = self._construct_E_prior()
        
        ### STEP 12: Setup Backwards Induction (Advanced Planning) ###
        
        # H and I are for sophisticated planning using backwards induction
        # This is an advanced feature for goal-directed behavior where the agent
        # works backwards from desired end states to plan optimal action sequences
        
        if H is not None:
            # H specifies goal states that the agent should try to reach
            self.H = H
            # I is computed via backwards induction - working backwards from goals
            # to determine optimal value function for each state
            self.I = control.backwards_induction(H, B, B_factor_list, threshold=ii_threshold, depth=ii_depth)
        else:
            # No goal-directed planning - agent uses forward inference only
            self.H = None
            self.I = None

        ### STEP 13: Setup Advanced Belief Management ###
        
        # These parameters control how beliefs are updated at temporal boundaries
        # (e.g., when the inference horizon moves forward in time)
        self.edge_handling_params = {}
        
        # Use Bayesian Model Averaging: blend beliefs across different policies
        # This creates a "summary" belief that averages over policy uncertainty
        self.edge_handling_params['use_BMA'] = use_BMA
        
        # Use policy-separated priors: maintain separate belief histories for each policy
        # This keeps track of what would have happened under each different policy path
        self.edge_handling_params['policy_sep_prior'] = policy_sep_prior

        # These two options are mutually exclusive - can't do both at once
        if policy_sep_prior:
            if use_BMA:
                warnings.warn(
                    "Inconsistent choice of `policy_sep_prior` and `use_BMA`.\
                    You have set `policy_sep_prior` to True, so we are setting `use_BMA` to False"
                )
                self.edge_handling_params['use_BMA'] = False
        
        ### STEP 14: Setup Inference Algorithm ###
        
        # Choose the algorithm for inferring hidden states from observations
        # VANILLA: Simple, fast inference for reactive behavior
        # MMP: Complex message-passing for planning with temporal depth
        
        if inference_algo == None:
            self.inference_algo = "VANILLA"
            self.inference_params = self._get_default_params()
            if inference_horizon > 1:
                warnings.warn(
                    "If `inference_algo` is VANILLA, then inference_horizon must be 1\n. \
                    Setting inference_horizon to default value of 1...\n"
                    )
                self.inference_horizon = 1
            else:
                self.inference_horizon = 1
        else:
            self.inference_algo = inference_algo
            self.inference_params = self._get_default_params()
            self.inference_horizon = inference_horizon

        ### STEP 15: Initialize History Tracking and State Variables ###
        
        # Optionally save history of beliefs and decisions for analysis/debugging
        if save_belief_hist:
            self.qs_hist = []      # History of beliefs about hidden states over time
            self.q_pi_hist = []    # History of beliefs about policies over time
        
        # Initialize tracking variables for agent's experience
        self.prev_obs = []         # List of previously observed observations
        self.reset()               # Set initial beliefs to D prior (calls reset method)
        
        # These will be set during operation:
        self.action = None         # Current action (set by sample_action method)
        self.prev_actions = None   # History of previous actions taken

    def _construct_C_prior(self):
        """
        HELPER: Create default preference vector (all zeros = no preferences)
        """
        C = utils.obj_array_zeros(self.num_obs)
        return C

    def _construct_D_prior(self):
        """
        HELPER: Create default initial state prior (uniform = no prior knowledge about starting states)
        """
        D = utils.obj_array_uniform(self.num_states)
        return D

    def _construct_policies(self):
        """
        HELPER: Generate all possible action sequences (policies) based on controllable factors
        """
        policies =  control.construct_policies(
            self.num_states, self.num_controls, self.policy_len, self.control_fac_idx
        )
        return policies

    def _construct_num_controls(self):
        """
        HELPER: Infer number of possible actions from existing policies
        """
        num_controls = control.get_num_controls_from_policies(
            self.policies
        )
        return num_controls
    
    def _construct_E_prior(self):
        """
        HELPER: Create default policy prior (uniform = no bias toward any particular policy)
        """
        E = np.ones(len(self.policies)) / len(self.policies)
        return E

    def reset(self, init_qs=None):
        """
        AGENT RESET: Starting Fresh for a New Episode
        
        This method prepares the agent for a new episode or simulation by resetting its
        beliefs to initial conditions. It's like "clearing the agent's memory" and
        starting with fresh prior beliefs.
        
        THE BASIC IDEA:
        When starting a new episode, the agent should:
        1. Reset its beliefs about hidden states to the initial prior (D vector)
        2. Reset the simulation time back to timestep 0
        3. Refresh its models from the learning parameters (if learning is enabled)
        4. Initialize the appropriate belief structure for the chosen inference algorithm
        
        WHEN TO USE:
        - At the beginning of a new episode/trial
        - When switching to a new environment
        - When you want the agent to "forget" its current state beliefs
        
        DIFFERENT ALGORITHMS:
        - VANILLA: Simple uniform beliefs over states
        - MMP: Complex structure with beliefs for each policy and timestep

        Parameters
        -----------
        init_qs: ``numpy.ndarray`` of dtype object or ``None``
            Optional custom initial beliefs about hidden states.
            If None, uses default initialization based on the inference algorithm.

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Initialized posterior beliefs over hidden states.
            Structure depends on inference algorithm:
            - VANILLA: qs[f] = uniform belief about factor f
            - MMP: qs[policy][timestep][factor] = belief structure for planning
        """

        # Reset simulation time to the beginning
        self.curr_timestep = 0

        ### Initialize Belief Structure ###
        if init_qs is None:
            
            ### VANILLA: Simple Uniform Beliefs ###
            if self.inference_algo == 'VANILLA':
                # Start with uniform beliefs over all hidden states
                # This means "I don't know what state I'm in, all states seem equally likely"
                self.qs = utils.obj_array_uniform(self.num_states)
                
            ### MMP: Complex Policy-Conditioned Beliefs ###
            else:
                # For MMP, we need beliefs for each policy and each timestep
                # This allows planning over multiple timesteps and policies
                
                # Create belief structure: qs[policy][timestep][factor]
                self.qs = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    # Each policy needs beliefs for: inference_horizon + policy_len + current_timestep
                    self.qs[p_i] = utils.obj_array(self.inference_horizon + self.policy_len + 1)
                    # Initialize only the first timestep with uniform beliefs
                    self.qs[p_i][0] = utils.obj_array_uniform(self.num_states)
                
                # Setup initial beliefs for the inference horizon
                first_belief = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    first_belief[p_i] = copy.deepcopy(self.D)  # Use learned initial state prior
                
                # Set the belief management based on configuration
                if self.edge_handling_params['policy_sep_prior']:
                    # Use policy-separated priors
                    self.set_latest_beliefs(last_belief = first_belief)
                else:
                    # Use Bayesian model averaging
                    self.set_latest_beliefs(last_belief = self.D)
            
        else:
            # Use custom provided initial beliefs
            self.qs = init_qs
        
        ### Update Models from Learning Parameters ###
        # If the agent has been learning, update models from latest parameters
        
        if self.pA is not None:
            # Update observation model from learned parameters
            self.A = utils.norm_dist_obj_arr(self.pA)
        
        if self.pB is not None:
            # Update transition model from learned parameters
            self.B = utils.norm_dist_obj_arr(self.pB)

        return self.qs

    def step_time(self):
        """
        Advances time by one step. This involves updating the ``self.prev_actions``, and in the case of a moving
        inference horizon, this also shifts the history of post-dictive beliefs forward in time (using ``self.set_latest_beliefs()``),
        so that the penultimate belief before the beginning of the horizon is correctly indexed.

        Returns
        ---------
        curr_timestep: ``int``
            The index in absolute simulation time of the current timestep.
        """

        if self.prev_actions is None:
            self.prev_actions = [self.action]
        else:
            self.prev_actions.append(self.action)

        self.curr_timestep += 1

        if self.inference_algo == "MMP" and (self.curr_timestep - self.inference_horizon) >= 0:
            self.set_latest_beliefs()
        
        return self.curr_timestep
    
    def set_latest_beliefs(self,last_belief=None):
        """
        Both sets and returns the penultimate belief before the first timestep of the backwards inference horizon. 
        In the case that the inference horizon includes the first timestep of the simulation, then the ``latest_belief`` is
        simply the first belief of the whole simulation, or the prior (``self.D``). The particular structure of the ``latest_belief``
        depends on the value of ``self.edge_handling_params['use_BMA']``.

        Returns
        ---------
        latest_belief: ``numpy.ndarray`` of dtype object
            Penultimate posterior beliefs over hidden states at the timestep just before the first timestep of the inference horizon. 
            Depending on the value of ``self.edge_handling_params['use_BMA']``, the shape of this output array will differ.
            If ``self.edge_handling_params['use_BMA'] == True``, then ``latest_belief`` will be a Bayesian model average 
            of beliefs about hidden states, where the average is taken with respect to posterior beliefs about policies.
            Otherwise, `latest_belief`` will be the full, policy-conditioned belief about hidden states, and will have indexing structure
            policies->factors, such that ``latest_belief[p_idx][f_idx]`` refers to the penultimate belief about marginal factor ``f_idx``
            under policy ``p_idx``.
        """

        if last_belief is None:
            last_belief = utils.obj_array(len(self.policies))
            for p_i, _ in enumerate(self.policies):
                last_belief[p_i] = copy.deepcopy(self.qs[p_i][0])

        begin_horizon_step = self.curr_timestep - self.inference_horizon
        if self.edge_handling_params['use_BMA'] and (begin_horizon_step >= 0):
            if hasattr(self, "q_pi_hist"):
                self.latest_belief = inference.average_states_over_policies(last_belief, self.q_pi_hist[begin_horizon_step]) # average the earliest marginals together using contemporaneous posterior over policies (`self.q_pi_hist[0]`)
            else:
                self.latest_belief = inference.average_states_over_policies(last_belief, self.q_pi) # average the earliest marginals together using posterior over policies (`self.q_pi`)
        else:
            self.latest_belief = last_belief

        return self.latest_belief
    
    def get_future_qs(self):
        """
        Returns the last ``self.policy_len`` timesteps of each policy-conditioned belief
        over hidden states. This is a step of pre-processing that needs to be done before computing
        the expected free energy of policies. We do this to avoid computing the expected free energy of 
        policies using beliefs about hidden states in the past (so-called "post-dictive" beliefs).

        Returns
        ---------
        future_qs_seq: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states under a policy, in the future. This is a nested ``numpy.ndarray`` object array, with one
            sub-array ``future_qs_seq[p_idx]`` for each policy. The indexing structure is policy->timepoint-->factor, so that 
            ``future_qs_seq[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at future timepoint ``t_idx``, relative to the current timestep.
        """
        
        future_qs_seq = utils.obj_array(len(self.qs))
        for p_idx in range(len(self.qs)):
            future_qs_seq[p_idx] = self.qs[p_idx][-(self.policy_len+1):] # this grabs only the last `policy_len`+1 beliefs about hidden states, under each policy

        return future_qs_seq


    def infer_states(self, observation, distr_obs=False):
        """
        STATE INFERENCE: The Core Perceptual Process
        
        This is where the agent updates its beliefs about what's happening in the world
        based on what it observes. This is like the "perception" part of the brain.
        
        THE BASIC IDEA:
        The agent doesn't directly know the true hidden states (what's really happening).
        It only gets observations (what it can see/hear/feel). Using Bayes' rule, it
        combines:
        1. Prior beliefs (what it expected to happen)
        2. Observation likelihood (how likely this observation is given different states)
        → Posterior beliefs (updated beliefs about what's actually happening)
        
        MATHEMATICAL FOUNDATION:
        P(hidden_state | observation) ∝ P(observation | hidden_state) × P(hidden_state)
        
        Where:
        - P(observation | hidden_state) comes from the A matrix
        - P(hidden_state) is the prior (from previous beliefs or D vector)
        - P(hidden_state | observation) is what we want to compute

        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.
            Example: [2, 1] might mean "see bright light (obs 2 for vision), hear loud sound (obs 1 for hearing)"
        distr_obs: ``bool``
            Whether the observation is a distribution over possible observations, rather than a single observation.
            False: observation is certain (normal case)
            True: observation is uncertain/noisy (advanced case)

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states after seeing the observation.
            This represents the agent's updated beliefs about what's happening in the world.
            Structure depends on the inference algorithm:
            - VANILLA: qs[f] = belief about factor f
            - MMP: qs[policy][timepoint][factor] = belief under different policy/time combinations
        """

        # Convert observation to tuple format for consistency
        observation = tuple(observation) if not distr_obs else observation

        # Initialize beliefs if this is the first observation
        if not hasattr(self, "qs"):
            self.reset()

        ### VANILLA INFERENCE: Simple, Fast State Inference ###
        if self.inference_algo == "VANILLA":
            
            # STEP 1: Determine the prior belief about current state
            if self.action is not None:
                # If we took an action, predict where we should be now using transition model B
                # This computes: P(current_state) = ∑ P(current_state | previous_state, action) × P(previous_state)
                empirical_prior = control.get_expected_states_interactions(
                    self.qs, self.B, self.B_factor_list, self.action.reshape(1, -1) 
                )[0]
            else:
                # If no action taken yet, use initial state prior D
                empirical_prior = self.D
            
            # STEP 2: Update beliefs using Bayes' rule with observation
            # This solves: P(state | observation) ∝ P(observation | state) × P(state)
            # Uses variational inference for computational efficiency
            qs = inference.update_posterior_states_factorized(
                self.A,                    # Observation model: P(obs | state)
                observation,               # What we actually observed
                self.num_obs,             # Dimensions of observation space
                self.num_states,          # Dimensions of state space  
                self.mb_dict,             # Factorization structure for efficiency
                empirical_prior,          # Prior belief: P(state)
                **self.inference_params   # Algorithm parameters (iterations, convergence, etc.)
            )
            
        ### MMP INFERENCE: Complex Planning with Temporal Depth ###
        elif self.inference_algo == "MMP":
            
            # STEP 1: Add this observation to our history
            self.prev_obs.append(observation)
            
            # STEP 2: Extract relevant observation and action history for inference
            # Only use the last 'inference_horizon' observations to limit computational cost
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]              # Recent observations
                latest_actions = self.prev_actions[-(self.inference_horizon-1):] # Recent actions
            else:
                # If we don't have enough history yet, use everything we have
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions

            # STEP 3: Solve the full temporal inference problem
            # This simultaneously infers:
            # - What states the agent was in over the recent past
            # - What states the agent expects to be in under different future policies
            # This is much more complex than VANILLA but enables planning
            qs, F = inference.update_posterior_states_full_factorized(
                self.A,                                              # Observation model
                self.mb_dict,                                        # Factorization structure
                self.B,                                              # Transition model
                self.B_factor_list,                                  # Transition factorization
                latest_obs,                                          # Sequence of recent observations
                self.policies,                                       # All possible action policies
                latest_actions,                                      # Sequence of recent actions
                prior = self.latest_belief,                          # Beliefs at start of horizon
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )

            # Store the variational free energy for each policy
            # F[p] = "surprise" or "cost" of explaining observations under policy p
            # Lower F = better policy (explains observations with less surprise)
            self.F = F

        # STEP 4: Store results and update agent's state
        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)  # Save to history if tracking is enabled
        self.qs = qs                 # Update current beliefs

        return qs

    def _infer_states_test(self, observation, distr_obs=False):
        """
        Test version of ``infer_states()`` that additionally returns intermediate variables of MMP, such as
        the prediction errors and intermediate beliefs from the optimization. Used for benchmarking against SPM outputs.
        """
        observation = tuple(observation) if not distr_obs else observation

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo == "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B, self.action.reshape(1, -1) 
                )[0]
            else:
                empirical_prior = self.D
            qs = inference.update_posterior_states(
                self.A,
                observation,
                empirical_prior,
                **self.inference_params
            )
        elif self.inference_algo == "MMP":

            self.prev_obs.append(observation)
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]
                latest_actions = self.prev_actions[-(self.inference_horizon-1):]
            else:
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions

            qs, F, xn, vn = inference._update_posterior_states_full_test(
                self.A,
                self.B, 
                latest_obs,
                self.policies, 
                latest_actions, 
                prior = self.latest_belief, 
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )

            self.F = F # variational free energy of each policy  

        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)

        self.qs = qs

        if self.inference_algo == "MMP":
            return qs, xn, vn
        else:
            return qs
    
    def infer_policies(self):
        """
        POLICY INFERENCE: The Core Decision-Making Process
        
        This is where the agent decides what to do next by evaluating different possible
        action sequences (policies). This is like the "planning" and "decision-making" 
        parts of the brain working together.
        
        THE BASIC IDEA:
        The agent imagines what would happen if it followed different action sequences:
        1. What observations would it see? (prediction using A and B matrices)
        2. How much would it like those observations? (evaluation using C preferences)
        3. How much would it learn? (information gain calculations)
        4. Which action sequence gives the best combination of rewards and learning?
        
        MATHEMATICAL FOUNDATION:
        The agent computes "Expected Free Energy" G for each policy π:
        G(π) = Expected Utility + Expected Information Gain
        
        Where:
        - Expected Utility = how much the agent expects to like future observations
        - Expected Information Gain = how much the agent expects to learn
        
        Then it chooses policies with probability:
        P(π) ∝ exp(-γ × G(π) + ln E(π))
        
        Where:
        - γ (gamma) controls how much the agent trusts its evaluations
        - E(π) represents prior preferences over policies

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies - probability distribution over action sequences.
            Higher values = more preferred policies.
            Example: [0.1, 0.7, 0.2] means 70% preference for policy 1, etc.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy.
            Lower values = better policies (less expected "cost" or "surprise").
            These are the raw scores before converting to probabilities.
        """

        ### VANILLA ALGORITHM: Simple Policy Evaluation ###
        if self.inference_algo == "VANILLA":
            
            ### Sophisticated Inference: Tree Search Planning ###
            if self.sophisticated:
                # Uses tree search to explore deep policy sequences
                # This is computationally expensive but can find very good long-term plans
                q_pi, G = control.sophisticated_inference_search(
                    self.qs,                         # Current state beliefs
                    self.policies,                   # All possible action sequences  
                    self.A,                         # Observation model
                    self.B,                         # Transition model
                    self.C,                         # Preferences
                    self.A_factor_list,             # Factorization for efficiency
                    self.B_factor_list,
                    self.I,                         # Value function (from backwards induction)
                    self.si_horizon,                # How deep to search
                    self.si_policy_prune_threshold, # Ignore very bad policies
                    self.si_state_prune_threshold,  # Ignore very unlikely states
                    self.si_prune_penalty,          # Cost of exploring pruned branches
                    1.0,                           # Temperature parameter
                    self.inference_params,          # Algorithm settings
                    n=0                            # Recursion depth
                )
            else:
                ### Standard Inference: Direct Policy Evaluation ###
                # Efficiently computes expected free energy for each policy
                q_pi, G = control.update_posterior_policies_factorized(
                    self.qs,                    # Current beliefs about states
                    self.A,                     # Observation model
                    self.B,                     # Transition model  
                    self.C,                     # Preferences (what agent wants to see)
                    self.A_factor_list,         # Which factors affect which observations
                    self.B_factor_list,         # Which factors affect which transitions
                    self.policies,              # All possible action sequences to evaluate
                    self.use_utility,           # Whether to seek preferred observations
                    self.use_states_info_gain,  # Whether to seek info about states
                    self.use_param_info_gain,   # Whether to seek info about model parameters
                    self.pA,                    # Learning parameters for A matrix
                    self.pB,                    # Learning parameters for B matrix
                    E = self.E,                 # Prior preferences over policies
                    I = self.I,                 # Value function (if using backwards induction)
                    gamma = self.gamma          # Policy precision parameter
                )
                
        ### MMP ALGORITHM: Complex Temporal Planning ###
        elif self.inference_algo == "MMP":

            # Extract future-oriented beliefs from the temporal inference
            # Only use beliefs about future states, not past states
            future_qs_seq = self.get_future_qs()

            # Compute policy values using the full temporal model
            # This considers how policies affect states over multiple timesteps
            q_pi, G = control.update_posterior_policies_full_factorized(
                future_qs_seq,               # Beliefs about future states under each policy
                self.A,                      # Observation model
                self.B,                      # Transition model
                self.C,                      # Preferences
                self.A_factor_list,          # Factorization structure
                self.B_factor_list,
                self.policies,               # Action sequences to evaluate
                self.use_utility,            # Goal-seeking behavior
                self.use_states_info_gain,   # Curiosity about states
                self.use_param_info_gain,    # Curiosity about learning
                self.latest_belief,          # Beliefs at start of planning horizon
                self.pA,                     # Learning parameters
                self.pB,
                F=self.F,                    # Variational free energy from state inference
                E=self.E,                    # Policy priors
                I=self.I,                    # Value function
                gamma=self.gamma             # Policy precision
            )

        # Store results for history tracking and future use
        if hasattr(self, "q_pi_hist"):
            self.q_pi_hist.append(q_pi)
            # Keep only recent history to limit memory usage
            if len(self.q_pi_hist) > self.inference_horizon:
                self.q_pi_hist = self.q_pi_hist[-(self.inference_horizon-1):]

        # Cache results in agent for other methods to use
        self.q_pi = q_pi    # Posterior over policies
        self.G = G          # Expected free energies
        
        return q_pi, G

    def sample_action(self):
        """
        ACTION SELECTION: Converting Decisions into Actions
        
        This is where the agent converts its policy preferences into an actual action
        to take in the world. This is like the "motor control" part of the brain that
        translates decisions into movement.
        
        THE BASIC IDEA:
        The agent has computed preferences over different action sequences (policies).
        Now it needs to decide what specific action to take RIGHT NOW. There are two ways:
        
        1. MARGINAL SAMPLING: Look at all policies and find the most popular first action
           - Like asking "what's the most common first move across all good plans?"
           - This integrates over uncertainty about which long-term plan is best
           
        2. FULL SAMPLING: Pick one complete policy and follow its first action  
           - Like committing to one specific plan and starting to execute it
           - This might be more decisive but ignores uncertainty
        
        MATHEMATICAL FOUNDATION:
        The action is chosen with probability:
        P(action) ∝ exp(α × preference_for_action)
        
        Where:
        - α (alpha) controls randomness vs determinism
        - Higher α = more deterministic (always pick best action)
        - Lower α = more random (explore different actions)

        Returns
        ----------
        action: 1D ``numpy.ndarray``
            Vector containing the indices of the actions for each control factor.
            Example: [1, 0, 2] might mean "move right, don't touch switch, press button C"
            Each number corresponds to which action to take for each controllable factor.
        """

        ### MARGINAL SAMPLING: Integrate Over Policy Uncertainty ###
        if self.sampling_mode == "marginal":
            # Compute marginal action preferences by averaging over all policies
            # This asks: "What action is most popular across all good policies?"
            action = control.sample_action(
                self.q_pi,                    # Policy preferences from infer_policies()
                self.policies,                # All possible action sequences
                self.num_controls,            # Number of actions for each factor
                action_selection = self.action_selection,  # "deterministic" or "stochastic"
                alpha = self.alpha            # Action precision (randomness vs determinism)
            )
            
        ### FULL SAMPLING: Commit to One Policy ###
        elif self.sampling_mode == "full":
            # First pick one complete policy, then take its first action
            # This asks: "What's my best complete plan, and what's its first step?"
            action = control.sample_policy(
                self.q_pi,                    # Policy preferences
                self.policies,                # All possible action sequences  
                self.num_controls,            # Number of actions for each factor
                action_selection=self.action_selection,    # Selection method
                alpha=self.alpha              # Action precision
            )

        # Store the chosen action and advance time
        self.action = action    # Cache action for next inference cycle
        self.step_time()        # Update time and manage belief histories

        return action
    
    def _sample_action_test(self):
        """
        Sample or select a discrete action from the posterior over control states.
        This function both sets or cachés the action as an internal variable with the agent and returns it.
        This function also updates time variable (and thus manages consequences of updating the moving reference frame of beliefs)
        using ``self.step_time()``.
        
        Returns
        ----------
        action: 1D ``numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """

        if self.sampling_mode == "marginal":
            action, p_dist = control._sample_action_test(self.q_pi, self.policies, self.num_controls,
                                                         action_selection=self.action_selection, alpha=self.alpha)
        elif self.sampling_mode == "full":
            action, p_dist = control._sample_policy_test(self.q_pi, self.policies, self.num_controls,
                                                         action_selection=self.action_selection, alpha=self.alpha)

        self.action = action

        self.step_time()

        return action, p_dist

    def update_A(self, obs):
        """
        OBSERVATION MODEL LEARNING: "I was wrong about what I'd see"
        
        This is where the agent learns to improve its observation model (A matrix) based on experience.
        The A matrix encodes: "If the world is in state X, what observation will I see?"
        
        THE BASIC IDEA:
        The agent predicted it would see certain observations given its beliefs about the current state.
        But it actually observed something specific. If there's a mismatch, the agent should update
        its A matrix to better predict what it will see in similar situations in the future.
        
        MATHEMATICAL FOUNDATION:
        Uses Bayesian learning with Dirichlet distributions. The Dirichlet is the "uncertainty 
        distribution" over probability distributions - it represents how confident the agent is
        about its A matrix parameters.
        
        The update rule is:
        qA_new = qA_old + learning_rate × (observation_evidence)
        
        Where:
        - Higher learning_rate = faster adaptation to new observations
        - observation_evidence = how much this observation supports different A matrix entries
        
        EXAMPLE:
        - Agent believes: "In bright rooms, I usually see brightness"  
        - Agent observes: "I'm in a bright room but see darkness"
        - Learning: "Maybe bright rooms aren't always bright - update my A matrix"

        Parameters
        ----------
        obs: ``list`` or ``tuple`` of ints
            The observation that was actually received from the environment.
            Each entry ``obs[m]`` stores the index of the discrete observation for modality ``m``.
            Example: [2, 1] = "observed brightness level 2, sound level 1"

        Returns
        -----------
        qA: ``numpy.ndarray`` of dtype object
            Updated Dirichlet parameters over observation model (same shape as ``A``).
            These represent the agent's new uncertainty/confidence about its observation model.
            Higher values = more confident about those A matrix entries.
        """

        # Update the Dirichlet parameters using factorized learning
        # This efficiently handles cases where observations depend on multiple hidden factors
        qA = learning.update_obs_likelihood_dirichlet_factorized(
            self.pA,                    # Current Dirichlet parameters (agent's uncertainty about A)
            self.A,                     # Current observation model 
            obs,                        # What was actually observed
            self.qs,                    # Agent's current beliefs about hidden states
            self.A_factor_list,         # Which hidden factors affect which observations
            self.lr_pA,                 # Learning rate (how fast to adapt)
            self.modalities_to_learn    # Which observation types to update
        )

        # Update the agent's internal models
        self.pA = qA                                    # Store new uncertainty parameters
        self.A = utils.norm_dist_obj_arr(qA)           # Convert to new observation model
        # The new A matrix is the "expected value" of the Dirichlet distribution

        return qA

    def _update_A_old(self, obs):
        """
        Update approximate posterior beliefs about Dirichlet parameters that parameterise the observation likelihood or ``A`` array.

        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.

        Returns
        -----------
        qA: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over observation model (same shape as ``A``), after having updated it with observations.
        """

        qA = learning.update_obs_likelihood_dirichlet(
            self.pA, 
            self.A, 
            obs, 
            self.qs, 
            self.lr_pA, 
            self.modalities_to_learn
        )

        self.pA = qA # set new prior to posterior
        self.A = utils.norm_dist_obj_arr(qA) # take expected value of posterior Dirichlet parameters to calculate posterior over A array

        return qA

    def update_B(self, qs_prev):
        """
        TRANSITION MODEL LEARNING: "I was wrong about what my actions do"
        
        This is where the agent learns to improve its transition model (B matrix) based on experience.
        The B matrix encodes: "If I take action A when in state X, what state will I end up in?"
        
        THE BASIC IDEA:
        The agent took an action expecting to end up in a certain state. After taking the action,
        it inferred what state it actually ended up in. If there's a mismatch between expectation
        and reality, the agent should update its B matrix to better predict the effects of actions.
        
        MATHEMATICAL FOUNDATION:
        Uses Bayesian learning with Dirichlet distributions for transition probabilities.
        The update considers the transition: (previous_state, action) → current_state
        
        The update rule is:
        qB_new = qB_old + learning_rate × (transition_evidence)
        
        Where:
        - transition_evidence = how much this experience supports different B matrix entries
        - The evidence is weighted by the agent's confidence in its state beliefs
        
        EXAMPLE:
        - Agent believes: "When I press the light switch while it's OFF, it turns ON"
        - Agent takes action: "Press switch" from state "OFF"  
        - Agent observes: Still in state "OFF" (switch was broken)
        - Learning: "Maybe pressing switches doesn't always work - update my B matrix"

        Parameters
        -----------
        qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
            The agent's beliefs about what hidden states it was in at the previous timestep.
            This is needed because learning requires knowing the transition:
            previous_state + action → current_state
    
        Returns
        -----------
        qB: ``numpy.ndarray`` of dtype object
            Updated Dirichlet parameters over transition model (same shape as ``B``).
            These represent the agent's new uncertainty/confidence about its transition model.
            Higher values = more confident about those B matrix entries.
        """

        # Update the Dirichlet parameters using factorized learning
        # This efficiently handles interactions between different hidden state factors
        qB = learning.update_state_likelihood_dirichlet_interactions(
            self.pB,                    # Current Dirichlet parameters (uncertainty about B)
            self.B,                     # Current transition model
            self.action,                # Action that was taken
            self.qs,                    # Current beliefs about states (where we ended up)
            qs_prev,                    # Previous beliefs about states (where we started)
            self.B_factor_list,         # Which factors can influence which other factors
            self.lr_pB,                 # Learning rate (how fast to adapt)
            self.factors_to_learn       # Which hidden state factors to update
        )

        # Update the agent's internal models
        self.pB = qB                                    # Store new uncertainty parameters
        self.B = utils.norm_dist_obj_arr(qB)           # Convert to new transition model
        # The new B matrix is the "expected value" of the Dirichlet distribution

        return qB
    
    def _update_B_old(self, qs_prev):
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the transition likelihood 
        
        Parameters
        -----------
        qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
            Marginal posterior beliefs over hidden states at previous timepoint.
    
        Returns
        -----------
        qB: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over transition model (same shape as ``B``), after having updated it with state beliefs and actions.
        """

        qB = learning.update_state_likelihood_dirichlet(
            self.pB,
            self.B,
            self.action,
            self.qs,
            qs_prev,
            self.lr_pB,
            self.factors_to_learn
        )

        self.pB = qB # set new prior to posterior
        self.B = utils.norm_dist_obj_arr(qB)  # take expected value of posterior Dirichlet parameters to calculate posterior over B array

        return qB
    
    def update_D(self, qs_t0 = None):
        """
        INITIAL STATE LEARNING: "I was wrong about likely starting states"
        
        This is where the agent learns to improve its initial state beliefs (D vector) based on experience.
        The D vector encodes: "When episodes begin, what state is the world likely to start in?"
        
        THE BASIC IDEA:
        Over time, the agent observes what states the world actually starts in at the beginning
        of episodes or inference windows. If there's a pattern (e.g., episodes usually start in
        the kitchen rather than the bedroom), the agent should update its D vector to reflect
        this learned knowledge about typical starting conditions.
        
        MATHEMATICAL FOUNDATION:
        Uses Bayesian learning with Dirichlet distributions for initial state probabilities.
        The update considers what state the agent believes it was in at the beginning of 
        the current inference window.
        
        The update rule is:
        qD_new = qD_old + learning_rate × (initial_state_evidence)
        
        Where:
        - initial_state_evidence = how much this episode supports different starting states
        - This helps the agent learn typical environmental patterns
        
        EXAMPLE:
        - Agent assumes: "Episodes equally likely to start in any room"
        - Experience: "Episodes usually start in the living room"  
        - Learning: "Update D to reflect that living room is the most common starting location"

        Parameters
        -----------
        qs_t0: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, or ``None``
            The agent's beliefs about what hidden states it was in at the beginning of the current
            inference window. This provides evidence about typical starting states.
            If None, the method will try to automatically determine this from the agent's history.
      
        Returns
        -----------
        qD: ``numpy.ndarray`` of dtype object
            Updated Dirichlet parameters over initial hidden state prior (same shape as ``D``).
            These represent the agent's new uncertainty/confidence about its initial state model.
            Higher values = more confident about those starting state probabilities.
        """
        
        ### VANILLA ALGORITHM: Use First Timestep Beliefs ###
        if self.inference_algo == "VANILLA":
            
            if qs_t0 is None:
                # Try to get beliefs from the first timestep of this episode
                try:
                    qs_t0 = self.qs_hist[0]  # First recorded state belief
                except (ValueError, IndexError, AttributeError):
                    print("qs_t0 must either be passed as argument to `update_D` or `save_belief_hist` must be set to True!")
                    return self.pD  # Return unchanged if we can't get the data

        ### MMP ALGORITHM: Use Beliefs from Inference Horizon ###
        elif self.inference_algo == "MMP":
            
            if self.edge_handling_params['use_BMA']:
                # Use Bayesian Model Average of beliefs at horizon start
                qs_t0 = self.latest_belief
                
            elif self.edge_handling_params['policy_sep_prior']:
                # Use policy-separated beliefs, then average over policies
                qs_pi_t0 = self.latest_belief  # Policy-conditioned beliefs

                # Get the policy preferences from the beginning of the inference horizon
                if hasattr(self, "q_pi_hist"):
                    begin_horizon_step = max(0, self.curr_timestep - self.inference_horizon)
                    q_pi_t0 = np.copy(self.q_pi_hist[begin_horizon_step])
                else:
                    q_pi_t0 = np.copy(self.q_pi)
            
                # Average state beliefs over policies using policy preferences
                qs_t0 = inference.average_states_over_policies(qs_pi_t0, q_pi_t0)
        
        # Update the Dirichlet parameters for initial state distribution
        qD = learning.update_state_prior_dirichlet(
            self.pD,                    # Current Dirichlet parameters (uncertainty about D)
            qs_t0,                      # Evidence about initial state from this episode
            self.lr_pD,                 # Learning rate (how fast to adapt)
            factors = self.factors_to_learn  # Which hidden state factors to update
        )
        
        # Update the agent's internal models
        self.pD = qD                                    # Store new uncertainty parameters
        self.D = utils.norm_dist_obj_arr(qD)           # Convert to new initial state prior
        # The new D vector is the "expected value" of the Dirichlet distribution

        return qD

    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001, "compute_vfe": True}
        elif method == "MMP":
            default_params = {"num_iter": 10, "grad_descent": True, "tau": 0.25}
        elif method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        elif method == "BP":
            raise NotImplementedError("BP is not implemented")
        elif method == "EP":
            raise NotImplementedError("EP is not implemented")
        elif method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params

    
    
