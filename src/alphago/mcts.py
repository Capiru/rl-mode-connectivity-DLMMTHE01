"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import copy
import math

import numpy as np
import torch
from torch.distributions import Categorical

from alphago.config import cfg
from alphago.env import get_env

import asyncio
import concurrent.futures
from typing import Optional

def env_get_copy(env,raw_env = None):
    if raw_env is None:
        new_env = get_env()
    else:
        new_env = raw_env
    if cfg.env_type == "connect4":
        new_env.board[:] = env.board[:]
    elif cfg.env_type == "chess":
        new_env.board = copy.deepcopy(env.board)
    # Only works for two player environments
    if not new_env.agent_selection == env.agent_selection:
        new_env.agent_selection = new_env._agent_selector.next()
    return new_env

_executor: Optional[concurrent.futures.Executor] = None
_env_future: Optional[asyncio.Future] = None

def _init_executor():
    """Initialize the global executor if it is None."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def env_get_copy_async(env, env_type: str = cfg.env_type):
    """
    Returns a copy of the given src_env, but also manages a future
    that asynchronously prepares a fresh environment for next time.
    """
    global _env_future

    # Ensure we have an executor to run sync code in background threads
    _init_executor()

    # 1. Check if an existing future for a new environment is pending/available.
    if _env_future is None:
        # Optionally cancel any other pending futures here if you want:
        #   for f in all_other_futures:
        #       f.cancel()

        # Create a new background future to fetch a fresh environment
        loop = asyncio.get_running_loop()
        _env_future = loop.run_in_executor(_executor, get_env, env_type)

    
    copied_env = _env_future.result()
    # 2. Actually get a copy of the existing environment, *awaiting* the sync call in a thread.
    loop = asyncio.get_running_loop()
    new_env = await loop.run_in_executor(_executor, env_get_copy, env, copied_env)
    # 3. Schedule (non-blocking) the next environment initialization for the future.
    #    This means by the time we come back here next time, hopefully it's done.
    async def schedule_next_env():
        # If you want to store it in the global _env_future again, you can:
        global _env_future
        _env_future = loop.run_in_executor(_executor, get_env, env_type)

    # Create a background task so it does not block returning new_env.
    asyncio.create_task(schedule_next_env())

    return new_env

class Node:
    def __init__(
        self, state, mcts, action = None, obs = None, done = None, reward = None, parent=None, multi_agent=False
    ):
        self.state = state
        self.initial_state = env_get_copy(state)
        self.parent = parent
        self.children = {}
        self.action_space_size = self.state.action_space(self.state.agent_selection).n
        # Multiply by a number to incentivise exploration
        self.child_total_value = (
            np.ones([self.action_space_size], dtype=np.float32)
        )  # Q
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # N
        if parent:
            self.reward = reward
            self.done = done
            self.state = state
            self.obs = obs
            self.action = action  # Action used to go to this state
            self.is_expanded = False
            self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        else:
            self.obs,self.reward,self.done,_ ,_ = state.last()
            self.is_expanded = True

            self.child_priors,self._total_value = mcts.model(torch.tensor(self.obs["observation"].copy(),dtype = torch.float32))
            self.child_priors = self.child_priors.detach().numpy()

        current_agents = self.state.agents
        current_agent = self.state.agent_selection
        self.agent = current_agent
        self.mcts = mcts
        if len(current_agents)>1:
            self.multi_agent = False
        if self.multi_agent:
            self.reward = self.reward[current_agent]
            self.done = self.done[current_agent]
            self.valid_actions = obs[current_agent]["action_mask"].astype(bool)
            self.obs = obs[current_agent]
        else:
            self.valid_actions = self.obs["action_mask"].astype(bool)
            self.obs = torch.tensor(self.obs["observation"].copy(),dtype = torch.float32)
        if np.sum(self.valid_actions.astype(int)) > 1 and (self.mcts.cfg.env_type == "go" or self.mcts.cfg.env_type == "go_5"):
            self.valid_actions[-1] = False
        

    @property
    def number_visits(self):
        if self.parent:
            return self.parent.child_number_visits[self.action]
        else:
            return np.sum(self.child_number_visits)

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        if self.parent:
            return self.parent.child_total_value[self.action]
        else:
            return self._total_value

    @total_value.setter
    def total_value(self, value):
        if self.parent:
            self.parent.child_total_value[self.action] = value
        else:
            self._total_value = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )

    def best_action(self):
        """
        :return: action
        """
        epsilon_exploration = np.random.choice(
            [True, False], p=[self.mcts.epsilon, 1 - self.mcts.epsilon]
        )
        if epsilon_exploration and not self.mcts.eval:
            m = Categorical(torch.tensor(self.valid_actions.astype(float) ))
            action = int(m.sample().detach().cpu().numpy().astype(np.int64))
            return action
        
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        # We set a value for stability in the probs and softmax
        masked_child_score[~self.valid_actions] = 0 #-1e12
        # if self.mcts.eval:
        #     action = np.argmax(masked_child_score)
        #     assert self.valid_actions[action] == 1
        #     return action
        # else:
        masked_child_score -= min(masked_child_score)
        masked_child_score[self.valid_actions] += 1e-8
        masked_child_score[~self.valid_actions] = 1e-18

            # This might be bad for performance
            #m = Categorical((torch.tensor(masked_child_score)+ 1e-9)*torch.tensor(self.valid_actions) )
            #action = int(m.sample().detach().cpu())
        _max_count = 5
        while _max_count >0:
            action = np.random.choice(
                np.arange(len(masked_child_score)),
                p=masked_child_score / np.sum(masked_child_score),
            )
            _max_count -= 1
            if self.valid_actions[action] == 1:
                return action
        print(action, self.done,masked_child_score)
        masked_child_score[~self.valid_actions] = 0
        chosen_action = np.argmax(masked_child_score)
        assert self.valid_actions[action] == 1
        return chosen_action


    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors, value):
        self.is_expanded = True
        self.total_value = -value
        self.initial_value = -value
        self.child_priors = child_priors
        self.number_visits = 1

    def get_child(self, action):
        if action not in self.children:
            self.state.step(action)
            obs, reward, done, _, _ = self.state.last()
            try:
                reward = self.state.rewards[self.state.agent_selection]
            except KeyError:
                reward = 0.0
            next_state = env_get_copy(self.state)
            self.state = env_get_copy(self.initial_state)
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts,
            )
        return self.children[action]
    
    def backup(self, value):
        current_node = self.parent
        while current_node.parent is not None:
            current_node.number_visits += 1
            current_node.total_value += value
            current_node = current_node.parent
            if current_node.mcts.turn_based_flip:
                value = -value



class MCTS:
    def __init__(self, model, eval = False, eps = None,num_sims = None, cfg = cfg):
        if not eps:
            eps = cfg.eps
        if not num_sims:
            num_sims = cfg.num_simulations
        
        self.model = model
        self.temperature = cfg.temperature
        self.dir_epsilon = cfg.dirichlet_epsilon
        self.dir_noise = cfg.dirichlet_noise
        self.num_sims = num_sims
        self.add_dirichlet_noise = cfg.add_dirichlet_noise
        self.c_puct = cfg.puct_coefficient
        self.epsilon = eps
        self.turn_based_flip = cfg.turn_based_flip
        self.eval = eval

        self.cfg = cfg
        self.reward_multiplier = 1
    
        
    def compute_action(self, node):
        show_render = 0
        for sim_num in range(self.num_sims):
            node.state = env_get_copy(node.initial_state)
            leaf = node.select()
            if leaf.done:
                value = -leaf.reward * self.reward_multiplier
                leaf.number_visits += 1
                leaf.total_value = value
            else:
                child_priors, value = self.model(leaf.obs)
                # value = -value
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )

                leaf.expand(child_priors.detach().numpy(), value.detach().numpy())
            leaf.backup(-value)

        # Tree policy target (TPT)

        tree_policy = node.child_number_visits / self.num_sims
        
        #tree_policy = node.child_Q() + node.mcts.c_puct * node.child_U()
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy[~node.valid_actions] = 0
        #tree_policy = tree_policy / np.max(tree_policy)  # to avoid overflows when computing softmax
        
        if sum(node.valid_actions)>1:
            tree_policy -= np.min(tree_policy)
            tree_policy[~node.valid_actions] = + 1e-29
        tree_policy = tree_policy/ np.sum(tree_policy)

        epsilon_exploration = np.random.choice(
            [True, False], p=[self.epsilon, 1 - self.epsilon]
        )
        epsilon_exploration = False
        if self.eval:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        elif  epsilon_exploration:
            action = np.random.choice(np.arange(node.action_space_size))
        else:
            # otherwise sample an action according to tree policy probabilities
            try:
                action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
            except Exception as e:
                print(tree_policy, node.valid_actions, node.child_Q() + node.mcts.c_puct * node.child_U())
                print(node.child_Q() + node.mcts.c_puct * node.child_U() / np.max(node.child_Q() + node.mcts.c_puct * node.child_U()))
                print(np.power(node.child_Q() + node.mcts.c_puct * node.child_U(),self.temperature))
        assert node.valid_actions[action] == 1
        node.state = env_get_copy(node.initial_state)
        try:
            node.children[action].parent = None
            node.children[action].env = node.children[action].initial_state
        except KeyError:
            return tree_policy, action, None
        #print(np.max(tree_policy), tree_policy[4390], node.children[action].number_visits, node.children[4390].number_visits, node.child_Q()[action], node.child_Q()[4390])
        return tree_policy, action, node.children[action]
