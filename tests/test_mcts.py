from alphago.env import get_env
from alphago.utils import get_model
from alphago.config import cfg
from alphago.mcts import MCTS, Node
import numpy as np
import chess as ch
import torch
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

def test_win_in_one_c4():
    cfg.env_type = "connect4"
    cfg.num_simulations = 50
    env = get_env("connect4")
    env.step(0)
    env.step(6)
    
    env.step(1)
    env.step(5)
    
    env.step(2)
    env.step(6)
    
    mcts = MCTS(get_model(),eval = True,eps = 0.3)
    mate_in_1_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_1_node)
    print(action)
    env.step(action)
    _,_,done, _,_ = env.last()

    assert done
    assert action == 3
    
def test_win_in_two_c4():
    cfg.env_type = "connect4"
    cfg.num_simulations = 400
    env = get_env("connect4")
    env.step(2)
    env.step(4)
    
    env.step(3)
    env.step(3)
    
    env.step(5)
    env.step(5)
    
    env.step(2)
    env.step(4)
    
    env.step(3)
    env.step(2)
    
    env.step(5)
    env.step(6)
    
    env.step(6)
    env.step(5)
    
    env.step(4)
    env.step(6)
    
    mcts = MCTS(get_model(),eval = True,eps = 0.3)
    mate_in_2_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_2_node)
    print(action)
    # plt.imshow(env.render())
    # plt.title(f"{action}")
    # plt.show()
    env.step(action)
    _,_,done, _,_ = env.last()
    assert action == 1 or action == 4
    
def test_defense_in_one_c4():
    cfg.env_type = "connect4"
    cfg.num_simulations = 200
    env = get_env("connect4")
    env.step(2)
    env.step(4)
    
    env.step(3)
    env.step(3)
    
    env.step(5)
    env.step(5)
    
    env.step(2)
    env.step(4)
    
    env.step(3)
    env.step(2)
    
    env.step(5)
    env.step(6)
    
    env.step(6)
    env.step(5)
    
    env.step(4)
    env.step(6)
    
    env.step(5)
    
    mcts = MCTS(get_model(),eval = True,eps = 0.3)
    mate_in_2_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_2_node)
    print(action)
    # plt.imshow(env.render())
    # plt.title(f"{action}")
    # plt.show()
    env.step(action)
    _,_,done, _,_ = env.last()
    assert action == 4

class TestModel():
    def forward(self,x):
        return torch.tensor([0.1,0.0,0.1,0.1,0,0.6,0.1]),torch.tensor([0.7]) 
    
    def __call__(self,x):
        return self.forward(x)
    

def test_value_policy_networks():
    cfg.env_type = "connect4"
    cfg.num_simulations = 200
    env = get_env("connect4")
    mcts = MCTS(TestModel(),eval = True,eps = 0.3)
    mate_in_2_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_2_node)
    assert action == 5

def test_mate_in_one():
    cfg.env_type = "chess"
    cfg.num_simulations = 1000
    cfg.eps = 0
    env = get_env("chess")
    env.board.push_uci("e2e4")
    env.board.push_uci("e7e5")
    env.board.push_uci("f1c4")
    env.board.push_uci("f8c5")
    env.board.push_uci("d1h5")
    env.board.push_uci("b8c6")
    mcts = MCTS(get_model(),eval = True,eps = 0.3)
    mate_in_1_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_1_node)
    print(action)
    env.step(action)
    _,_,done, _,_ = env.last()
    plt.imshow(env.render())
    plt.title(f"{action}")
    plt.show()
    assert done
    assert action == 4390
    
def test_mate_in_two():
    puzzle_fen = "1k6/ppp5/8/8/4Qn2/5q2/5P1P/6K1 b - - 2 21"
    cfg.env_type = "chess"
    cfg.num_simulations = 2000
    env = get_env("chess")
    env.board = ch.Board(puzzle_fen)
    env.agent_selection = env._agent_selector.next()
    mcts = MCTS(get_model(),eval = False,eps = 0.1)
    mate_in_2_node = Node(
            state=env,
            mcts=mcts,
        )
    tp,action,node = mcts.compute_action(mate_in_2_node)
    print(action)
    env.step(action)

    plt.imshow(env.render())
    plt.title(f"{action}")
    plt.show()
    assert env.board.is_check()
    assert action == 3295