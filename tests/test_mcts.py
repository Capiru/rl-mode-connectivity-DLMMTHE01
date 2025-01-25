from rl.env import get_env
from rl.utils import get_model
from rl.mcts import MCTS, Node
import numpy as np
import chess as ch
import torch
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


def test_select_best_child():
    assert True


def test_backup_depth_1(cfg):
    cfg.env_type = "connect4"
    cfg.num_simulations = 1
    env = get_env("connect4")
    model = get_model(cfg.model_type, cfg)
    assert model
    mcts = MCTS(model=model, eval=True, eps=0.3, cfg=cfg)
    node = Node(
        state=env,
        mcts=mcts,
    )

    leaf = node.select()
    leaf.expand(0, 1)
    leaf.backup(1)

    assert node.child_total_value[leaf.action] == -1

    assert node.number_visits == 1


def test_backup_depth_2(cfg):
    cfg.env_type = "connect4"
    cfg.num_simulations = 1
    env = get_env("connect4")
    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    node = Node(
        state=env,
        mcts=mcts,
    )
    leaf = node.select()
    leaf.expand(0, 0)
    leaf.backup(0)

    leaf_2nd = leaf.select()
    leaf_2nd.expand(0, 1)
    leaf_2nd.backup(1)

    assert node.child_total_value[leaf.action] == 1
    assert leaf.child_total_value[leaf_2nd.action] == -1

    assert node.number_visits == 2
    assert leaf.number_visits == 2
    assert leaf_2nd.number_visits == 1


def test_backup_depth_3(cfg):
    cfg.env_type = "connect4"
    cfg.num_simulations = 1
    env = get_env("connect4")
    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    node = Node(
        state=env,
        mcts=mcts,
    )
    leaf = node.select()
    leaf.expand(0, 0)
    leaf.backup(0)

    leaf_2nd = leaf.select()
    leaf_2nd.expand(0, 0)
    leaf_2nd.backup(0)

    leaf_3rd = leaf_2nd.select()
    leaf_3rd.expand(0, 1)
    leaf_3rd.backup(1)

    assert node.child_total_value[leaf.action] == -1
    assert leaf.child_total_value[leaf_2nd.action] == 1
    assert leaf_2nd.child_total_value[leaf_3rd.action] == -1

    assert node.number_visits == 3
    assert leaf.number_visits == 3
    assert leaf_2nd.number_visits == 2
    assert leaf_3rd.number_visits == 1


def test_win_in_one_c4(cfg):
    cfg.env_type = "connect4"
    cfg.num_simulations = 150
    env = get_env("connect4")
    env.step(0)
    env.step(6)

    env.step(1)
    env.step(5)

    env.step(2)
    env.step(6)

    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    mate_in_1_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_1_node)
    print(action)
    env.step(action)
    _, _, done, _, _ = env.last()

    assert done
    assert action == 3


def test_win_in_two_c4(cfg):
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

    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    mate_in_2_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_2_node)
    print(action)
    # plt.imshow(env.render())
    # plt.title(f"{action}")
    # plt.show()
    env.step(action)
    _, _, done, _, _ = env.last()
    assert action == 1 or action == 4


def test_defense_in_one_c4(cfg):
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

    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    mate_in_2_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_2_node)
    print(action)
    # plt.imshow(env.render())
    # plt.title(f"{action}")
    # plt.show()
    env.step(action)
    _, _, done, _, _ = env.last()
    assert action == 4


class TestModel:
    def forward(self, x):
        return torch.tensor([0.1, 0.0, 0.1, 0.1, 0, 0.6, 0.1]), torch.tensor([0.7])

    def __call__(self, x):
        return self.forward(x)


def test_value_policy_networks(cfg):
    cfg.env_type = "connect4"
    cfg.num_simulations = 200
    env = get_env("connect4")
    mcts = MCTS(TestModel(), eval=True, eps=0.3, cfg=cfg)
    mate_in_2_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_2_node)
    assert action == 5


def test_mate_in_one(cfg, render=False):
    cfg.env_type = "chess"
    cfg.num_simulations = 600
    cfg.eps = 0
    env = get_env("chess")
    env.board.push_uci("e2e4")
    env.board.push_uci("e7e5")
    env.board.push_uci("f1c4")
    env.board.push_uci("f8c5")
    env.board.push_uci("d1h5")
    env.board.push_uci("b8c6")
    mcts = MCTS(get_model(cfg=cfg), eval=True, eps=0.3, cfg=cfg)
    mate_in_1_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_1_node)
    print(action)
    env.step(action)
    _, _, done, _, _ = env.last()
    if render:
        plt.imshow(env.render())
        plt.title(f"{action}")
        plt.show()
    assert done
    assert action == 4390


def test_mate_in_two(cfg, render=False):
    puzzle_fen = "1k6/ppp5/8/8/4Qn2/5q2/5P1P/6K1 b - - 2 21"
    cfg.env_type = "chess"
    cfg.num_simulations = 1600
    env = get_env("chess")
    env.board = ch.Board(puzzle_fen)
    env.agent_selection = env._agent_selector.next()
    mcts = MCTS(get_model(cfg=cfg), eval=False, eps=0.1, cfg=cfg)
    mate_in_2_node = Node(
        state=env,
        mcts=mcts,
    )
    tp, action, node = mcts.compute_action(mate_in_2_node)
    print(action)
    env.step(action)
    if render:
        plt.imshow(env.render())
        plt.title(f"{action}")
        plt.show()
    assert env.board.is_check()
    assert action == 3295 or action == 3275
