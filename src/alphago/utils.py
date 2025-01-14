import os
import torch
import torch.nn as nn
from torch.distributions import Categorical

from alphago.config import cfg
from alphago.mcts import MCTS, Node
from alphago.model import SimpleConvnet, SimpleModel, AlphaGoZeroResnet

import pandas as pd
import numpy as np
import math
import random

import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


### Util Functions
def play_move(env, agent, model, eval=False, eps=None, num_sims=None, cfg=cfg):
    if not eps:
        eps = cfg.eps
    if not num_sims:
        num_sims = cfg.num_simulations
    observation, previous_reward, termination, truncation, info = env.last()
    return_action_probs = False
    if termination or truncation:
        action = None
    else:
        mask = torch.tensor(observation["action_mask"], dtype=torch.float32)
        if cfg.env_type == "go":
            mask[-1] = 1
        if (
            isinstance(model, nn.Module)
            and (random.random() < 1 - eps or eval)
            and num_sims == 0
        ):
            policy, value = model(
                torch.tensor(observation["observation"].copy(), dtype=torch.float32)
            )
            if cfg.env_type == "go":
                if torch.count_nonzero(mask) == 0:
                    mask[-1] = 1
            m = Categorical((policy + 1e-9) * mask)
            action = m.sample().detach().cpu().numpy().astype(np.int64)
            logger.debug(f"Action chosen - {action}")
        elif isinstance(model, nn.Module) and num_sims > 0:
            mcts = MCTS(model=model, eval=eval, eps=eps, num_sims=num_sims)
            tp, action, node = mcts.compute_action(
                Node(
                    state=env,
                    mcts=mcts,
                )
            )
            return_action_probs = True
        else:
            action = env.action_space(agent).sample(observation["action_mask"])
    env.step(action)

    _, _, done, _, _ = env.last()
    try:
        reward = env.rewards[agent]
    except Exception:
        reward = 0

    if return_action_probs:
        return observation, reward, tp, done
    elif isinstance(action, np.ndarray):
        action = int(action)
    return observation, reward, action, done


def find_winner(reward, cfg=cfg):
    if reward == -1.0:
        return cfg.agents[cfg.env_type][1]
    elif reward == 1.0:
        return cfg.agents[cfg.env_type][0]
    elif reward == 0.0 or reward == cfg.draw_reward:
        return "DRAW"
    else:
        raise ValueError(f"Invalid Reward Signal {reward}")


def get_base_path(folder_path, num_simulations, elo, cfg=cfg):
    elo_bin = (elo // cfg.elo_bins) * cfg.elo_bins
    return f"{folder_path}/{cfg.env_type}/num_sims_{num_simulations}/elo_{elo_bin}/"


def get_game_id(base_path):
    with open(f"{base_path}/game_id.txt", "r") as f:
        game_id = int(f.read())
    return game_id


def get_episode_path(folder_path, num_simulations, elo, reward, cfg=cfg):
    base_path = get_base_path(folder_path, num_simulations, elo, cfg=cfg)
    try:
        game_id = get_game_id(base_path)
        game_id += 1
    except Exception:
        os.makedirs(base_path)
        game_id = 0
    subfolder = game_id // 200
    with open(f"{base_path}/game_id.txt", "w") as f:
        f.write(f"{game_id}")
    return base_path + f"{subfolder}/obs_game_{game_id}_r_{reward}.pt"


def save_game(
    episode_path,
    action_history,
    game_history,
    reward,
    model_elo=0,
    n_moves=0,
    cfg=cfg,
    model_id=0,
):
    if not os.path.exists(os.path.dirname(episode_path)):
        os.makedirs(os.path.dirname(episode_path))
    torch.save([game_history, action_history], episode_path)
    base_path = os.path.dirname(os.path.dirname(episode_path))
    game_id = get_game_id(base_path)
    new_row = pd.DataFrame(
        {
            "game": episode_path,
            "elo": model_elo,
            "move_count": n_moves,
            "model_id": model_id,
            "reward": reward,
        },
        index=[game_id],
    )
    cfg.episodes_df = pd.concat([cfg.episodes_df, new_row])

    if game_id % 500 == 0:
        cfg.episodes_df.to_csv(os.path.dirname(base_path) + "/games.csv")


def get_model(model_type=None, cfg=cfg):
    if not model_type:
        model_type = cfg.model_type
    if model_type == "mlp":
        return SimpleModel(cfg=cfg)
    elif model_type == "convnet":
        return SimpleConvnet(cfg=cfg)
    elif model_type == "ag0_resnet":
        return AlphaGoZeroResnet(cfg=cfg)


def save_model(model, model_elo, avg_moves, epoch, model_id, num_sims, cfg=cfg):
    model_save_path = f"{cfg.episode_save_path}/models/{cfg.env_type}_{cfg.model_type}_{epoch}_elo_{model_elo}.pth"

    try:
        torch.save(model.state_dict(), model_save_path)
    except Exception:
        os.makedirs(f"{cfg.episode_save_path}/models/")
        torch.save(model.state_dict(), model_save_path)
    new_model = pd.DataFrame(
        {
            "model": model_save_path,
            "elo": model_elo,
            "avg_moves": avg_moves,
            "model_id": model_id,
            "epoch": epoch,
            "num_sims": num_sims,
            "model_type": cfg.model_type,
            "episodes": epoch * cfg.episodes_per_epoch,
            "num_parameters": model.num_parameters(),
        },
        index=[len(cfg.models_df)],
    )
    try:
        cfg.models_df = pd.concat([cfg.models_df, new_model])
    except Exception:
        cfg.models_df = new_model


def get_elo_diff_from_outcomes(outcomes):
    n_draws = outcomes[1]
    white_wins = outcomes[0]
    black_wins = outcomes[2]
    total_games = n_draws + white_wins + black_wins
    white_score = 1 * white_wins + 0.5 * n_draws
    black_score = 1 * black_wins + 0.5 * n_draws
    score = max(white_score, black_score)
    # TODO: What to do here? Cannot take log of 0
    if white_score == total_games:
        return 400
    elif black_score == total_games:
        return -400
    elo_diff = round(-400 * math.log(1 / (score / total_games) - 1) / math.log(10), 0)
    # TODO: evaluate if this is needed
    # Clamp to a Maximum of 400
    elo_diff = min(elo_diff, 400)
    if white_score >= black_score:
        return elo_diff
    else:
        return -elo_diff
