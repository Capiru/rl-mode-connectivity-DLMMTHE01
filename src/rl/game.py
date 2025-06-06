import torch
import numpy as np

from rl.config import Settings

from rl.env import get_env
from rl.utils import (
    play_move,
    save_game,
    get_episode_path,
    find_winner,
    get_model,
    get_elo_diff_from_outcomes,
    save_model,
    save_eval_result,
)
from rl.eval import eval_model
from rl.data import GameHistoryDataset
from rl.train import train_model
from rl.schedulers import lr_scheduler, eps_scheduler
from visualizations.img import model_iterations_plot

from IPython.display import clear_output
import pandas as pd
import gc
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter

import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def play_game(
    env,
    models={"black_0": None, "white_0": None},
    episode_length=None,
    model_elo=0,
    save_game_flag=True,
    render=False,
    eval=False,
    eps=None,
    num_sims=None,
    model_id=0,
    player_to_learn=2,
    cfg=None,
):
    if not eps:
        eps = cfg.eps
    if not num_sims:
        num_sims = cfg.num_simulations
    if not episode_length:
        episode_length = cfg.max_episode_length

    ply_count = 0
    game_history = torch.zeros((*cfg.obs_size[cfg.env_type], episode_length))
    action_history = torch.zeros((cfg.action_size[cfg.env_type], episode_length))
    reward = 0
    done = False

    if not eval and cfg.random_moves_at_start:
        rand_moves = np.random.randint(0, cfg.random_moves_at_start + 1) * 2
        rand_moves_count = 0
        if rand_moves > 0:
            for agent in env.agent_iter():
                obs, reward, action, done = play_move(
                    env,
                    agent,
                    model=None,
                    eval=eval,
                    eps=eps,
                    num_sims=num_sims,
                    cfg=cfg,
                )
                rand_moves_count += 1
                if rand_moves_count >= rand_moves or done:
                    break
    if done:
        return 0, 10
    for agent in env.agent_iter():
        model = models[agent]
        obs, reward, action, done = play_move(
            env, agent, model, eval=eval, eps=eps, num_sims=num_sims, cfg=cfg
        )
        if isinstance(action, np.ndarray):
            action_history[:, ply_count] = torch.tensor(action)
        elif action:
            action_history[action, ply_count] = 1

        game_history[:, :, :, ply_count] = torch.tensor(obs["observation"].copy())
        ply_count += 1
        if reward != 0.0 or done:
            if agent == cfg.agents[cfg.env_type][1]:
                reward *= -1
            if cfg.env_type == "chess" and reward == 0.0:
                reward = cfg.draw_reward
            break
        if ply_count >= episode_length and reward == 0:
            break
        if render:
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.title(f"Move {ply_count} - action chosen {action}")
            plt.show()
    if reward != 0.0 and save_game_flag:
        episode_path = get_episode_path(
            cfg.episode_save_path, num_sims, model_elo, reward, cfg=cfg
        )
        save_game(
            episode_path,
            action_history[:, 0:ply_count],
            game_history[:, :, :, 0:ply_count],
            reward=reward,
            model_elo=model_elo,
            n_moves=ply_count,
            model_id=model_id,
            player_to_learn=player_to_learn,
            eval=eval,
            cfg=cfg,
        )

    if render:
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.title(f"{find_winner(reward, cfg = cfg)} Wins - Move {ply_count}")
        plt.show()
        print(action_history)
    return reward, ply_count


def one_eval_round(
    i,
    models,
    models_inverted,
    episode_length,
    wins,
    dnfs,
    losses,
    outcomes,
    ply_total,
    num_sims=None,
    cfg=None,
):
    if not num_sims:
        num_sims = cfg.num_simulations
    env = get_env(cfg.env_type)
    if i % 2 == 0:
        outcome, ply_game = play_game(
            env,
            models,
            save_game_flag=True,
            episode_length=episode_length,
            eval=True,
            eps=0,
            num_sims=num_sims,
            cfg=cfg,
        )
        outcomes.append(outcome)
    else:
        outcome, ply_game = play_game(
            env,
            models_inverted,
            save_game_flag=True,
            episode_length=episode_length,
            eval=True,
            eps=0,
            num_sims=num_sims,
            cfg=cfg,
        )
        outcomes.append(-outcome)
    if outcomes[-1] == 0 or outcomes[-1] == cfg.draw_reward:
        dnfs += 1
    elif outcomes[-1] == 1:
        wins += 1
    elif outcomes[-1] == -1:
        losses += 1
    ply_total += ply_game
    return outcomes, wins, dnfs, losses, ply_total


def tournament_match(
    models=None,
    n_games=100,
    episode_length=None,
    opponent_elo=0,
    num_sims=None,
    cfg=None,
    use_tqdm=True,
):
    if not num_sims:
        num_sims = cfg.num_simulations
    if not models:
        models = {x: None for x in cfg.agents[cfg.env_type]}
    if not episode_length:
        episode_length = cfg.max_episode_length * 2

    outcomes = []
    model_keys = [x for x in models.keys()]
    models_inverted = {
        model_keys[0]: models[model_keys[1]],
        model_keys[1]: models[model_keys[0]],
    }
    wins = 0
    dnfs = 0
    losses = 0
    ply_total = 0
    if use_tqdm:
        with tqdm(total=n_games) as pbar:
            for i in range(n_games):
                outcomes, wins, dnfs, losses, ply_total = one_eval_round(
                    i,
                    models,
                    models_inverted,
                    episode_length,
                    wins,
                    dnfs,
                    losses,
                    outcomes,
                    ply_total,
                    num_sims=num_sims,
                    cfg=cfg,
                )
                pbar.set_description(
                    f"Tournament Eval - Elo {opponent_elo} - Game {i+1} - Wins {wins} - Dnfs {dnfs} - Losses {losses} - Avg. Moves {ply_total/(i+1)}"
                )
                pbar.update(1)
    else:
        for i in range(n_games):
            outcomes, wins, dnfs, losses, ply_total = one_eval_round(
                i,
                models,
                models_inverted,
                episode_length,
                wins,
                dnfs,
                losses,
                outcomes,
                ply_total,
                num_sims=num_sims,
                cfg=cfg,
            )
    return [wins, dnfs, losses], ply_total / (i + 1)


def eval_losses(model, eval_episodes: int, iteration=0, cfg=None):
    dataset = GameHistoryDataset(cfg=cfg, eval=True, only_last_n=eval_episodes)
    _, (policy_loss, value_loss) = eval_model(
        model, dataset=dataset, epochs=1, iteration=iteration, cfg=cfg
    )
    print(f"Eval - Policy Loss {policy_loss:.4f} - Value Loss {value_loss:.4f}")
    return policy_loss, value_loss


def eval_tournament(
    models_df,
    model,
    iteration,
    model_id=0,
    n_games=100,
    num_sims=None,
    static_eval=False,
    cfg=None,
):
    if not num_sims:
        num_sims = cfg.num_simulations
    # Test versus Random Agent
    total_eval_games = 0
    outcomes, avg_moves = tournament_match(
        {cfg.agents[cfg.env_type][0]: model, cfg.agents[cfg.env_type][1]: None},
        n_games=n_games,
        num_sims=num_sims,
        cfg=cfg,
    )
    total_eval_games += n_games
    elo_diff = get_elo_diff_from_outcomes(outcomes)

    actual_elo = elo_diff

    # Test Versus Randomly Initialized Net with Same Parameters
    outcomes_nn, avg_moves_nn = tournament_match(
        {
            cfg.agents[cfg.env_type][0]: model,
            cfg.agents[cfg.env_type][1]: get_model(cfg=cfg),
        },
        n_games=n_games,
        num_sims=num_sims,
        cfg=cfg,
    )
    total_eval_games += n_games
    elo_diff_nn = get_elo_diff_from_outcomes(outcomes_nn)
    print(
        f"Elo diff from Random {elo_diff} Avg. Moves {avg_moves} - From Randomly Initialized Net {elo_diff_nn} Avg. Moves {avg_moves_nn}"
    )
    if elo_diff >= 400.0 and cfg.num_simulations > 0.0:
        actual_elo += elo_diff_nn
    sorted_models = models_df.loc[models_df["elo"] >= actual_elo].sort_values(
        by=["elo", "epoch"], ascending=[True, False]
    )
    sorted_models = sorted_models.loc[sorted_models["model_id"] == model_id]
    # print(sorted_models)
    save_eval_result(
        iteration=iteration,
        elo_random_play=elo_diff,
        move_count_random_play=avg_moves,
        elo_random_init=elo_diff_nn,
        move_count_random_init=avg_moves_nn,
        model_id=model_id,
        elo_opp=0,
        move_count=0,
        elo_diff=0,
        winrate=outcomes[1] / sum(outcomes),
        cfg=cfg,
    )
    if actual_elo >= 35 and not static_eval:
        for i in range(50):
            if len(sorted_models) < 1:
                break
            eval_model = get_model(cfg=cfg)
            eval_model.load_state_dict(
                torch.load(sorted_models.iloc[0]["model"], weights_only=True)
            )
            outcomes_new, avg_moves_new = tournament_match(
                {
                    cfg.agents[cfg.env_type][0]: model,
                    cfg.agents[cfg.env_type][1]: eval_model,
                },
                n_games=n_games,
                opponent_elo=sorted_models.iloc[0]["elo"],
                num_sims=num_sims,
                cfg=cfg,
            )
            total_eval_games += n_games
            elo_diff_new = get_elo_diff_from_outcomes(outcomes_new)

            actual_elo = sorted_models.iloc[0]["elo"] + elo_diff_new
            save_eval_result(
                iteration=iteration,
                elo_random_play=elo_diff,
                move_count_random_play=avg_moves,
                elo_random_init=elo_diff_nn,
                move_count_random_init=avg_moves_nn,
                model_id=model_id,
                elo_opp=sorted_models.iloc[0]["elo"],
                move_count=avg_moves_new,
                elo_diff=elo_diff_new,
                winrate=outcomes_new[1] / sum(outcomes_new),
                cfg=cfg,
            )
            print(f"Estimated agent Elo {actual_elo}")
            if elo_diff_new <= 0 or len(sorted_models) <= 1:
                break
            else:
                sorted_models = sorted_models.loc[sorted_models["elo"] >= actual_elo]
                del eval_model
                gc.collect()
    policy_loss, value_loss = eval_losses(
        model=model, eval_episodes=total_eval_games, iteration=iteration, cfg=cfg
    )
    cfg.eval_df.to_csv(
        cfg.episode_save_path
        + f"/{cfg.env_type}/num_sims_{cfg.num_simulations}"
        + "/eval.csv"
    )
    return actual_elo, outcomes, avg_moves, (policy_loss, value_loss)


def generate_one_game(
    i,
    models,
    models_inverted,
    wins,
    dnfs,
    losses,
    model_elo,
    ply_total,
    eps=None,
    num_sims=None,
    model_id=0,
    cfg=None,
):
    if not num_sims:
        num_sims = cfg.num_simulations
    env = get_env(cfg.env_type)
    if i % 2 == 0:
        outcome, ply_game = play_game(
            env,
            models,
            save_game_flag=True,
            model_elo=model_elo,
            eps=eps,
            num_sims=num_sims,
            model_id=model_id,
            cfg=cfg,
        )
    else:
        outcome, ply_game = play_game(
            env,
            models_inverted,
            save_game_flag=True,
            model_elo=model_elo,
            eps=eps,
            num_sims=num_sims,
            model_id=model_id,
            cfg=cfg,
        )
        outcome = -outcome
    if outcome == 0:
        dnfs += 1
    elif outcome == 1:
        wins += 1
    elif outcome == -1:
        losses += 1
    ply_total += ply_game
    return wins, dnfs, losses, ply_total


def generate_games(
    models=None,
    num_games=1000,
    model_elo=0,
    eps=None,
    num_sims=None,
    model_id=0,
    cfg=None,
    use_tqdm=True,
):
    if not num_sims:
        num_sims = cfg.num_simulations
    if not models:
        models = {x: None for x in cfg.agents[cfg.env_type]}
    if not eps:
        eps = cfg.eps

    model_keys = [x for x in models.keys()]
    models_inverted = {
        model_keys[0]: models[model_keys[1]],
        model_keys[1]: models[model_keys[0]],
    }
    wins = 0
    dnfs = 0
    losses = 0
    ply_total = 0
    if use_tqdm:
        with tqdm(total=num_games) as pbar:
            for i in range(num_games):
                wins, dnfs, losses, ply_total = generate_one_game(
                    i,
                    models,
                    models_inverted,
                    wins,
                    dnfs,
                    losses,
                    model_elo,
                    ply_total,
                    eps=eps,
                    num_sims=num_sims,
                    model_id=model_id,
                    cfg=cfg,
                )

                pbar.set_description(
                    f"Model{model_id}- Self-Play - Game {i+1} - Wins {wins} - Dnfs {dnfs} - Losses {losses} - Avg. Moves {ply_total/(i+1)}"
                )
                pbar.update(1)
    else:
        for i in range(num_games):
            wins, dnfs, losses, ply_total = generate_one_game(
                i,
                models,
                models_inverted,
                wins,
                dnfs,
                losses,
                model_elo,
                ply_total,
                eps=eps,
                num_sims=num_sims,
                model_id=model_id,
                cfg=cfg,
            )

    cfg.episodes_df.to_csv(
        f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}"
        + "/games.csv"
    )


def self_play(cfg, max_patience=5):
    model = get_model(cfg=cfg)
    print(f"Model with {model.num_parameters()} parameters")

    # Resume Training
    if cfg.resume_from:
        model.load_state_dict(torch.load(cfg.resume_from))
        cfg.models_df = pd.read_csv(
            f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}/models.csv"
        )
        cfg.episodes_df = pd.read_csv(
            f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}/games.csv"
        )
        start_epoch = cfg.models_df["epoch"].max() + 1
        eps, learning_rate = (
            eps_scheduler(start_epoch, cfg),
            lr_scheduler(start_epoch, cfg),
        )
        print(
            f"Loaded model from {cfg.resume_from}: start epoch {start_epoch} lr {learning_rate} eps {eps}"
        )
    else:
        start_epoch = 0
        learning_rate = cfg.learning_rate
        eps = cfg.eps
    eval_games = cfg.eval_games

    try:
        # model.load_state_dict(torch.load("best_model.pth"))
        # model = torch.load("best_model.pth")
        # print("Previous best model loaded")
        model_elo, outcomes, avg_moves, eval_losses = eval_tournament(
            cfg.models_df, model, iteration=0, n_games=eval_games, cfg=cfg
        )
        print(
            "Epoch: 0 Total episodes generated: 0",
            "Winrate:",
            outcomes[0] / sum([outcomes[0], outcomes[2]]),
            "Draws: ",
            outcomes[1],
            "Elo diff:",
            model_elo,
            "lr:",
            cfg.learning_rate,
            "eps:",
            cfg.eps,
            "avg_moves:",
            avg_moves,
        )
        save_model(
            model=model,
            model_elo=model_elo,
            avg_moves=avg_moves,
            epoch=0,
            model_id=0,
            num_sims=cfg.num_simulations,
            policy_loss=0,
            value_loss=0,
            eval_policy_loss=eval_losses[0],
            eval_value_loss=eval_losses[1],
            cfg=cfg,
        )

    except Exception as e:
        print(e)
        pass

    if cfg.use_only_best_model:
        best_model = copy.deepcopy(model)
        models = {
            cfg.agents[cfg.env_type][0]: best_model,
            cfg.agents[cfg.env_type][1]: best_model,
        }
    else:
        models = {
            cfg.agents[cfg.env_type][0]: model,
            cfg.agents[cfg.env_type][1]: model,
        }

    best_elo = 0
    model_elo = 0
    for i in range(start_epoch, int(cfg.max_n_episodes // cfg.episodes_per_epoch)):
        eps, learning_rate = eps_scheduler(i, cfg), lr_scheduler(i, cfg)
        if cfg.use_only_best_model:
            best_model.eval()
        else:
            model.eval()
        generate_games(
            models,
            num_games=cfg.episodes_per_epoch,
            model_elo=model_elo,
            eps=eps,
            num_sims=cfg.num_simulations,
            cfg=cfg,
        )

        dataset = GameHistoryDataset(cfg=cfg)

        model, losses = train_model(
            model, dataset, iteration=i, learning_rate=learning_rate, cfg=cfg
        )
        policy_loss, value_loss = losses

        if i % cfg.eval_every_n_epochs == 0 and i > 0:
            model.eval()
            model_elo, outcomes, avg_moves, eval_losses = eval_tournament(
                cfg.models_df, model, iteration=i, n_games=eval_games, cfg=cfg
            )
            print(
                "Epoch:",
                i,
                "Total episodes generated: ",
                (i + 1) * cfg.episodes_per_epoch,
                "Winrate:",
                outcomes[0] / sum(outcomes),
                "Elo diff:",
                model_elo,
                "lr:",
                learning_rate,
                "eps:",
                eps,
                "avg_moves:",
                avg_moves,
            )

            save_model(
                model=model,
                model_elo=model_elo,
                avg_moves=avg_moves,
                epoch=i,
                model_id=0,
                num_sims=cfg.num_simulations,
                policy_loss=policy_loss,
                value_loss=value_loss,
                eval_policy_loss=eval_losses[0],
                eval_value_loss=eval_losses[1],
                cfg=cfg,
            )
            cfg.models_df.to_csv(
                f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}"
                + "/models.csv"
            )
            if model_elo >= best_elo:
                best_elo = model_elo
                torch.save(model.state_dict(), "best_model.pth")
                if cfg.use_only_best_model:
                    print("Best model updated for game generation!")
                    best_model = copy.deepcopy(model)
                    models = {
                        cfg.agents[cfg.env_type][0]: best_model,
                        cfg.agents[cfg.env_type][1]: best_model,
                    }
                # print("Saved best model so far")
                # patience = 0
            # elif model_elo <= best_elo - 400:
            #     patience += 1
            #     print(f"Model underperformed for {max_patience} epochs, loading best model")
            #     if patience >= max_patience:
            #         model = get_model(cfg = cfg)
            #         try:
            #             model.load_state_dict(torch.load("best_model.pth", weights_only=True))
            #         except Exception as e:
            #             pass
            #         models = {cfg.agents[cfg.env_type][0]:model,cfg.agents[cfg.env_type][1]:model}
            gc.collect()
            model_iterations_plot(0, cfg=cfg)


if __name__ == "__main__":
    # asyncio.run()
    start_timer = perf_counter()
    self_play(cfg=Settings())
    print("Time taken:", perf_counter() - start_timer)
