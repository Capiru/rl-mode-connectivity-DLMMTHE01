from rl.utils import get_model, save_model
from rl.train import train_model
from rl.game import eval_tournament, generate_games
from rl.data import GameHistoryDataset
from mode_connectivity.interpolate import interpolate_models
from rl.schedulers import lr_scheduler, eps_scheduler
from visualizations.img import model_iterations_plot, interpolation_plot
import copy

import pandas as pd
import numpy as np
import torch


import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

"""
Experiment Workflow:
1. Define in which fractional epochs the model should be split
2. Initialize vars for each epoch
3. If current epoch is in the 'split_epoch_list', duplicate model and continue training.

"""


def lmc_experiment(split_epoch_list=[0, 0.5], cfg=None):
    # 1. Define which fractional epochs to split

    # We start with the first on the list
    split_epoch = split_epoch_list.pop(0)
    epoch_split = {}

    for n_sims in [0]:
        # Vars Initialization
        model_id = 0
        model_ids = [0]
        model = [get_model(cfg=cfg)]
        learning_rate = cfg.learning_rate
        eps = cfg.eps

        model_elo, outcomes, avg_moves, eval_losses = eval_tournament(
            cfg.models_df,
            model[0],
            iteration=0,
            n_games=cfg.eval_games,
            num_sims=n_sims,
            static_eval=True,
            cfg=cfg,
        )
        print(
            "Epoch: 0 Total episodes generated: 0",
            "Winrate:",
            outcomes[0] / sum(outcomes),
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
            model=model[0],
            model_elo=model_elo,
            avg_moves=avg_moves,
            epoch=0,
            model_id=model_id,
            num_sims=n_sims,
            policy_loss=0,
            value_loss=0,
            eval_policy_loss=eval_losses[0],
            eval_value_loss=eval_losses[1],
            cfg=cfg,
        )
        models = [
            {
                cfg.agents[cfg.env_type][0]: model[0],
                cfg.agents[cfg.env_type][1]: model[0],
            }
        ]
        model_elo = [0]
        model_losses = [eval_losses]
        split = False
        for i in range(int(cfg.max_n_episodes // cfg.episodes_per_epoch)):
            # Each epoch we update epsilon and the learning rate based on the given scheduler
            eps, learning_rate = eps_scheduler(i, cfg), lr_scheduler(i, cfg)
            cfg.models_df.to_csv(
                f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}"
                + "/models.csv"
            )
            if (
                i / int(cfg.max_n_episodes // cfg.episodes_per_epoch) >= split_epoch
                and not split
            ):
                # 3. Split models

                model_ids.append(len(model_ids))
                model_elo.append(model_elo[0])
                model_losses.append(eval_losses)
                model.append(copy.deepcopy(model[0]))
                models.append(
                    {
                        cfg.agents[cfg.env_type][0]: model[-1],
                        cfg.agents[cfg.env_type][1]: model[-1],
                    }
                )
                epoch_split[len(model_ids) - 1] = split_epoch
                print(
                    f"Models split at {i} Epoch (ratio {i/ int(cfg.max_n_episodes//cfg.episodes_per_epoch)}) !",
                    model_ids,
                )
                try:
                    split_epoch = split_epoch_list.pop(0)
                except IndexError:
                    split = True
                # We do a sanity check if the models were initialized correctly
                params_model_1 = [param.clone() for param in model[0].parameters()]
                params_model_2 = [param.clone() for param in model[-1].parameters()]
                for param_1, param_2 in zip(params_model_1, params_model_2):
                    assert torch.equal(param_1, param_2)
                print("Models are split correctly, with same weights!")

            # 4. Start epoch logic for each model_id being trained
            for m_id in model_ids:
                print(f"Epoch {i} - Model ID {m_id}")
                model[m_id].eval()
                # 4.a Generate a batch of games
                generate_games(
                    models[m_id],
                    num_games=cfg.episodes_per_epoch,
                    model_elo=model_elo[m_id],
                    eps=eps,
                    num_sims=n_sims,
                    model_id=m_id,
                    cfg=cfg,
                )

                # 4.b Initialize Dataset, only loading data from a given model_id
                dataset = GameHistoryDataset(cfg=cfg, model_id=m_id)

                # 4.c Train the model with the dataset
                model[m_id], losses = train_model(
                    model[m_id],
                    dataset,
                    iteration=i,
                    learning_rate=learning_rate,
                    cfg=cfg,
                )
                policy_loss, value_loss = losses

                model[m_id].eval()
                model_elo[m_id], outcomes, avg_moves, model_losses[m_id] = (
                    eval_tournament(
                        cfg.models_df,
                        model[m_id],
                        iteration=i,
                        model_id=m_id,
                        n_games=cfg.eval_games,
                        num_sims=n_sims,
                        static_eval=True,
                        cfg=cfg,
                    )
                )

                print(
                    "Epoch:",
                    i,
                    "Total episodes generated: ",
                    (i + 1) * cfg.episodes_per_epoch,
                    "Winrate:",
                    outcomes[0] / sum(outcomes),
                    "Elo diff:",
                    model_elo[m_id],
                    "lr:",
                    learning_rate,
                    "eps:",
                    eps,
                    "avg_moves:",
                    avg_moves,
                )
                save_model(
                    model=model[m_id],
                    model_elo=model_elo[m_id],
                    avg_moves=avg_moves,
                    epoch=i,
                    model_id=m_id,
                    num_sims=n_sims,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    eval_policy_loss=model_losses[m_id][0],
                    eval_value_loss=model_losses[m_id][1],
                    cfg=cfg,
                )

                if m_id > 0:
                    n_points = 5
                    alpha_space = np.linspace(
                        0.1, 0.9, n_points
                    )  # We start the linspace not at 0 and not until 1, since we already have the eval for those.
                    losses_model_1 = model_losses[m_id]
                    losses_model_0 = model_losses[0]
                    plot_df = pd.DataFrame(
                        {
                            "alpha": [0, 1],
                            "elo": [model_elo[m_id], model_elo[0]],
                            "policy_loss": [losses_model_1[0], losses_model_0[0]],
                            "value_loss": [losses_model_1[1], losses_model_0[1]],
                        },
                        columns=["alpha", "elo", "policy_loss", "value_loss"],
                    )
                    for j in range(n_points):
                        interpolated_model = interpolate_models(
                            model[0], model[m_id], alpha_space[j]
                        )
                        int_model_elo, outcomes, avg_moves, eval_losses = (
                            eval_tournament(
                                cfg.models_df,
                                interpolated_model,
                                iteration=i,
                                model_id=int(f"{m_id}{j}"),
                                n_games=cfg.eval_games,
                                num_sims=n_sims,
                                static_eval=True,
                                cfg=cfg,
                            )
                        )
                        new_eval = pd.DataFrame(
                            {
                                "alpha": [alpha_space[j]],
                                "elo": [int_model_elo],
                                "policy_loss": [eval_losses[0]],
                                "value_loss": [eval_losses[1]],
                            }
                        )
                        plot_df = pd.concat([plot_df, new_eval])

                    plot_df = plot_df.sort_values(by="alpha")
                    print(plot_df)
                    plot_df.to_csv(
                        f"{cfg.episode_save_path}"
                        + f"/lmc_{str(i)}_epoch_split_{epoch_split[m_id]}.csv"
                    )
                    # Unnormalized Plots
                    interpolation_plot(plot_df, "elo", i, epoch_split[m_id], cfg)
                    interpolation_plot(
                        plot_df, "policy_loss", i, epoch_split[m_id], cfg
                    )
                    interpolation_plot(plot_df, "value_loss", i, epoch_split[m_id], cfg)

                    # Normalized Plots
                    plot_df["elo"] = plot_df["elo"] / np.max(plot_df["elo"])
                    plot_df["policy_loss"] = np.abs(plot_df["policy_loss"])
                    plot_df["policy_loss"] = plot_df["policy_loss"] / np.max(
                        plot_df["policy_loss"]
                    )
                    plot_df["value_loss"] = plot_df["value_loss"] / np.max(
                        plot_df["value_loss"]
                    )

                    interpolation_plot(plot_df, "elo", i, epoch_split[m_id], cfg)
                    interpolation_plot(
                        plot_df, "policy_loss", i, epoch_split[m_id], cfg
                    )
                    interpolation_plot(plot_df, "value_loss", i, epoch_split[m_id], cfg)

                model_iterations_plot(m_id, cfg=cfg)
