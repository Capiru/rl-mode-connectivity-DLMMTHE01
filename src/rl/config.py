### Hyperparameters
from pydantic_settings import BaseSettings
import pandas as pd
import torch


class Settings(BaseSettings):
    max_episode_length: int = 170
    max_n_episodes: int = 100000  # 500000
    eval_games: int = 100
    episodes_per_epoch: int = 5000
    elo_bins: int = 50
    eval_every_n_epochs: int = 2
    random_moves_at_start: int = 8

    episodes_replay_buffer_size: int = 50000
    sampling_ratio: float = 1.0

    episode_save_path: str = "./calibration/go5_mlp_mcts_20/"

    batch_size: int = 1024
    learning_rate: float = 1e-2
    min_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2

    # Schedulers
    lr_schedule_max_epochs: int = 500000
    eps_schedule_max_epochs: int = 50000
    buffer_size_schedule_max_epochs: int = 15000

    # Sets the env type to be trained
    env_type: str = "go_5"  # ["tictactoe","connect4","chess","go","go_5"]

    draw_reward: float = 0.3

    model_type: str = "ag0_resnet"  # ["mlp" , "convnet", "resnet", "ag0_resnet"]
    model_size: str = "large"  # ["small" , "medium", "large", "xlarge"]

    action_size: dict = {
        "go_5": (26),
        "go": (82),
        "tictactoe": (9),
        "chess": (4672),
        "connect4": (7),
    }
    obs_size: dict = {
        "go_5": (5, 5, 17),
        "go": (9, 9, 17),
        "tictactoe": (3, 3, 2),
        "chess": (8, 8, 111),
        "connect4": (6, 7, 2),
    }

    agents: dict = {
        "go_5": ["black_0", "white_0"],
        "go": ["black_0", "white_0"],
        "tictactoe": ["player_1", "player_2"],
        "chess": ["player_0", "player_1"],
        "connect4": ["player_0", "player_1"],
    }

    models_df: pd.DataFrame = pd.DataFrame(
        columns=[
            "model",
            "elo",
            "avg_moves",
            "model_id",
            "epoch",
            "num_sims",
            "model_type",
            "episodes",
            "num_parameters",
            "policy_loss",
            "value_loss",
            "eval_policy_loss",
            "eval_value_loss",
        ]
    )
    episodes_df: pd.DataFrame = pd.DataFrame(
        columns=[
            "game",
            "actions",
            "elo",
            "move_count",
            "model_id",
            "reward",
            "iteration",
            "player_to_learn",
            "eval",
        ]
    )

    eval_df: pd.DataFrame = pd.DataFrame(
        columns=[
            "iteration",
            "elo_random_play",
            "move_count_random_play",
            "elo_random_init",
            "move_count_random_init",
            "model_id",
            "elo_opp",
            "move_count",
            "elo_diff",
            "winrate",
        ]
    )

    # RL Hyperparameters
    eps: float = 0.25
    discount_factor: float = 0.98  # Connect 4 - Go 5x5 0.93

    # Game Generation
    use_models_to_generate_games: bool = False
    use_only_best_model: bool = True

    # MCTS Hyperparams
    temperature: float = 1.0
    num_simulations: int = 20
    search_time: int = 0
    dirichlet_alpha: float = 0.1
    puct_coefficient: float = 1.0
    turn_based_flip: bool = True

    # Resume Training
    resume_from: str = ""

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
