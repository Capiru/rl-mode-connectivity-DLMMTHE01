### Hyperparameters
from pydantic_settings import BaseSettings
import pandas as pd


class Settings(BaseSettings):
    max_episode_length: int = 170
    max_n_episodes: int = 20000
    eval_games: int = 20
    episodes_per_epoch: int = 32
    elo_bins: int = 50
    eval_every_n_epochs: int = 2

    episodes_replay_buffer_size: int = 1024
    batch_samples_from_buffer: int = 8
    sampling_ratio: float = 0.7

    episode_save_path: str = "./data/"

    batch_size: int = 1024
    learning_rate: float = 1e-2
    min_learning_rate: float = 1e-4

    env_type: str = "go_5"

    draw_reward: float = 0.3

    model_type: str = "ag0_resnet"  # ["mlp" , "convnet", "resnet"]

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
        ]
    )
    episodes_df: pd.DataFrame = pd.DataFrame(
        columns=["game", "actions", "elo", "move_count", "model_id", "reward"]
    )

    # RL Hyperparameters
    eps: float = 0.9
    discount_factor: float = 0.98

    # MCTS Hyperparams
    temperature: float = 1.0
    num_simulations: int = 150
    search_time: int = 0
    dirichlet_alpha: float = 0.3
    puct_coefficient: float = 1.0
    turn_based_flip: bool = True

    # Resume Training
    resume_from: str = ""  # "data/models/go_5_convnet_170_elo_147.0.pth"


cfg = Settings()
