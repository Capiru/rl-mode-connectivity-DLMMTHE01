from rl.config import Settings
from experiments.lmc_base_experiment import lmc_experiment

import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

if __name__ == "__main__":
    cfg_lmc = Settings(
        max_n_episodes=150000,
        env_type="go_5",
        model_type="mlp",
        model_size="large",
        temperature=1.0,
        num_simulations=0,
        puct_coefficient=1.0,
        episodes_per_epoch=1000,
        episode_save_path="./experiments/lmc_go5_long_mlp/",
        episodes_replay_buffer_size=2000,
        eval_games=100,
    )
    lmc_experiment(cfg=cfg_lmc)
