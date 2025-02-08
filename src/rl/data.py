### Dataset
import torch
from torch.utils.data import Dataset
from rl.schedulers import buffer_size_scheduler
import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class GameHistoryDataset(Dataset):
    def __init__(
        self,
        cfg,
        min_elo=0,
        model_id: int = 0,
        smoke_test=False,
        drop_zero_rewards=True,
        shuffle=True,
        eval=False,
        only_last_n: int = 0,
    ):
        self.cfg = cfg
        self.min_elo = min_elo
        self.smoke_test = smoke_test
        self.folder_path = f"{self.cfg.episode_save_path}/{self.cfg.env_type}/num_sims_{self.cfg.num_simulations}/"
        self.df = self.cfg.episodes_df.copy()

        if drop_zero_rewards:
            self.df = self.df.loc[self.df["reward"] != 0]

        self.df = self.df.loc[self.df["elo"] >= min_elo]
        if model_id:
            self.df = self.df.loc[self.df["model_id"] == model_id]
        self.df = self.df.loc[self.df["eval"] == eval]
        df_sorted = self.df.sort_index()
        if self.cfg.episodes_replay_buffer_size and not eval:
            self.df = df_sorted.tail(
                buffer_size_scheduler(episodes=len(self.df), cfg=self.cfg)
            )
        elif eval and only_last_n > 0:
            self.df = df_sorted.tail(only_last_n)
        self.game_history_list = self.df["game"]

        self.df.reset_index(drop=True, inplace=True)
        self.df["episode"] = self.df.index

        if "go" in self.cfg.env_type:
            self.df["move_count"] = self.df["move_count"] - 2
            self.df = self.df.loc[self.df["move_count"] > 0]

        self.episode_replay_df = self.df.loc[
            self.df.index.repeat(self.df["move_count"])
        ].copy()
        self.episode_replay_df["timestep"] = self.episode_replay_df.groupby(
            level=0
        ).cumcount()
        self.episode_replay_df.reset_index(drop=True, inplace=True)
        # If episodes are being saved for only one or both policies
        self.episode_replay_df = self.episode_replay_df[
            (
                (self.episode_replay_df["player_to_learn"] == 0)
                & (self.episode_replay_df["timestep"] % 2 == 0)
            )
            | (
                (self.episode_replay_df["player_to_learn"] == 1)
                & (self.episode_replay_df["timestep"] % 2 == 1)
            )
            | (self.episode_replay_df["player_to_learn"] == 2)
        ]
        if shuffle:
            self.episode_replay_df = self.episode_replay_df.sample(
                frac=self.cfg.sampling_ratio, weights="iteration"
            ).reset_index(drop=True)

    def __len__(self):
        if self.smoke_test:
            return 100
        return len(self.episode_replay_df)

    def _get_discounted_reward(self, reward, timestep, episode_len):
        return reward * self.cfg.discount_factor ** (episode_len - timestep - 1)

    def _did_player_0_win(self, timestep):
        return 1 if timestep % 2 == 0 else -1

    def _get_return_reward(self, reward, timestep, episode_len):
        return self._get_discounted_reward(
            reward, timestep, episode_len
        ) * self._did_player_0_win(timestep)

    def __getitem__(self, idx):
        # Load tensors from the specified files
        row = self.episode_replay_df.iloc[idx]

        try:
            obs, action = torch.load(
                str(self.game_history_list.iloc[row["episode"]])
            )  # Observation Tensor (9,9,17)
        except Exception:
            print(row, self.game_history_list.iloc[row["episode"]])
        reward = row["reward"]  # Value (1,)
        episode_len = int(row["move_count"])
        timestep = int(row["timestep"])

        return (
            obs[:, :, :, timestep],
            action[:, timestep],
            torch.tensor(
                self._get_return_reward(reward, timestep, episode_len),
                dtype=torch.float32,
            ),
        )
