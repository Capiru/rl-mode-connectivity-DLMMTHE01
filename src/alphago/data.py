### Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from time import perf_counter
import pandas as pd
import numpy as np
from alphago.config import cfg

import logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

class GameHistoryDataset(Dataset):
    def __init__(self, cfg = cfg,min_elo = 0,model_id:int = 0,smoke_test = False, drop_zero_rewards = True):
        self.min_elo = min_elo
        self.smoke_test = smoke_test
        self.folder_path = f"{cfg.episode_save_path}/{cfg.env_type}/num_sims_{cfg.num_simulations}/"
        self.df = cfg.episodes_df.copy()
        
        if drop_zero_rewards:
            self.df = self.df.loc[self.df["reward"]!=0]

        self.df = self.df.loc[self.df["elo"]>=min_elo]
        if model_id:
            self.df = self.df.loc[self.df["model_id"]==model_id]
        df_sorted = self.df.sort_index()
        if cfg.episodes_replay_buffer_size:
            self.df = df_sorted.tail(cfg.episodes_replay_buffer_size)
        self.game_history_list = self.df["game"]
        
        self.df.reset_index(drop=True, inplace=True)
        self.df["episode"] = self.df.index
        
        if "go" in cfg.env_type:
            self.df["move_count"] = self.df["move_count"]-2
        
        self.episode_replay_df = self.df.loc[self.df.index.repeat(self.df["move_count"])].copy()
        self.episode_replay_df["timestep"] = self.episode_replay_df.groupby(level=0).cumcount()
        self.episode_replay_df.reset_index(drop=True, inplace=True)  
        

    def __len__(self):
        if self.smoke_test:
            return 100
        return len(self.episode_replay_df)
    
    def __getitem__(self, idx):
        # Load tensors from the specified files
        row = self.episode_replay_df.iloc[idx]
        
        try:
            obs,action = torch.load(str(self.game_history_list.iloc[row["episode"]]))  # Observation Tensor (9,9,17)
        except Exception as e:
            print(row,self.game_history_list.iloc[row["episode"]])
        reward = row["reward"] # Value (1,)
        episode_len = obs.shape[-1]
        time_step = int(row["timestep"])

        reward *= cfg.discount_factor**(episode_len-time_step)
        did_player_win = 1 if time_step % 2 == 0 else -1

        return obs[:,:,:,time_step], action[:,time_step], torch.tensor(reward*did_player_win,dtype=torch.float32)
