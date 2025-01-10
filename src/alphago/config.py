### Hyperparameters
from pydantic_settings import BaseSettings
import pandas as pd

class Settings(BaseSettings):
    max_episode_length:int = 300
    max_n_episodes:int = 20000
    eval_games:int = 10
    episodes_per_epoch:int = 32
    elo_bins:int = 50
    
    episodes_replay_buffer_size:int = 256
    batch_samples_from_buffer:int = 8

    episode_save_path:str = "./data/"

    batch_size:int = 256
    learning_rate:float  = 1e-3
    min_learning_rate:float = 1e-5
    eps:float = 0.9
    discount_factor:float = 0.98
    env_type:str = "chess"
    
    draw_reward:float = 0.3
    
    model_type:str = "convnet" # ["mlp" , "convnet", "resnet"]

    action_size:dict = {"go_5":(26),"go":(82),"tictactoe":(9),"chess":(4672),"connect4":(7)}
    obs_size:dict = {"go_5":(5,5,17),"go":(9,9,17),"tictactoe":(3,3,2),"chess":(8,8,111),"connect4":(6, 7, 2)}

    agents:dict = {"go_5":["black_0","white_0"],"go":["black_0","white_0"],"tictactoe":["player_1","player_2"],"chess":['player_0', 'player_1'],"connect4":['player_0', 'player_1']}

    models_df:pd.DataFrame = pd.DataFrame(columns = ["model","elo","avg_moves","model_id","epoch","num_sims","model_type","episodes"])
    episodes_df:pd.DataFrame = pd.DataFrame(columns = ["game","actions","elo","move_count","model_id","reward"])

    # MCTS Hyperparams
    temperature: float = 1.0
    num_simulations:int = 0
    search_time:int = 0
    dirichlet_epsilon: float = 0.0
    dirichlet_noise: float = 0.0
    add_dirichlet_noise: bool = False
    puct_coefficient: float = 1.0
    turn_based_flip: bool = True

cfg = Settings()