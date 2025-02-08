from rl.data import GameHistoryDataset
import pandas as pd


def test_dataset(cfg):
    cfg.episodes_df = pd.read_csv("tests/data/games.csv")
    dataset = GameHistoryDataset(cfg=cfg, shuffle=False)
    ## Game 0 - Player 0 wins
    assert len(dataset) == 12
    _, _, reward = dataset.__getitem__(0)

    assert round(reward.detach().item(), 4) == 0.9224
    assert dataset._did_player_0_win(0) == 1

    _, _, reward_4 = dataset.__getitem__(4)
    assert round(reward_4.detach().item(), 4) == 1.0000
    assert dataset._did_player_0_win(4) == 1

    _, _, reward_3 = dataset.__getitem__(3)

    assert round(reward_3.detach().item(), 4) == -1.0000 * 0.98
    assert dataset._did_player_0_win(3) == -1

    ## Game 1 - Player 1 Wins
    _, _, reward_player_0 = dataset.__getitem__(5)
    assert reward_player_0.detach().item() < 0.0
    assert dataset._did_player_0_win(0) == 1

    _, _, reward_player_0_end = dataset.__getitem__(11)
    assert dataset._did_player_0_win(6) == 1
    assert round(reward_player_0_end.detach().item(), 4) == -1.0000

    _, _, reward_player_1_end = dataset.__getitem__(10)
    assert dataset._did_player_0_win(5) == -1
    assert round(reward_player_1_end.detach().item(), 4) == 0.9800


def test_shuffle(cfg):
    cfg.episodes_df = pd.read_csv("tests/data/games.csv")
    shuffled_dataset = GameHistoryDataset(cfg, shuffle=True)
    dataset = GameHistoryDataset(cfg, shuffle=False)
    assert not shuffled_dataset.episode_replay_df["episode"].equals(
        dataset.episode_replay_df["episode"]
    )


def test_model_id(cfg):
    cfg.episodes_df = pd.read_csv("tests/data/games.csv")
    dataset = GameHistoryDataset(cfg=cfg, shuffle=False, model_id=0)
    assert len(dataset) == 12
    dataset_model_id1 = GameHistoryDataset(cfg=cfg, shuffle=False, model_id=1)
    assert len(dataset_model_id1) == 0
