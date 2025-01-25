from pettingzoo.classic import go_v5, chess_v6, tictactoe_v3, connect_four_v3


def get_env(env_type):
    if env_type == "go_5":
        env = go_v5.raw_env(board_size=5, komi=0.5, render_mode="rgb_array")
        env.reset(seed=0)
    elif env_type == "go":
        env = go_v5.raw_env(board_size=9, komi=6.5, render_mode="rgb_array")
        env.reset(seed=0)
    elif env_type == "chess":
        env = chess_v6.raw_env(render_mode="rgb_array")
        env.reset(seed=0)
    elif env_type == "tictactoe":
        env = tictactoe_v3.raw_env(render_mode="rgb_array")
        env.reset(seed=0)
    elif env_type == "connect4":
        env = connect_four_v3.raw_env(render_mode="rgb_array")
        env.reset(seed=0)
    return env
