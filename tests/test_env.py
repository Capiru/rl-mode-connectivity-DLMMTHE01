from alphago.env import get_env, tictactoe_v3, connect_four_v3, chess_v6, go_v5

def test_env_tictactoe():
    env = get_env("tictactoe")
    assert isinstance(env,tictactoe_v3.raw_env)
    
def test_env_chess():
    env = get_env("connect4")
    assert isinstance(env,connect_four_v3.raw_env)
    
def test_env_tictactoe():
    env = get_env("chess")
    assert isinstance(env,chess_v6.raw_env)
    
def test_env_tictactoe():
    env = get_env("go")
    assert isinstance(env,go_v5.raw_env)