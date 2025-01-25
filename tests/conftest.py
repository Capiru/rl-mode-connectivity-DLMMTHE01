import pytest
from rl.config import Settings


@pytest.fixture
def cfg():
    return Settings(
        env_type="connect4", model_type="mlp", num_simulations=100, discount_factor=0.98
    )
