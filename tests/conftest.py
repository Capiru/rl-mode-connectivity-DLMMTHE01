import pytest
from alphago.config import Settings


@pytest.fixture
def cfg():
    return Settings(env_type="connect4", model_type="convnet", num_simulations=100)
