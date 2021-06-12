from environments.cartpole_env import CartPole
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv


def create_env(env_name:str):
    """Initializes an environment based on the provided environment name.
    Args:
        env_name (str): Name of the to be instantiated environment
    Returns:
        env: Returns the selected environment instance.
    """
    if env_name == "PocMemoryEnv":
        return PocMemoryEnv()
    if env_name == "CartPole":
        return CartPole(mask_velocity=False)
    if env_name == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if env_name == "Minigrid":
        return Minigrid()