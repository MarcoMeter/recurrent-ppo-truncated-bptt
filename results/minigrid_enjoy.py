"""The purpose of this script is to render one minigrid episode to a gif."""

import parent_path
import numpy as np
import pickle
import torch
import os
import inspect
from docopt import docopt
from model import ActorCriticModel
from utils import create_env
import imageio

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        migrid_enjoy.py [options]
        migrid_enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/minigrid.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cpu")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = create_env(config["env"])

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []

    # Init recurrent cell
    hxs, cxs = model.init_recurrent_cell_states(1, device)
    if config["recurrence"]["layer_type"] == "gru":
        recurrent_cell = hxs
    elif config["recurrence"]["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)

    obs = env.reset()
    images = []
    while not done:
        # Render environment
        images.append(env.render())
        # Forward model
        policy, value, recurrent_cell = model(np.expand_dims(obs, 0), recurrent_cell, device, 1)
        # Sample action
        action = policy.sample().cpu().numpy()
        # Step environemnt
        obs, reward, done, info = env.step(int(action))
        episode_rewards.append(reward)
    
    images.append(env.render())
    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))
    
    imageio.mimsave('results/minigrid_gif/MiniGrid.gif', images, fps=1)

    env.close()

if __name__ == "__main__":
    main()