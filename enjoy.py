import numpy as np
import pickle
import torch
from configs import cartpole_masked_config, minigrid_config, poc_memory_env_config
from model import ActorCriticModel
from utils import create_env

def main():
    # Inference device
    device = torch.device("cpu")

    # Instantiate environment
    env = create_env("Minigrid")

    # Initialize model and load its parameters
    model = ActorCriticModel(minigrid_config(), env.observation_space, (env.action_space.n,))
    model.load_state_dict(pickle.load(open("./models/minigrid.nn", "rb")))
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []
    recurrent_cell = model.init_recurrent_cell_states(1, device)
    obs = env.reset()
    while not done:
        # Render environment
        env.render()
        # Forward model
        policy, value, recurrent_cell = model(np.expand_dims(obs, 0), recurrent_cell, device, 1)
        # Sample action
        action = policy.sample().cpu().data.numpy()
        # Step environemnt
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    
    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

if __name__ == "__main__":
    main()