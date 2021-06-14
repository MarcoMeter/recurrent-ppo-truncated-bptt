def cartpole_masked_config():
    return {
        "env": "CartPole",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 100,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 4,
        "learning_rate": 1.0e-4,
        "beta": 0.001,
        "value_loss_coefficient": 0.1,
        "clip_range": 0.2,
        "hidden_layer_size": 128,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 64,
            "layer_type": "lstm"
            }
    }

def minigrid_config():
    return {
        "env": "Minigrid",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 300,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "learning_rate": 2.0e-4,
        "beta": 0.001,
        "value_loss_coefficient": 0.1,
        "clip_range": 0.2,
        "hidden_layer_size": 512,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 256,
            "layer_type": "lstm"
            }
    }

def poc_memory_env_config():
    return {
        "env": "PocMemoryEnv",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 30,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 128,
        "n_mini_batch": 8,
        "learning_rate": 1.0e-4,
        "beta": 0.0001,
        "value_loss_coefficient": 0.1,
        "clip_range": 0.2,
        "hidden_layer_size": 64,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 32,
            "layer_type": "gru"
            }
    }