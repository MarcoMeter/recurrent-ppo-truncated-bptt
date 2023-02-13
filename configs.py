def cartpole_masked_config():
    return {
        "environment":
            {"type": "CartPoleMasked"},
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 100,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 4,
        "value_loss_coefficient": 0.2,
        "hidden_layer_size": 128,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 64,
            "layer_type": "lstm",
            "reset_hidden_state": True
            },
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-6,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 1000
            }
    }

def minigrid_config():
    return {
        "environment":
            {"type": "Minigrid"},
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 500,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.25,
        "hidden_layer_size": 512,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 256,
            "layer_type": "lstm",
            "reset_hidden_state": False
            },
        "learning_rate_schedule":
            {
            "initial": 2.0e-4,
            "final": 2.0e-4,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.001,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 300
            }
    }

def poc_memory_env_config():
    return {
        "environment":
            {"type": "PocMemoryEnv"},
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 30,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 128,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.1,
        "hidden_layer_size": 64,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 32,
            "layer_type": "gru",
            "reset_hidden_state": True
            },
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-4,
            "power": 1.0,
            "max_decay_steps": 30
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 30
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 30
            }
    }
  
def memory_gym_config():
    return {
        "environment":
            {
            "type": "MortarMayhem-Grid",
            "name": "MortarMayhem-Grid-v0",
            "reset_params": 
                {
                "start-seed": 0,
                "num-seeds": 10000,
                "agent_scale": 0.25,
                "arena_size": 5,
                "allowed_commands": 5,
                "command_count": [10],
                "explosion_duration": [2],
                "explosion_delay": [5],
                "reward_command_failure": 0.0,
                "reward_command_success": 0.1,
                "reward_episode_success": 0.0
                }
            }, 
        "gamma": 0.995,
        "lamda": 0.95,
        "updates": 10000,
        "epochs": 3,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.25,
        "hidden_layer_size": 512,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 256,
            "layer_type": "lstm",
            "reset_hidden_state": False
            },
        "learning_rate_schedule":
            {
            "initial": 2.0e-4,
            "final": 2.0e-4,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.001,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 300
            }
    }