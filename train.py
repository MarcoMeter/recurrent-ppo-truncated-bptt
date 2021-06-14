from trainer import PPOTrainer
from docopt import docopt
import torch

def create_default_config() -> dict:
    return {
        "env": "Minigrid", # PocMemoryEnv, CartPole, CartPoleMasked, Minigrid
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 300,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "learning_rate": 2.0e-4,
        "beta": 0.001,
        "value_loss_coefficient": 0.25,
        "clip_range": 0.2,
        "hidden_layer_size": 512,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 256,
            "layer_type": "lstm"
            }
    }

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Init Trainer and Launch training
    PPOTrainer(create_default_config(), run_id=run_id, device=device).run_training()

if __name__ == "__main__":
    main()


