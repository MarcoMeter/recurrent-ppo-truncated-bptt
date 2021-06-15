# Train a model

The training is launched via the command `python train.py`.

```python
"""
    Usage:
        train.py [options]
        train.py --help

    Options:
        --run-id=<path>            Specifies the tag of the tensorboard summaries and the model's filename [default: run].
        --cpu                      Whether to enforce training on the CPU, otherwwise an available GPU will be used. [default: False].
"""
```

Hyperparameters are configured inside of `configs.py`. The to be used config has to be specified inside of `train.py`. Once the training is done, the final model will be saved to `./models/run-id.nn`. Training statistics are stored inside the `./summaries` directory.