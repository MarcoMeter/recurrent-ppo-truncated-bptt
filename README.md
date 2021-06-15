# Recurrent Proximal Policy Optimization using Truncated BPTT

This repository features a PyTorch based implementation of PPO using a recurrent policy supporting truncated backpropagation through time. Its intention is to provide a baseline/reference implementation on how to successfully employ recurrent neural networks alongside PPO and similar policy gradient algorithms.

## Features

- Environments
  - Proof-of-concept Memory Task (PocMemoryEnv)
  - CartPole with masked velocity
  - Minigrid Memory using visual observations of size 84x84
- Recurrent Policy
  - GRU
  - LSTM
  - Truncated BPTT
- Tensorboard
- Enjoy (watch a trained agent play)

## Getting Started

To get started check out the [documentation](/documentation/)!


