# Recurrent Policy

## Implementation Concept

## Found & Fixed Bugs

As a reinforcement learning engineer, one has to have high endurance. Therefore, we are providing some information on the bugs that slowed us down for months.

### Feeding None to nn.GRU/nn.LSTM

We observed an **exploding value function**. This was due to unintentionally feeding None to the recurrent layer. In this case, PyTorch uses zeros for the hidden states as shown by its [source code](https://github.com/pytorch/pytorch/blob/8d50a4e326e10fe29e322753bb90be15546e5435/torch/nn/modules/rnn.py#L662).

```python
if hx is None:
    num_directions = 2 if self.bidirectional else 1
    real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
    h_zeros = torch.zeros(self.num_layers * num_directions,
                          max_batch_size, real_hidden_size,
                          dtype=input.dtype, device=input.device)
    c_zeros = torch.zeros(self.num_layers * num_directions,
                          max_batch_size, self.hidden_size,
                          dtype=input.dtype, device=input.device)
    hx = (h_zeros, c_zeros)
```

### Reshaping an Entire Batch into Sequences

Training an agent using a **sequence length greater than 1** caused the agent to just achieve a **performance of a random agent**. The issue behind this bug was found in reshaping the data right before feeding it to the recurrent layer. In general, the desire is to feed the entire training batch instead of sequences to the encoder (e.g. convolutional layers). Before feeding the processed batch to the recurrent layer, it has to be rearranged into sequences. At the point of this bug, the recurrent layer was initialized with `batch_first=False`. Hence, the data was reshaped using `h.reshape(sequence_length, num_sequences, data)`. This messed up the structure of the sequences and ultimately caused this bug. We fixed this by setting `batch_first` to `True` and therefore reshaping the data by `h.reshape(num_sequences, sequence_length, data)`.