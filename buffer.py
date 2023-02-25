from gymnasium import spaces
import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, device:torch.device) -> None:
        """
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"]
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        hidden_state_size = config["recurrence"]["hidden_state_size"]
        self.layer_type = config["recurrence"]["layer_type"]
        self.sequence_length = config["recurrence"]["sequence_length"]
        self.true_sequence_length = 0

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.hxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size))
        self.cxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size))
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "obs": self.obs,
            "actions": self.actions,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool)
        }
        
        # Add data concerned with the memory based on recurrence and arrange the entire training data into sequences
        max_sequence_length = 1

        # The loss mask is used for masking the padding while computing the loss function.
        samples["loss_mask"] = torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool)

        # Add collected recurrent cell states to the dictionary
        # Add collected recurrent cell states to the dictionary
        samples["hxs"] =  self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        # Split data into sequences and apply zero-padding
        # Retrieve the indices of dones as these are the last step of a whole episode
        episode_done_indices = []
        for w in range(self.n_workers):
            episode_done_indices.append(list(self.dones[w].nonzero()[0]))
            # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
            if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                episode_done_indices[w].append(self.worker_steps - 1)

        # Retrieve unpadded sequence indices
        self.flat_sequence_indices = np.asarray(self._arange_sequences(
                    np.arange(0, self.n_workers * self.worker_steps).reshape(
                        (self.n_workers, self.worker_steps)), episode_done_indices)[0], dtype=object)
        
        # Split vis_obs, vec_obs, recurrent cell states and actions into episodes and then into sequences
        for key, value in samples.items():
            # Split data into episodes or sequences
            sequences, max_sequence_length = self._arange_sequences(value, episode_done_indices)

            # Apply zero-padding to ensure that each episode has the same length
            # Therfore we can train batches of episodes in parallel instead of one episode at a time
            for i, sequence in enumerate(sequences):
                sequences[i] = self._pad_sequence(sequence, max_sequence_length)

            # Stack sequences (target shape: (Sequence, Step, Data ...) & apply data to the samples dict
            samples[key] = torch.stack(sequences, axis=0)

            if (key == "hxs" or key == "cxs"):
                # Select the very first recurrent cell state of a sequence and add it to the samples
                samples[key] = samples[key][:, 0]

        # Store important information
        self.num_sequences = len(sequences)
            
        self.actual_sequence_length = max_sequence_length
        
        # Add remaining data samples
        samples["values"] = self.values
        samples["log_probs"] = self.log_probs
        samples["advantages"] = self.advantages

        # Flatten samples
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    def _pad_sequence(self, sequence:np.ndarray, target_length:int) -> np.ndarray:
        """Pads a sequence to the target length using zeros.

        Arguments:
            sequence {np.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def _arange_sequences(self, data, episode_done_indices):
        """Splits the povided data into episodes and then into sequences.
        The split points are indicated by the envrinoments' done signals.
        
        Arguments:
            data {torch.tensor} -- The to be split data arrange into num_worker, worker_steps
            episode_done_indices {list} -- Nested list indicating the indices of done signals. Trajectory ends are treated as done
            
        Returns:
            {list} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        for w in range(self.n_workers):
            start_index = 0
            for done_index in episode_done_indices[w]:
                # Split trajectory into episodes
                episode = data[w, start_index:done_index + 1]
                # Split episodes into sequences
                if self.sequence_length > 0:
                    for start in range(0, len(episode), self.sequence_length):
                        end = start + self.sequence_length
                        sequences.append(episode[start:end])
                    max_length = self.sequence_length
                else:
                    # If the sequence length is not set to a proper value, sequences will be based on episodes
                    sequences.append(episode)
                    max_length = len(episode) if len(episode) > max_length else max_length
                start_index = done_index + 1
        return sequences, max_length

    def recurrent_mini_batch_generator(self) -> dict:
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.
        
        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += 1 # Add the remainder if the sequence count and the number of mini batches do not share a common divider
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
            # Unpadded and flat indices are used to sample unpadded training data
            mini_batch_unpadded_indices = self.flat_sequence_indices[sequence_indices[start:end].tolist()]
            mini_batch_unpadded_indices = [item for sublist in mini_batch_unpadded_indices for item in sublist]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "hxs" or key == "cxs":
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
                elif key == "log_probs" or "advantages" in key or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices].to(self.device)
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.device)
            start = end
            yield mini_batch
            
    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]