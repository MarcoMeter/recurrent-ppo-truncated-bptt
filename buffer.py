from gym import spaces
import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, device:torch.device):
        """
        Args:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
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
        self.num_sequences = 0
        self.actual_sequence_length = 0

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = np.zeros((self.n_workers, self.worker_steps) + observation_space.shape, dtype=np.float32)
        self.hxs = np.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=np.float32)
        self.cxs = np.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=np.float32)
        self.log_probs = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

    def prepare_batch_dict(self, episode_done_indices:list):
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        
        Args:
            episode_done_indices {list} -- Nested list that stores the done indices of each worker"""
        # Supply training samples
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "obs": self.obs,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": np.ones((self.n_workers, self.worker_steps), dtype=np.float32)
        }
        
        # Add collected recurrent cell states to the dictionary
        samples["hxs"] =  self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        # Split data into sequences and apply zero-padding
        # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
        for w in range(self.n_workers):
            if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                episode_done_indices[w].append(self.worker_steps - 1)
        
        # Split obs, values, advantages, recurrent cell states, actions and log_probs into episodes and then into sequences
        max_sequence_length = 1
        for key, value in samples.items():
            sequences = []
            for w in range(self.n_workers):
                start_index = 0
                for done_index in episode_done_indices[w]:
                    # Split trajectory into episodes
                    episode = value[w, start_index:done_index + 1]
                    start_index = done_index + 1
                    # Split episodes into sequences
                    if self.sequence_length > 0:
                        for start in range(0, len(episode), self.sequence_length):
                            end = start + self.sequence_length
                            sequences.append(episode[start:end])
                        max_sequence_length = self.sequence_length
                    else:
                        # If the sequence length is not set to a proper value, sequences will be based on whole episodes
                        sequences.append(episode)
                        max_sequence_length = len(episode) if len(episode) > max_sequence_length else max_sequence_length
            
            # Apply zero-padding to ensure that each sequence has the same length
            # Therfore we can train batches of sequences in parallel instead of one episode at a time
            for i, sequence in enumerate(sequences):
                sequences[i] = self.pad_sequence(sequence, max_sequence_length)

            # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dict
            samples[key] = np.stack(sequences, axis=0)

            if (key == "hxs" or key == "cxs"):
                # Select only the very first recurrent cell state of a sequence and add it to the samples
                samples[key] = samples[key][:, 0]

        # Store important information concerning the sequences
        self.num_sequences = len(samples["values"])
        self.actual_sequence_length = max_sequence_length
        
        # Flatten all samples and convert them to a tensor
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = torch.tensor(value, dtype = torch.float32, device = self.device)

    def pad_sequence(self, sequence:np.ndarray, target_length:int):
        """Pads a sequence to the target length using zeros.

        Args:
            sequence {np.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {np.ndarray} -- Returns the padded sequence
        """
        # If a tensor is provided, convert it to a numpy array
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.numpy()
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = np.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = np.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return np.concatenate((sequence, padding), axis=0)

    def recurrent_mini_batch_generator(self):
        """A recurrent generator that returns a dictionary providing training data arranged in mini batches.
        This generator shuffles the data by sequences.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        # Arrange a list that determines the sequence count for each mini batch
        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            # Add the remainder if the sequence count and the number of mini batches do not share a common divider
            num_sequences_per_batch[i] += 1
        # Prepare indices, but only shuffle the sequence indices and not the entire batch.
        indices = np.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences)
        # At this point it is assumed that all of the available training data (values, observations, actions, ...) is padded.

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key != "hxs" and key != "cxs":
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
                else:
                    # Collect only the recurrent cell states that are at the beginning of a sequence
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
            start = end
            yield mini_batch

    def calc_advantages(self, last_value:np.ndarray, gamma:float, lamda:float):
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {np.ndarray} -- Value of the last agent"s state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        last_advantage = 0
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - self.dones[:, t] # mask value on a terminal state (i.e. done)
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = self.rewards[:, t] + gamma * last_value - self.values[:, t]
            last_advantage = delta + gamma * lamda * last_advantage
            self.advantages[:, t] = last_advantage
            last_value = self.values[:, t]