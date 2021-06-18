from gym.core import Env
import numpy as np
import random
import copy
import torch

class Buffer:
    def __init__(self, n_workers, n_steps, sequence_length) -> None:
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.sequence_length = sequence_length
        self.n_mini_batches = 3

        self.obs = np.zeros((n_workers, n_steps, 2), dtype=np.float32)
        self.dones = np.zeros((n_workers, n_steps), dtype=bool)

    def prepare_data(self):
        samples = {
            "obs": self.obs,
            "dones": self.dones,
            "hxs": copy.deepcopy(self.obs)
        }

        episode_done_indices = []
        for w in range(self.n_workers):
            episode_done_indices.append(list(self.dones[w].nonzero()[0]))
            # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
            if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.n_steps - 1:
                episode_done_indices[w].append(self.n_steps - 1)

        # Test episode done indices
        # Check if the episode done index is actually done
        # Don't do that for the last element of a worker'S data
        for i in range(len(episode_done_indices)):
            for index in episode_done_indices[i]:
                if index != self.n_steps - 1:
                    assert self.dones[i, index]

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

            # Test sequence data being equal to n_workers * n_steps
            count = 0
            for sequence in sequences:
                count += len(sequence)
            assert self.n_workers * self.n_steps == count

            # Test sequence length
            assert self.sequence_length == max_sequence_length

            # Pad sequences
            for i, sequence in enumerate(sequences):
                sequences[i] = self.pad_sequence(sequence, max_sequence_length)

            # Check that each padded sequence is as long as desired
            for seq in sequences:
                assert len(seq) == max_sequence_length

            # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dict
            samples[key] = np.stack(sequences, axis=0)

            # select hidden states
            if key == "hxs":
                samples[key] = samples[key][:, 0]

        self.true_sequence_length = max_sequence_length

        # Flatten data
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            else:
                self.samples_flat[key] = value
            self.samples_flat[key] = torch.tensor(value, dtype = torch.float32)
        
        # print(self.samples_flat["hxs"].size())
        # print(self.samples_flat["hxs"])

    def pad_sequence(self, sequence:np.ndarray, target_length:int) -> np.ndarray:
            """Pads a sequence to the target length using zeros.

            Args:
                sequence {np.ndarray} -- The to be padded array (i.e. sequence)
                target_length {int} -- The desired length of the sequence

            Returns:
                {np.ndarray} -- Returns the padded sequence
            """
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

    def recurrent_mini_batch_generator(self) -> dict:
            """A recurrent generator that returns a dictionary providing training data arranged in mini batches.
            This generator shuffles the data by sequences.

            Yields:
                {dict} -- Mini batch data for training
            """
            # Determine the number of sequences per mini batch
            num_sequences = len(self.samples_flat["dones"]) // self.true_sequence_length
            num_sequences_per_batch = num_sequences // self.n_mini_batches
            # Arrange a list that determines the sequence count for each mini batch
            num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches
            remainder = num_sequences % self.n_mini_batches
            for i in range(remainder):
                # Add the remainder if the sequence count and the number of mini batches do not share a common divider
                num_sequences_per_batch[i] += 1
            # Prepare indices, but only shuffle the sequence indices and not the entire batch.
            indices = np.arange(0, num_sequences * self.true_sequence_length).reshape(num_sequences, self.true_sequence_length)
            sequence_indices = torch.randperm(num_sequences)
            # At this point it is assumed that all of the available training data (values, observations, actions, ...) is padded.

            # Compose mini batches
            start = 0
            for n_sequences in num_sequences_per_batch:
                end = start + n_sequences
                mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
                mini_batch = {}
                for key, value in self.samples_flat.items():
                    if key != "hxs" and key != "cxs":
                        mini_batch[key] = value[mini_batch_indices]
                    else:
                        # Collect only the recurrent cell states that are at the beginning of a sequence
                        mini_batch[key] = value[sequence_indices[start:end]]
                start = end
                yield mini_batch

class Environment:
    def __init__(self, id):
        self.id = id

    def reset(self):
        self.steps = 0
        self.done_target = random.randint(3, 12)
        return [self.id, self.steps]

    def step(self):
        self.steps += 1
        done = False
        if self.done_target == self.steps:
            done = True
        return [self.id, self.steps], done

def main():
    # Init stuff
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_workers = 2
    n_steps = 40
    sequence_length = 8
    buffer = Buffer(n_workers, n_steps, sequence_length)
    envs = [Environment(w) for w in range(n_workers)]
    obs_placeholder = np.zeros((n_workers, 2), dtype=np.float32)

    # Reset envs
    for e, env in enumerate(envs):
        obs_placeholder[e] = np.asarray(env.reset(), dtype=np.float32)

    # Sample data
    for s in range(n_steps):
        buffer.obs[:, s] = obs_placeholder
        for e, env in enumerate(envs):
            obs_placeholder[e], buffer.dones[e, s] = env.step()
            if buffer.dones[e, s]:
               obs_placeholder[e] = env.reset()

    # Prepare data
    buffer.prepare_data()

    # generate mini batches
    for mb in buffer.recurrent_mini_batch_generator():
        pass


if __name__ == "__main__":
    main()
