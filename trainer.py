import numpy as np
import torch
import time
import os
from torch import optim
from buffer import Buffer
from model import ActorCriticModel
from worker import Worker
from utils import create_env
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device="cpu") -> None:
        """Initializes all needed training components.

        Args:
            config (dict): Configuration and hyperparameters of the environment, trainer and model.
            run_id (str, optional): A tag used to save Tensorboard Summaries. Defaults to "run".
            device (torch.device, optional): Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device

        # Init dummy env and retrieve action and obs spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["env"])
        observation_space = dummy_env.observation_space
        action_space_shape = (dummy_env.action_space.n,)
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, action_space_shape).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["env"]) for w in range(self.config["n_workers"])]

        # Setup observation placeholder   
        self.obs = np.zeros((self.config["n_workers"],) + observation_space.shape, dtype=np.float32)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholders
        for w, worker in enumerate(self.workers):
            obs = worker.child.recv()
            self.obs[w] = obs

    def run_training(self):
        """Runs the entire training logic from sampling data to optimizing the model.
        """
        print("Step 6: Starting training")
        # Store episode results for monitoring these statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer
            self.buffer.prepare_batch_dict(self.episode_done_indices)

            training_stats = self._train_epochs(self.config["learning_rate"], self.config["clip_range"], self.config["beta"])
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)

            episode_result = self._process_episode_info(episode_infos)

            # Print training stats to console and Tensorboard
            if "success_percent" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success = {:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success_percent"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], np.mean(self.buffer.values), np.mean(self.buffer.advantages))
            else:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], np.mean(self.buffer.values), np.mean(self.buffer.advantages))
            print(result)

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            list: list of results of completed episodes.
        """
        episode_infos = []
        # Save the index of a completed episode, which is needed later on to 
        # split the data into episodes and sequences of fixed length
        self.episode_done_indices = [[] for w in range(self.config["n_workers"])]

        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Save the initial observations and hidden states
                self.buffer.obs[:, t] = self.obs
                # Store recurrent cell states inside the buffer
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0).cpu().numpy()
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0).cpu().numpy()
                    self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0).cpu().numpy()

                # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                policy, value, self.recurrent_cell = self.model(self.obs, self.recurrent_cell, self.device)
                self.buffer.values[:, t] = value.cpu().data.numpy()

                # Sample actions
                action = policy.sample()
                log_prob = policy.log_prob(action).cpu().data.numpy()
                action = action.cpu().data.numpy()
                self.buffer.actions[:, t] = action
                self.buffer.log_probs[:, t] = log_prob

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Save the index of a completed episode, which is needed later on to seperate the data into episodes and sequences of fixed length
                    self.episode_done_indices[w].append(t)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Reset recurrent cell states
                    hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                    if self.recurrence["layer_type"] == "gru":
                        self.recurrent_cell[:, w] = hxs
                    elif self.recurrence["layer_type"] == "lstm":
                        self.recurrent_cell[0][:, w] = hxs
                        self.recurrent_cell[1][:, w] = cxs
                # Store latest observations
                self.obs[w] = obs
                            
        # Calculate advantages
        _, last_value, _ = self.model(self.obs, self.recurrent_cell, self.device)
        self.buffer.calc_advantages(last_value.cpu().data.numpy(), self.config["gamma"], self.config["lamda"])

        return episode_infos

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                res = self._train_mini_batch(samples=mini_batch,
                                        learning_rate=learning_rate,
                                        clip_range=clip_range,
                                        beta = beta)
                train_info.append(res)
        return train_info

    def _train_mini_batch(self, samples:Buffer, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Args:
            mini_batch (Buffer): The to be used mini batch data to optimize the model
            learning_rate (float): Current learning rate
            clip_range (float): Current clip range
            beta (float): Current entropy bonus coefficient

        Returns:
            list: list of trainig statistics (e.g. loss)
        """
        # Convert data to tensors
        sampled_return = samples['values'] + samples['advantages']
        # Repeat is necessary for multi-discrete action spaces
        sampled_normalized_advantage = PPOTrainer._normalize(samples['advantages'])

        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Calculate return
        sampled_return = samples['values'] + samples['advantages']
        # Normalize advantages
        sampled_normalized_advantage = (samples['advantages'] - samples['advantages'].mean()) / (samples['advantages'].std() + 1e-8)

        # Forward model
        policy, value, _ = self.model(samples['obs'],
                                    recurrent_cell,
                                    self.device,
                                    self.recurrence["sequence_length"])

        # Compute surrogates
        log_probs = policy.log_prob(samples['actions'])
        ratio = torch.exp(log_probs - samples['log_probs'])
        surr1 = ratio * sampled_normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * sampled_normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = PPOTrainer._masked_mean(policy_loss, samples["loss_mask"])

        # Value
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = PPOTrainer._masked_mean(vf_loss, samples["loss_mask"])
        vf_loss = .25 * vf_loss

        # Entropy Bonus
        entropy_bonus = PPOTrainer._masked_mean(policy.entropy(), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss - vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy()]

    @staticmethod
    def _normalize(adv: np.ndarray):
        """Normalizes the advantage
        Arguments:
            adv {numpy.ndarray} -- The to be normalized advantage
        Returns:
            (adv - adv.mean()) / (adv.std() + 1e-8) {np.ndarray} -- The normalized advantage
        """
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    @staticmethod
    def _masked_mean(tensor:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
            """
            Returns the mean of the tensor but ignores the values specified by the mask.
            This is used for masking out the padding of loss functions.

            Args:
                tensor {Tensor}: The to be masked tensor
                mask {Tensor}: The mask that is used to mask out padded values of a loss function

            Returns:
                {tensor}: Returns the mean of the masked tensor.
            """
            return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

    @staticmethod
    def _process_episode_info(episode_info:list) -> dict:
        """Extracts the mean and std of completed episodes. At minimum the episode length and the collected reward is available.

        Args:
            episode_info (list): list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            dict: Processed episode results (computes the mean, std, min, max for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            keys = episode_info[0].keys()
            for key in keys:
                # skip seed
                if key == "seed":
                    continue
                if key == "success":
                    # This concerns the SimpleMemoryTask only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)

                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_min"] = np.min([info[key] for info in episode_info])
                result[key + "_max"] = np.max([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result