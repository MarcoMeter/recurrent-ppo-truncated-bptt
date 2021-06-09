import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape, recurrence):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant detais:
                - layer type {stirng}, sequence length {int}, hidden state size {int}, hiddens state initialization {string}, fake recurrence {bool}
        """
        super().__init__()
        # Size of hidden layers
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = recurrence
        
        # Set the activation function for most layers of the neural net
        available_activ_fns = {
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "relu": F.relu,
            "swish": F.silu
        }
        self.activ_fn = available_activ_fns[config["activation"]]

        # Observation encoder
        if vis_obs_space is not None:
            # Case: visual observation available
            vis_obs_shape = vis_obs_space.shape
            # Visual Encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(in_channels=vis_obs_shape[0],
                                out_channels=32,
                                kernel_size=8,
                                stride=4,
                                padding=0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))

            self.conv2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=4,
                                stride=2,
                                padding=0)
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))

            self.conv3 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=0)
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))

            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(vis_obs_shape)
            in_features_next_layer = self.conv_out_size

            # Determine number of features for the next layer's input
            if vec_obs_shape is not None:
                # Case: vector observation is also available
                in_features_next_layer = in_features_next_layer + vec_obs_shape[0]
        else:
            # Case: only vector observation is available
            in_features_next_layer = vec_obs_shape[0]

        # Recurrent Layer (GRU or LSTM)
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
            elif self.recurrence["layer_type"] == "lstm":
                self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
            # Init recurrent layer
            for name, param in self.recurrent_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, np.sqrt(2))
            # Hidden layer
            self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        else:
            # Hidden layer
            self.lin_hidden = nn.Linear(in_features_next_layer, self.hidden_size)

        # Init Hidden layer
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model Heads
        # Policy
        self.policy = nn.Linear(self.hidden_size, action_space_shape[0])
        nn.init.orthogonal_(self.policy.weight, np.sqrt(0.01))

        # Value Function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, vis_obs, vec_obs, recurrent_cell, device:torch.device, sequence_length:int=1):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch,tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer (None if not available)
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        h: torch.Tensor

        # Forward observation encoder
        if vis_obs is not None:
            vis_obs = torch.tensor(vis_obs, dtype=torch.float32, device=device)      # Convert vis_obs to tensor
            batch_size = vis_obs.size()[0]
            # Propagate input through the visual encoder
            h = self.activ_fn(self.conv1(vis_obs))
            h = self.activ_fn(self.conv2(h))
            h = self.activ_fn(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, vec_obs), 1)
        else:
            h = torch.tensor(vec_obs, dtype=torch.float32, device=device)        # Convert vec_obs to tensor

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            if sequence_length == 1:
                # Case: sampling training data or model optimization using fake recurrence
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
                h = h.squeeze(1) # Remove sequence length dimension
            else:
                # Case: Model optimization given a sequence length > 1
                # Reshape the to be fed data to batch_size, sequence_length, data
                h_shape = tuple(h.size())
                h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

                # Forward recurrent layer
                h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

                # Reshape to the original tensor size
                h_shape = tuple(h.size())
                h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # Feed hidden layer
        h = self.activ_fn(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.activ_fn(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = self.activ_fn(self.lin_value(h))
        # Output: Value Function
        value = self.value(h_value).reshape(-1)
        # Output: Policy Branches
        pi = Categorical(logits=self.policy(h_policy))

        return pi, value, recurrent_cell

    def get_conv_output(self, shape):
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
 
    def init_recurrent_cell_states(self, num_sequences, device):
        """Initializes the recurrent cell states (hxs, cxs) based on the configured method and the used recurrent layer type.
        These states can be initialized in 4 ways:

        - zero
        - one
        - mean (based on the recurrent cell states of the sampled training data)
        - sample (based on the mean of all recurrent cell states of the sampled training data, the std is set to 0.01)

        Arugments:
            num_sequences {int}: The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device}: Target device.

        Returns:
            {tuple}: Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned using initial values.
        """
        hxs, cxs = None, None
        if self.recurrence["hidden_state_init"] == "zero":
            hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "one":
            hxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        return hxs, cxs