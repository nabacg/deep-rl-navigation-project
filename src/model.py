import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layer_1_size=64, layer_2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        self.model = nn.Sequential(
            nn.Linear(state_size, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
