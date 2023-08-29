import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):

    def __init__(self, input_shape, hidden_size, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())