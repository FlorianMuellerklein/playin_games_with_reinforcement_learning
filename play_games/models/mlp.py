import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MLPolicy(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_sz):
        super(MLPolicy, self).__init__()
        self.fc = nn.Linear(input_dims, hidden_sz)
        self.action = nn.Linear(hidden_sz, n_actions)
        self.value = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))

        action_scores = self.action(x)
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(x)

        return Categorical(action_dist), value.squeeze()

    def get_action(self, x):
        with torch.no_grad():
            x = F.relu(self.fc(x))

            action_scores = self.action(x)
            action_dist = F.softmax(action_scores, dim=-1)

        return Categorical(action_dist)