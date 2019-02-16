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
        x = self.fc(x)

        action_scores = self.action(F.relu(x))
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(F.relu(x))

        return Categorical(action_dist), value

    def sample_action_probs(self, x):
        x = self.fc(x)

        action_scores = self.action(F.relu(x))
        action_dist = F.softmax(action_scores, dim=-1)

        return Categorical(action_dist)

    def sample_value(self, x):
        x = self.fc(x)

        value = self.value(F.relu(x))

        return Categorical(value)