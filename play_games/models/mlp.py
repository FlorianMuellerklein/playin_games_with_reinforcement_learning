import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPolicy(nn.Module):
    def __init__(self, input_dims, n_actions, n_hidden):
        super(MLPolicy, self).__init__()
        self.fc = nn.Linear(input_dims, n_hidden)
        self.action = nn.Linear(n_hidden, n_actions)
        self.value = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))

        action_scores = self.action(x)
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(x)

        return action_dist, value.squeeze()

    def get_action(self, x):
        with torch.no_grad():
            x = F.relu(self.fc(x))

            action_scores = self.action(x)
            action_dist = F.softmax(action_scores, dim=-1)

        return action_dist