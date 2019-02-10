import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUPolicy(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(GRUPolicy, self).__init__()
        self.gru = nn.GRUCell(input_dims, 64)
        self.action = nn.Linear(64, n_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x, h):
        h = self.gru(x, h)

        action_scores = self.action(h)
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(h)

        return action_dist, value, h

    def sample_action_probs(self, x, h):
        h = self.gru(x, h)

        action_scores = self.action(h)
        action_dist = F.softmax(action_scores, dim=-1)

        return action_dist, h

    def sample_value(self, x, h):
        h = self.gru(x, h)

        value = self.value(h)

        return value, h