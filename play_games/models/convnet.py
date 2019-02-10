import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPolicy(nn.Module):
    def __init__(self, n_actions):
        super(ConvPolicy, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                                 nn.BatchNorm2d(16),
        	                     nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2))
        self.gru = nn.GRUCell(64, 64)
        self.action = nn.Linear(28800, n_actions)
        self.value = nn.Linear(28800, 1)

    def forward(self, x, h):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        h = self.gru(x, h)

        action_scores = self.action(h)
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(x)

        return action_dist, value, h

    def sample_action_probs(self, x, h):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        h = self.grux(x, h)
 
        action_scores = self.action(h)
        action_dist = F.softmax(action_scores, dim=-1)

        return action_dist, h

    def sample_value(self, x, h):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        h = self.gru(x, h)

        value = self.value(h)

        return value, h