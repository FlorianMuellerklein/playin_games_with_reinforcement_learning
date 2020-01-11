import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPolicy(nn.Module):
    def __init__(self, n_actions):
        super(ConvPolicy, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 16, kernel_size=8, stride=4),
                                 #nn.BatchNorm2d(32),
        	                     nn.ReLU(inplace=True),
                                 #nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(16, 32, kernel_size=4, stride=2),
                                 #nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 #nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                 #nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))

        #for m in self.cnn.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))

        self.fc = nn.Sequential(nn.Linear(3456, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU())

        self.action = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        action_scores = self.action(x)
        action_dist = F.softmax(action_scores, dim=-1)
       
        value = self.value(x)

        return action_dist, value