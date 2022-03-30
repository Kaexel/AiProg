import os
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self, x, y, w):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Conv2d(9, w, (5, 5), stride=(1, 1)), #TODO: remember to pad image with two rows/columns of black/white
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, 1, (1, 1), stride=(1, 1), padding=1),  # TODO: output must be x*y, and scaled to x*y - q
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

t = NeuralNetwork(1, 2, 3)
