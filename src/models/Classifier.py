import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.dif = models.Norm(p=1)
    
    def forward(self, X):
        return torch.exp(-self.dif(X, self.grp))
