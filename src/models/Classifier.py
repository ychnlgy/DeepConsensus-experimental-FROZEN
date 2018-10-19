import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.cos = models.CosineDissimilarity()
    
    def forward(self, X):
        return self.cos(X, self.grp)
