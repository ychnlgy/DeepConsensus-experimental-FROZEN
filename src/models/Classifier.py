import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.init_groups(classes, hiddensize)
        self.cos = models.Norm()
    
    def init_groups(self, classes, hiddensize):
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def get_mean_repr(self):
        return self.grp.mean(dim=0)
    
    def forward(self, X):
        return -self.cos(X, self.grp)
