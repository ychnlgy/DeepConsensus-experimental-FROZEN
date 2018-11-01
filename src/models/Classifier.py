import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes, miu=0, std=0.02):
        super(Classifier, self).__init__()
        self.init_groups(classes, hiddensize, miu, std)
        self.cos = models.CosineSimilarity()
    
    def init_groups(self, classes, hiddensize, miu, std):
        vecs = torch.Tensor(classes, hiddensize).normal_(mean=miu, std=std)
        self.grp = torch.nn.Parameter(vecs)
    
    def forward(self, X):
        return self.cos(X, self.grp)
