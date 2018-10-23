import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.cos = models.CosineSimilarity()
        self.dif = models.Norm(p=2)
        self.max = torch.nn.Softmax(dim=1)
    
    def forward(self, X):
        cs = self.cos(X, self.grp)
        confidence, indices = cs.max(dim=1)
        confidence = confidence.view(-1, 1)
        confusion = self.max(-self.dif(X, self.grp))
        assert len(confidence) == len(confusion)
        return confidence * confusion
