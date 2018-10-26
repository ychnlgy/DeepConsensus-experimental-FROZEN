import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.cos = models.CosineSimilarity()
        self.max = torch.nn.Softmax(dim=1)
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def forward(self, X):
#        N, D = X.size()
#        norm = (X.view(N, D, 1) - self.grp) ** 2
#        return self.min(norm.sum(dim=1))
        cs = self.cos(X, self.grp)
        confidence, indices = cs.max(dim=1)
        confidence = confidence.view(-1, 1)
        confusion = self.max(cs)
        assert len(confidence) == len(confusion)
        return confidence * confusion
