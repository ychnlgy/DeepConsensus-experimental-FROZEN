import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.init_groups(classes, hiddensize)
        self.cos = models.CosineSimilarity()
        self.max = torch.nn.Softmax(dim=1)
        self.min = torch.nn.Softmin(dim=1)
    
    def init_groups(self, classes, hiddensize):
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def forward(self, X):
        N, D = X.size()
        norm = (X.view(N, 1, D) - self.grp.view(1, -1, D)).abs().sum(dim=1)
        norm = self.min(norm)
        
        cs = self.cos(X, self.grp)
        confidence, indices = cs.max(dim=1)
        confidence = confidence.view(-1, 1)
        confusion = self.max(cs)
        assert len(confidence) == len(confusion)
        print(confidence.size(), confusion.size(), norm.size())
        input()
        return confidence * confusion * norm
