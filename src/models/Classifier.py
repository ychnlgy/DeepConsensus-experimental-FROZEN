import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.cos = models.CosineSimilarity()
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def forward(self, X):
        return self.cos(X, self.grp)
#        confidence, indices = cs.max(dim=1)
#        confidence = confidence.view(-1, 1)
#        confusion = self.max(cs)
#        assert len(confidence) == len(confusion)
#        return confidence * confusion
