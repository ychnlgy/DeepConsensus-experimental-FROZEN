import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.init_groups(classes, hiddensize)
        #self.cos = models.Norm(p=1)
        #self.max = torch.nn.Softmax(dim=1)
    
    def init_groups(self, classes, hiddensize):
        self.grp = torch.nn.Parameter(torch.rand(1, hiddensize, classes))
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def forward(self, X):
        N, D = X.size()
        norm = (X.view(N, D, 1) - self.grp).abs().sum(dim=1)
        return torch.exp(-norm)
#        cs = self.cos(X, self.grp)
#        confidence, indices = cs.max(dim=1)
#        confidence = confidence.view(-1, 1)
#        confusion = self.max(cs)
#        assert len(confidence) == len(confusion)
#        return confidence * confusion
