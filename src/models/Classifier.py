import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes, gamma=100):
        super(Classifier, self).__init__()
        self.gamma = gamma
        self.init_groups(classes, hiddensize)
        self.norm = models.SoftminNorm()
        self.cos = models.CosineSimilarity()
        self.max = torch.nn.Softmax(dim=1)
    
    def init_groups(self, classes, hiddensize):
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
    
    def get_class_vec(self, c):
        return self.grp[c]
    
    def forward(self, X):
        norm = self.norm(X, self.grp)

        cs = self.cos(X, self.grp)
        cs = self.max(cs * self.gamma)
        #confidence, indices = cs.max(dim=1)
        #confidence = confidence.view(-1, 1)
        #confusion = self.max(cs)
        #assert len(confidence) == len(confusion)
        return cs * norm#confidence * confusion
