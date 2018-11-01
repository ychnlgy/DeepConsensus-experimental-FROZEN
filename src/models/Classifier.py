import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes, useprototype, usenorm, miu=0, std=0.02):
        super(Classifier, self).__init__()
        
        useprototype, usenorm = int(useprototype), int(usenorm)
        
        self.grp  = self.init_groups(classes, hiddensize, miu, std, useprototype)
        self.mech = self.init_mech(useprototype, usenorm)
    
    def init_groups(self, classes, hiddensize, miu, std, useprototype):
        if useprototype:
            print(classes, hiddensize)
            input()
            vecs = torch.Tensor(classes, hiddensize).normal_(mean=miu, std=std)
            return [torch.nn.Parameter(vecs)]
        else:
            return []
    
    def init_mech(self, useprototype, usenorm):
        if not useprototype:
            return torch.nn.Sequential(
                torch.nn.Linear(hiddensize, classes),
                torch.nn.Tanh()
            )
        else:
            if usenorm:
                return models.SoftminNorm()
            else:
                return models.CosineSimilarity()
    
    def forward(self, X):
        return self.mech(X, *self.grp)
