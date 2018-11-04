import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes, useprototype, usenorm, miu=0, std=0.02):
        super(Classifier, self).__init__()
        
        self.hiddensize, self.classes = hiddensize, classes
        useprototype, usenorm = int(useprototype), int(usenorm)
        
        self.grp  = self.init_groups(classes, hiddensize, miu, std, useprototype)
        self.mech = self.init_mech(useprototype, usenorm)
        
        self.norm = models.SoftminNorm()
        self.coss = models.CosineSimilarity()
    
    def init_groups(self, classes, hiddensize, miu, std, useprototype):
        if useprototype:
            vecs = torch.Tensor(classes, hiddensize).normal_(mean=miu, std=std)
            self._grp = torch.nn.Parameter(vecs)
            return [self._grp]
        else:
            return []
    
    def init_mech(self, useprototype, usenorm):
        if not useprototype:
            return torch.nn.Sequential(
                torch.nn.Linear(self.hiddensize, self.classes),
                torch.nn.Tanh()
            )
        else:
            if usenorm:
                return models.SoftminNorm()
            else:
                return models.CosineSimilarity()
    
    def forward(self, X):
        #return self.mech(X, *self.grp)
        return self.norm(X, *self.grp) * self.coss(X, *self.grp)
