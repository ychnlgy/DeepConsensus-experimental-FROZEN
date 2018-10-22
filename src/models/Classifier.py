import torch

import models

class Classifier(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(Classifier, self).__init__()
        self.grp = torch.nn.Parameter(torch.rand(classes, hiddensize))
        self.cos = models.CosineSimilarity()
    
    def forward(self, X):

        '''
        
        Given:
            X - Tensor of shape (N, D)
        
        Returns:
            Tensor of shape (N, classes)
        
        '''
    
        return self.cos(X, self.grp)
    
    def get_groups(self):
        return self.grp
