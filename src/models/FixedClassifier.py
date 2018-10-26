import torch

from .Classifier import Classifier

class FixedClassifier(Classifier):

    def __init__(self, alpha, *args, **kwargs):
        super(FixedClassifier, self).__init__(*args, **kwargs)
        self.alpha = alpha
    
    def init_groups(self, classes, hiddensize):
        self.register_buffer("grp", torch.rand(classes, hiddensize))
    
    def forward(self, X, y=None):
        if y is not None:
            self.update(X, y)
        return super(FixedClassifier, self).forward(X)
    
    def update(self, X, y):
        self.grp[y] = self.alpha * self.grp[y] + (1-self.alpha) * X.mean(dim=0)
