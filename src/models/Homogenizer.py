import torch

from .Classifier import Classifier

class Homogenizer(Classifier):

    def forward(self, X):
        W = super(Homogenizer, self).forward(X) # N, classes
        G = self.get_groups() # classes, hiddensize
        C = W.matmul(G) # N, hiddensize
        return C
