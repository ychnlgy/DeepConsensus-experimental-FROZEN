import torch

from .Classifier import Classifier

class Homogenizer(Classifier):

    def forward(self, X):
        W = super(Homogenizer, self).forward(X) # N, classes
        W = torch.nn.functional.softmax(W, dim=1)
        G = self.get_groups() # classes, hiddensize
        C = W.matmul(G) # N, hiddensize
        return C
