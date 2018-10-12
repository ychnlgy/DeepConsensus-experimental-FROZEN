import torch

from .Savable import Savable

class Discriminator(Savable):
    
    def __init__(self, classifier, net):
        super(Discriminator, self).__init__()
        self.classifier = classifier
        self.net = net
    
    def forward(self):
        return self.net(self.get_classifierparams())
    
    def get_classifierparams(self):
        return torch.cat([
            p.view(-1) for p in self.classifier.parameters()
        ])
