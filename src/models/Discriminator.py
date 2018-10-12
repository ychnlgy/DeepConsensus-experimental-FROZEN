import torch

from .Savable import Savable

class Discriminator(Savable):
    
    def __init__(self, classifier, net):
        super(Discriminator, self).__init__()
        self.params = torch.cat([p.view(-1) for p in classifier.parameters()])
        self.net = net
    
    def forward(self):
        return self.net(self.params)
