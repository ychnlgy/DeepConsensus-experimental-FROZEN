import torch

from .Savable import Savable

class Discriminator(Savable):
    
    def __init__(self, classifier, net):
        super(Discriminator, self).__init__()
        self.classifier = classifier
        self.params = torch.nn.Parameter(
            self.get_classifierparams()
        )
        self.net = net
    
    def forward(self):
        assert (self.get_classifierparams() == self.params).all()
        return self.net(self.params)
    
    def get_classifierparams(self):
        return torch.cat([
            p.view(-1) for p in self.classifier.parameters()
        ])
