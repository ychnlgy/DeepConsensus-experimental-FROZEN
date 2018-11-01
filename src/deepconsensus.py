import torch

import models

from resnet64 import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes, imagesize):
        super(Model, self).__init__(channels, classes, imagesize)
        self.distills = torch.nn.ModuleList(self.make_distillpools(classes))
        self.max = torch.nn.Softmax(dim=1)

    def make_distillpools(self, classes):
        return [
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(32, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(32, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(64, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.6),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.6),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(256, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.6),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            ),
            
            models.GlobalSumPool(
                h = torch.nn.Sequential(
                    torch.nn.Linear(256, 64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Dropout(p=0.6),
                    torch.nn.Linear(64, 8),
                    torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes + 1)
            )
        ]
    
    def forward(self, X):
        return sum(self.do_consensus(X))
    
    def iter_forward(self, X):
        X = self.conv(X)
        it = list(self.resnet.iter_forward(X))
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            yield distill(X)
    
    def do_consensus(self, X):
        it = self.iter_forward(X)
        a = next(it)
        b = next(it)
        m = self.max(a)
        n = self.max(b)
        p = None
        yield a * n
        for c in it:
            p = self.max(c)
            yield (m + p)/2.0 * b
            a, b = b, c
            m, n = n, p
        yield m * b
