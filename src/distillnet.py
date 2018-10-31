import torch

import models

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes):
        super(Model, self).__init__(channels, classes)
        self.distills = torch.nn.ModuleList(self.make_distillpools(classes))
        self.max = torch.nn.Softmax(dim=1)
        print("Using max")

    def make_distillpools(self, classes):
        return [
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.4,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.4,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.4,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.4,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 128,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.6,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 128,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.6,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            ),
            
            models.DistillPool(
                h = models.DenseNet(
                    headsize = 256,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.6,
                    activation = torch.nn.Tanh()
                ),
                c = models.Classifier(8, classes)
            )
        ]
    
    def forward(self, X):
        return sum(self.combine(X))
    
    def iter_forward(self, X):
        X = self.conv(X)
        it = list(self.resnet.iter_forward(X))
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            yield distill(X)
    
    def combine(self, X):
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
