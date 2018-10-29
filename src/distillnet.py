import torch

import models

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes):
        super(Model, self).__init__(channels, classes)
        self.distills = torch.nn.ModuleList(self.make_distillpools(classes))

    def make_distillpools(self, classes):
        return [
            models.DistillPool(
                h = models.DenseNet(
                    headsize=32,
                    bodysize=64,
                    tailsize=32,
                    layers = 2,
                    dropout = 0.2
                ),
                c = models.Classifier(32, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize=64,
                    bodysize=128,
                    tailsize=64,
                    layers = 2,
                    dropout = 0.2
                ),
                c = models.Classifier(64, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize=128,
                    bodysize=64,
                    tailsize=128,
                    layers = 2,
                    dropout = 0.2
                ),
                c = models.Classifier(128, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(
                    headsize=256,
                    bodysize=64,
                    tailsize=128,
                    layers = 2,
                    dropout = 0.2
                ),
                c = models.Classifier(128, classes)
            )
        ]
    
    def forward(self, X):
        return sum(self.iter_forward(X))
    
    def iter_forward(self, X):
        X = self.conv(X)
        it = list(self.resnet.iter_forward(X))
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            yield distill(X)
