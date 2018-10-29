import torch

import models

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes):
        super(Model, self).__init__(channels, classes)
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 64 -> 32
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 32 -> 16
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 16 -> 32
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 32 -> 64
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
        )
        
        self.distills = torch.nn.ModuleList(self.make_distillpools(classes))

    def make_distillpools(self, classes):
        return [
            models.DistillPool(
                h = models.DenseNet(headsize = 32),
                c = models.Classifier(32, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 32),
                c = models.Classifier(32, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 32),
                c = models.Classifier(32, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 64),
                c = models.Classifier(64, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 64),
                c = models.Classifier(64, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 64),
                c = models.Classifier(64, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 128),
                c = models.Classifier(128, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 128),
                c = models.Classifier(128, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 128),
                c = models.Classifier(128, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 256),
                c = models.Classifier(256, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 256),
                c = models.Classifier(256, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 256),
                c = models.Classifier(256, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 512),
                c = models.Classifier(512, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 512),
                c = models.Classifier(512, classes)
            ),
            models.DistillPool(
                h = models.DenseNet(headsize = 512),
                c = models.Classifier(512, classes)
            ),
        ]
    
    def forward(self, X):
        return sum(self.iter_forward(X))
    
    def iter_forward(self, X):
        X = self.bottleneck(X)
        it = list(self.resnet.iter_forward(X))[-len(self.distills):]
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            yield distill(X)
