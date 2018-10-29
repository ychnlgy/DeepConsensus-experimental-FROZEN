import models

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes):
        super(Model, self).__init__(channels, classes)
        self.distills = torch.nn.ModuleList(self.make_distillpools())

    def make_distillpools(self):
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
        ]
    
    def forward(self, X):
        return sum(self.iter_forward(X))
    
    def iter_forward(self, X):
        X = self.conv(X)
        it = list(self.resnet.iter_forward(X))
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            yield distill(X)
