import torch

import models, misc

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes, imagesize, useconsensus, layers, squash, usetanh, optout, useprototype, usenorm, p, alpha):
        super(Model, self).__init__(channels, classes + int(optout), imagesize)
        
        self.useconsensus, layers, squash, usetanh, optout, self.alpha = misc.util.hardmap(
            int,
            useconsensus, layers, squash, usetanh, optout, alpha
        )
        
        self.layers = layers
        self.p = p
        if squash:
            self.squash = [16] * 8
        else:
            self.squash = [32, 32, 64, 64, 128, 128, 256, 256]
        
        self.act = [torch.nn.LeakyReLU, torch.nn.Tanh][usetanh]()
        self.usebias = True
        
        self.optout = optout
        self.useprototype = useprototype
        self.usenorm = usenorm
        
        self.distills = torch.nn.ModuleList(self.make_distillpools(classes))
        self.max = torch.nn.Softmax(dim=1)
        
        self.clear_layereval()
        
        self.register_buffer("layerweights", torch.ones(len(self.distills)))

    def make_distillpools(self, classes):
        return [
            
#            models.GlobalSumPool(
#                h = models.DenseNet(
#                    headsize = 32,
#                    bodysize = 256,
#                    tailsize = self.squash[0],
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = self.act,
#                    bias = self.usebias
#                ),
#                c = models.Classifier(
#                    self.squash[0],
#                    classes + self.optout,
#                    useprototype = self.useprototype,
#                    usenorm = self.usenorm,
#                    p = self.p
#                ),
#                g = models.DenseNet(
#                    headsize = 32,
#                    bodysize = 256,
#                    tailsize = 1,
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = torch.nn.Sigmoid(),
#                    bias = self.usebias
#                ),
#            ),
#            
#            models.GlobalSumPool(
#                h = models.DenseNet(
#                    headsize = 32,
#                    bodysize = 256,
#                    tailsize = self.squash[0],
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = self.act,
#                    bias = self.usebias
#                ),
#                c = models.Classifier(
#                    self.squash[0],
#                    classes + self.optout,
#                    useprototype = self.useprototype,
#                    usenorm = self.usenorm,
#                    p = self.p
#                ),
#                g = models.DenseNet(
#                    headsize = 32,
#                    bodysize = 256,
#                    tailsize = 1,
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = torch.nn.Sigmoid(),
#                    bias = self.usebias
#                ),
#            ),
            
#            models.GlobalSumPool(
#                h = models.DenseNet(
#                    headsize = 64,
#                    bodysize = 256,
#                    tailsize = self.squash[3],
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = self.act,
#                    bias = self.usebias
#                ),
#                c = models.Classifier(
#                    self.squash[3],
#                    classes + self.optout,
#                    useprototype = self.useprototype,
#                    usenorm = self.usenorm,
#                    p = self.p
#                ),
#                g = models.DenseNet(
#                    headsize = 64,
#                    bodysize = 256,
#                    tailsize = 1,
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = torch.nn.Sigmoid(),
#                    bias = self.usebias
#                ),
#            ),
            
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 64,
                    bodysize = 256,
                    tailsize = self.squash[3],
                    layers = self.layers,
                    dropout = 0.2,
                    activation = self.act,
                    bias = self.usebias
                ),
                c = models.Classifier(
                    self.squash[3],
                    classes + self.optout,
                    useprototype = self.useprototype,
                    usenorm = self.usenorm,
                    p = self.p
                ),
                g = models.DenseNet(
                    headsize = 64,
                    bodysize = 256,
                    tailsize = 1,
                    layers = self.layers,
                    dropout = 0.2,
                    activation = torch.nn.Sigmoid(),
                    bias = self.usebias
                ),
            ),
            
#            models.GlobalSumPool(
#                h = models.DenseNet(
#                    headsize = 128,
#                    bodysize = 256,
#                    tailsize = self.squash[5],
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = self.act,
#                    bias = self.usebias
#                ),
#                c = models.Classifier(
#                    self.squash[5],
#                    classes + self.optout,
#                    useprototype = self.useprototype,
#                    usenorm = self.usenorm,
#                    p = self.p
#                ),
#                g = models.DenseNet(
#                    headsize = 128,
#                    bodysize = 256,
#                    tailsize = 1,
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = torch.nn.Sigmoid(),
#                    bias = self.usebias
#                ),
#            ),
            
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = self.squash[5],
                    layers = self.layers,
                    dropout = 0.2,
                    activation = self.act,
                    bias = self.usebias
                ),
                c = models.Classifier(
                    self.squash[5],
                    classes + self.optout,
                    useprototype = self.useprototype,
                    usenorm = self.usenorm,
                    p = self.p
                ),
                g = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 1,
                    layers = self.layers,
                    dropout = 0.2,
                    activation = torch.nn.Sigmoid(),
                    bias = self.usebias
                ),
            ),
            
#            models.GlobalSumPool(
#                h = models.DenseNet(
#                    headsize = 256,
#                    bodysize = 1024,
#                    tailsize = self.squash[7],
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = self.act,
#                    bias = self.usebias
#                ),
#                c = models.Classifier(
#                    self.squash[7],
#                    classes + self.optout,
#                    useprototype = self.useprototype,
#                    usenorm = self.usenorm,
#                    p = self.p
#                ),
#                g = models.DenseNet(
#                    headsize = 256,
#                    bodysize = 64,
#                    tailsize = 1,
#                    layers = self.layers,
#                    dropout = 0.2,
#                    activation = torch.nn.Sigmoid(),
#                    bias = self.usebias
#                ),
#            ),
            
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 256,
                    bodysize = 1024,
                    tailsize = self.squash[7],
                    layers = self.layers,
                    dropout = 0.2,
                    activation = self.act,
                    bias = self.usebias
                ),
                c = models.Classifier(
                    self.squash[7],
                    classes + self.optout,
                    useprototype = self.useprototype,
                    usenorm = self.usenorm,
                    p = self.p
                ),
                g = models.DenseNet(
                    headsize = 256,
                    bodysize = 64,
                    tailsize = 1,
                    layers = self.layers,
                    dropout = 0.2,
                    activation = torch.nn.Sigmoid(),
                    bias = self.usebias
                ),
            )
        ]

from cnn_small import Cnn

class ModelCnn(Model):

    def __init__(self, channels, classes, imagesize, **kwargs):
        super(ModelCnn, self).__init__(channels, classes, imagesize, **kwargs)
        self.layers = Cnn.get_layers(channels, classes, imagesize)
    
    def forward(self, X):
        return sum(self.iter_forward(X))
    
    def iter_forward(self, X):
        assert len(self.distills) == len(self.layers)
        for layer, distill in zip(self.layers, self.distills):
            X = layer(X)
            yield distill(X)
