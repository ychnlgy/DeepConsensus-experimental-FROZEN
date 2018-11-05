import torch

import models, misc

from resnet import Model as ResNet

class Model(ResNet):

    def __init__(self, channels, classes, imagesize, useconsensus, layers, squash, usetanh, optout, useprototype, usenorm, p):
        super(Model, self).__init__(channels, classes, imagesize)
        
        self.useconsensus, layers, squash, usetanh, optout = misc.util.hardmap(
            int,
            useconsensus, layers, squash, usetanh, optout
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

    def make_distillpools(self, classes):
        return [
            
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
    
    def forward(self, X):
        out = sum(self.do_consensus(X))
        misc.debug.println(out[0])
        misc.debug.println(torch.argmax(out[0]))
        misc.debug.println("")
        return out
    
    def iter_forward(self, X):
        X = self.conv(X)
        it = list(self.resnet.iter_forward(X))
        assert len(it) == len(self.distills)
        for distill, X in zip(self.distills, it):
            out = distill(X)
            misc.debug.println(out[0])
            yield out
        
            
    def do_consensus(self, X):
        it = self.iter_forward(X)
        if self.useconsensus:
            return self._do_consensus(it)
        else:
            return it
    
    def _do_consensus(self, it):
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
