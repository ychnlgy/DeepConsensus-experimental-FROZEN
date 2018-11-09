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
    
    def forward(self, X):
        self.layer_outputs = list(self.do_consensus(X))
        
#        if self.training:
        out = sum(self.layer_outputs)
#        else:
#            out = sum([t*p for t, p in zip(self.layerweights, self.layer_outputs)])
        return out
    
    def set_layerweights(self, weights):
        self.layerweights = weights*(1-self.alpha) + self.layerweights*self.alpha
    
    def clear_layereval(self):
        self.matches = torch.zeros(len(self.distills))
        self.n = 0
    
    def get_layereval(self):
        if self.n == 0:
            return self.matches
        else:
            return self.matches/self.n
    
    def eval_layers(self, y):
        matches = [self.match_argmax(yh, y) for yh in self.layer_outputs]
        self.matches += torch.Tensor(matches)
        self.n += 1
    
    def match_argmax(self, yh, y):
        return (torch.argmax(yh, dim=1) == y).float().mean().item()
    
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
