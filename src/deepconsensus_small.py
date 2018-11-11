import torch

import models, misc

from cnn_small import Cnn

class ModelCnn(models.Savable):

    def __init__(self, channels, classes, imagesize, **kwargs):
        super(ModelCnn, self).__init__()
        self.layers = Cnn.get_layers(channels, classes, imagesize)
        self.distills = torch.nn.ModuleList([
        
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 32,
                    layers = 1,
                    dropout = 0.2
                ),
                c = models.Classifier(
                    32,
                    classes + 1,
                    useprototype = 1,
                    usenorm = 0,
                    p = 2
                ),
            ),
        
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 64,
                    layers = 1,
                    dropout = 0.2
                ),
                c = models.Classifier(
                    64,
                    classes + 1,
                    useprototype = 1,
                    usenorm = 0,
                    p = 2
                ),
            ),
            
            models.GlobalSumPool(
                h = models.DenseNet(
                    headsize = 64,
                    layers = 1,
                    dropout = 0.2
                ),
                c = models.Classifier(
                    64,
                    classes + 1,
                    useprototype = 1,
                    usenorm = 0,
                    p = 2
                ),
            )
        ])
    
    def forward(self, X):
        return sum(self.iter_forward(X))
    
    def iter_forward(self, X):
        assert len(self.distills) == len(self.layers)
        for layer, distill in zip(self.layers, self.distills):
            X = layer(X)
            yield distill(X)
