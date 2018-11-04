import torch

from .Norm import Norm

class SoftminNorm(Norm):

    def __init__(self, *args, **kwargs):
        super(SoftminNorm, self).__init__(*args, **kwargs)
        self.max = torch.nn.LogSoftmax(dim=1)

    def reduce(self, vectors, targets):
        N, C = self.get_NC()
        out = super(SoftminNorm, self).reduce(vectors, targets)
        return self.max(-out.view(N, C))
