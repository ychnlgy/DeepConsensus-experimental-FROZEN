import torch

class DistillationNet(torch.nn.Module):
    def __init__(self, *blocks, tail=None, weight=0.01):
        super(DistillationNet, self).__init__()
        self.tail = tail
        self.blocks = torch.nn.ModuleList(blocks)
        self.lam = weight
    
    def forward(self, X):
        predictions = []
        for block in self.blocks:
            X, pred = block(X)
            predictions.append(pred)
        
        predictions = sum(predictions) # TODO: can add penalty for depth later.
        
        if self.training:
            return self.lam * self.tail(X) + predictions # CNN output matters.
        else:
            return predictions # only feature analysis matters.
