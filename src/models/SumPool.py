import torch

class SumPool(torch.nn.Module):
    
    def __init__(self, paramsize, net, threshold):
        super(SumPool, self).__init__()
        self.thd = threshold
        self.prm = torch.nn.Parameter(torch.rand(paramsize))
        self.net = net
        self.sig = torch.nn.Sigmoid()
        self.max = torch.nn.Softmax(dim=-1)
    
    def forward(self, X):
        N, C, W, H = X.size()
    
        weights = self.max(self.net(self.prm))
        weights = weights.view(1, -1, 1, 1)
        X = self.sig(X) * weights
        
        if not self.training:
            X[X < self.thd] = 0
            X[X >=self.thd] = 1
        
        X = X.view(N, C, W*H).sum(dim=-1)
        assert X.size() == (N, C)
        return X
