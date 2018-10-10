import torch

class AttentionPool(torch.nn.Module):
    
    def __init__(self, net):
        super(AttentionPool, self).__init__()
        self.net = net
        self.max = torch.nn.Softmax(dim=1)
    
    def forward(self, X):
        N, C, W, H = X.size()
        X = X.view(N, C, W*H)
        X = X.transpose(1, 2) # N, W*H, C
        weights = self.max(self.net(X))
        assert weights.size() == (N, W*H, 1)
        weighted = weights * X
        out = weighted.sum(dim=1).squeeze(1)
        assert out.size() == (N, C)
        weighted = weighted.transpose(1, 2).view(N, C, W, H)
        return out, weighted
