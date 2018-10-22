import torch

class ChannelClassifier(torch.nn.Module):

    def __init__(self, observer, net, classifier):
        super(ChannelClassifier, self).__init__()
        self.obs = observer
        self.net = net
        self.cls = classifier
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, C', W, H)
        
        '''
    
        N, C, W, H = X.size()
        obs = self.obs(X)
        obs = obs.permute(0, 2, 3, 1).contiguous().view(-1, 9) # N*W*H, C
        lat = self.net(obs) # N*W*H, C'
        cls = self.cls(lat) # N*W*H, C''
        Np, Cp = cls.size()
        return cls.view(N, W, H, Cp).permute(0, 3, 1, 2)
