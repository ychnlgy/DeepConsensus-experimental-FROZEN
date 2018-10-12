import torch

import misc

class DistributionTracker(torch.nn.Module):
    
    def __init__(self, classes):
        super(DistributionTracker, self).__init__()
        self.c = classes
        
    def forward(self, X, labels):
        
        # TODO: Need to allow for labels of shape (N, classes) - multilabels
        
        '''
        
        Given:
            X - Tensor of shape (N, C), features for each batch-item
            labels - Tensor of shape (N), labels for each batch-item
        
        Description:
            Stores information about the labelled data X for
            calculating the mean and standard deviation for each label.
        
        '''
        self.setup(X)
        
        self.count(labels)
        self.add(labels, X)
        self.sqr(labels, X)
    
    def count(self, labels):
        ones = torch.ones(len(labels)).to(labels.device)
        self.num.put_(labels, ones, accumulate=True)
    
    def add(self, labels, X):
        self.miu.put_(self.ind[labels], X, accumulate=True)
    
    def sqr(self, labels, X):
        self.std.put_(self.ind[labels], X**2, accumulate=True)
    
    def setup(self, X):
        N, D = X.size()
        self.num = torch.zeros(self.c).view(-1, 1).to(X.device)
        self.miu = torch.zeros(self.c, D).to(X.device)
        self.std = torch.zeros(self.c, D).to(X.device)
        
        ind = torch.arange(self.c).view(-1, 1).repeat(1, D)*D + torch.arange(0, D).view(1, D)
        self.ind = ind.long().to(X.device)
        
        self.switch_setup()
    
    def switch_setup(self):
        self.setup, self._setup = self._setup, self.setup

    def _setup(self, X):
        return
        
    def correct_num(self, num):
        num = num.clone()
        num[num <= 0] = 1
        return num
    
    def stats(self):
        
        '''
        
        Returns:
            local_miu  - Tensor of shape (classes, C)
            local_std  - Tensor of shape (classes, C)
            global_miu - Tensor of shape (C)
            global_std - Tensor of shape (C)
        
        Description:
            miu is calculated by sum(X)/num(X)
            std is calculated by sqrt((sum(X^2)-sum(X)^2/n)/n)
        
        '''
        local_miu = self.calc_mean(self.miu, self.num)
        local_std = self.calc_std( self.std, self.num, local_miu)
        
        total = self.num.sum()
        global_miu = self.calc_mean(self.miu.sum(dim=0), total)
        global_std = self.calc_std( self.std.sum(dim=0), total, global_miu)

        self.switch_setup()
        return local_miu, local_std, global_miu, global_std
    
    def calc_mean(self, miu, num):
        return miu / self.correct_num(num)
    
    def calc_std(self, std, num, miu):
        return torch.sqrt((std - num*miu**2) / self.correct_num(num-1))
    
    @staticmethod
    def unittest():
    
        import math
    
        EPS = 1e-6
    
        def same(m1, m2):
            return (m1 - torch.Tensor(m2)).norm() < EPS
    
        tracker = DistributionTracker(3)
        
        batch1 = torch.Tensor([
            [2, 10],
            [3, 12],
            [2, 20],
            [4, 1],
            [0, 6]
        ])
        
        label1 = torch.LongTensor([
            0,
            2,
            2,
            2,
            0
        ])
        
        tracker(batch1, label1)
        
        miu, std, gmiu, gstd = tracker.stats()
        
        assert same(miu, [
            [1, 8],
            [0, 0],
            [3, 11]
        ])
        
        assert same(std, [
            [1.4142135623730951, 2.8284271247461903],
            [0, 0],
            [1, 9.539392014169456]
        ])
        
        assert same(gmiu, [11.0/5.0, 49.0/5.0])
        assert same(gstd, [1.4832396974191326, 7.085195833567341])
        
        batch2 = torch.Tensor([
            [4, 100],
            [7, 19],
            [9, 20],
            [5, 18],
            [6, 84]
        ])
        
        tracker(batch1, label1)
        tracker(batch2, label1)
        
        miu, std, gmiu, gstd = tracker.stats()
        
        assert same(miu, [
            [3, 50],
            [0, 0],
            [5, 15]
        ])
        
        assert same(std, [
            [2.581988897471611, 48.96257073860944],
            [0, 0],
            [2.6076809620810595, 7.483314773547883]
        ])
        
        assert same(gmiu, [4.2, 29])
        assert same(gstd, [2.6583202716502514, 34.013069383530926])
