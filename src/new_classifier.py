import torch

import models

class Model(torch.nn.Module):

    def __init__(self, channels, classes, l1, l2, l3, l4):
        super(Model, self).__init__()
        
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        
        # === VAE ===
        
        self.downconv = torch.nn.Sequential(
        
            # 28 -> 28
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 14
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 1
            torch.nn.MaxPool2d(4),
            
            models.Reshape(64),
            
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU()
        )
        
        self.upconv = torch.nn.Sequential(
            
            models.Reshape(128, 1, 1),
            
            # 1 -> 7
            torch.nn.Upsample(scale_factor=7),
            
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 14
            torch.nn.Upsample(scale_factor=2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 28
            torch.nn.Upsample(scale_factor=2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, channels, 3, padding=1),
            torch.nn.Tanh()
        )
        
        # === MEANS ===
        
        self.means = models.Classifier(
            hiddensize = 128,
            classes = classes
        )
        
        # === CLASS-DEPENDENT VARIANCE ===
        
        self.vartrans = torch.nn.Linear(128, 1, bias=False)
        self.clsvar = models.Classifier(
            hiddensize = 128,
            classes = classes
        )
        
        # === CLASS-INDEPENDENT VARIANCE ===
        
        self.var = torch.nn.Linear(128, 128, bias=False)
    
    def calc_loss(self, X, y):
        
        latent_vec = self.downconv(X)
        #reconstruction = self.upconv(latent_vec)
        #loss1 = ((reconstruction - X)**2).mean()
        
        means = self.means.get_class_vec(y)
        clsvr = self.clsvar.get_class_vec(y)
        multp = self.vartrans(latent_vec)
        ivars = self.var(latent_vec)
        loss2 = (latent_vec - (means + clsvr*multp + ivars)).norm(dim=1).mean()
        
        loss3 = sum(
            [
                self.l1 * clsvr.norm(dim=1).mean(),
                self.l2 * multp.mean(),
                self.l3 * ivars.norm(dim=1).mean(),
                self.l4 * self.var.weight.norm()
            ]
        )
        
        return loss2 + loss3
    
    def forward(self, X):
        latent_vec = self.downconv(X)
        ivars = self.var(latent_vec)
        mean_apprx = latent_vec - ivars
        return self.means(mean_apprx)
