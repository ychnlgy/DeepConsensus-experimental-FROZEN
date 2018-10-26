import torch

import models

class Model(torch.nn.Module):

    def __init__(self, channels, classes, lamb):
        super(Model, self).__init__()
        
        self.lamb = lamb
        
        self.cnn = torch.nn.Sequential(
        
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.DistillNet(
                
                # 28 -> 28
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 28 -> 14
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 14 -> 14
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 14 -> 7
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64),
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 7 -> 7
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 7 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.MaxPool2d(3, padding=1, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 4 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
                # 4 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            
                        ),
                        c = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 16,
                            layers = 1,
                        ),
                    )
                ),
                
            )
        )
        
        self.instance_separator = models.Norm(p=2)
        self.transform = torch.nn.Linear(128, 32)
        self.grouper = models.Classifier(32, classes)
        
        self.group_loss = torch.nn.CrossEntropyLoss()
    
    def calc_loss(self, X, y):
        latent_vecs, group = self._forward(X)
        norms = self.instance_separator(latent_vecs, latent_vecs)
        tloss = self.group_loss(group, y) - self.lamb*norms.mean()
        return group, tloss
    
    def forward(self, X):
        return self._forward(X)[1]
    
    def _forward(self, X):
        latent_vecs = self.cnn(X)
        trans = self.transform(latent_vecs)
        group = self.grouper(trans)
        return latent_vecs, group
