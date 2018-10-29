class Cnn(models.Savable):

    def __init__(self, channels, classes):
        super(Cnn, self).__init__()
        self.net = torch.nn.Sequential(
        
            # 64 -> 64
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 64 -> 32
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 32 -> 16
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 16 -> 8
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 8 -> 4
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 1
            
            torch.nn.AvgPool2d(4),
            
            models.Reshape(64),

            torch.nn.Linear(64, classes)

        )
    
    def forward(self, X):
        return self.net(X)

class Model(models.Savable):

    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # === Bottleneck ===
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 32 -> 32
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 32 -> 16
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 16 -> 32
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 32 -> 32
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # === Convolutions ===
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            models.DistillNet(
                
                # 32 -> 32
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 32 -> 32
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 32 -> 32
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 32 -> 16
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 16 -> 16
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 16 -> 8
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 8 -> 8
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                # 8 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
                 # 4 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.DistillPool(
                        channels = 32,
                        h = models.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            
                        ),
                        c = models.Classifier(
                            hiddensize = 32,
                            classes = classes
                        ),
                    )
                ),
                
            )
        )
    
    def forward(self, X):
        return self.net(X)
