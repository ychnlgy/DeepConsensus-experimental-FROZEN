#!/usr/bin/python3

import torch, tqdm, time, numpy, statistics

import misc, models

class ReverseDistill(torch.nn.Module):

    def __init__(self, channels, classes):
        super(ReverseDistill, self).__init__()
        
        inputsize = 32 + 2*16 + 2*8 + 3*4
        
        self.net = torch.nn.Sequential(
            
            torch.nn.Linear(inputsize, 196),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            
            models.Reshape(4, 7, 7),
            
            torch.nn.Conv2d(4, 64, 3, padding=1),
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
            
            torch.nn.Conv2d(64, channels, 3, padding=1)
        )
    
    def forward(self, distilled):
        return self.net(distilled)

class Cnn(models.Savable):

    def __init__(self, channels, classes):
        super(Cnn, self).__init__()
        self.net = torch.nn.Sequential(
        
            # 28 -> 28
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 28
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 28
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 14
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 14
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 7
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.AvgPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 4
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 1
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.AvgPool2d(4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.Reshape(64),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )
    
    def forward(self, X):
        return self.net(X)

class Model(models.Savable):

    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            # === Convolutions ===
            
            torch.nn.Sequential(
                torch.nn.Conv2d(channels, 64, 3, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(64)
            ),
            
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(64)
            ),
            
            models.DistillNet(
                
                # 28 -> 28
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 128,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 128,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            dropout = 0.2
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
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 16,
                            tailsize = 16,
                            layers = 1,
                            dropout = 0.2
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
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 16,
                            tailsize = 16,
                            layers = 1,
                            dropout = 0.2
                        ),
                    )
                ),
                
                # 14 -> 7
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 8,
                            tailsize = 8,
                            layers = 1,
                            dropout = 0.2
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
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 8,
                            tailsize = 8,
                            layers = 1,
                            dropout = 0.2
                        ),
                    )
                ),
                
                # 7 -> 4
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 1,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 4,
                            tailsize = 4,
                            layers = 1,
                            dropout = 0.2
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
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 1,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 4,
                            tailsize = 4,
                            layers = 1,
                            dropout = 0.2
                        ),
                    )
                ),
                
                # 4 -> 2
                models.DistillLayer(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1),
                        torch.nn.AvgPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.DistillPool(
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 1,
                            dropout = 0.2,
                            #activation =  torch.nn.Sigmoid()
                        ),
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 1,
                            dropout = 0.2
                        ),
                        s = models.DenseNet(
                            headsize = 64,
                            bodysize = 4,
                            tailsize = 4,
                            layers = 1,
                            dropout = 0.2
                        ),
                    )
                ),
                
            ),
            
            # === Classification ===
            
            models.Classifier(32 + 2*16 + 2*8 + 3*4, classes)
            
            
            
        )
    
    def forward(self, X):
        return self.net(X)

def main(dataset, classic=0, trainbatch=100, testbatch=300, cycle=10, datalimit=1.0, epochs=-1, device="cuda", silent=0, showparams=0, **dataset_kwargs):

    classic = int(classic)
    epochs = int(epochs)
    cycle = int(cycle)
    trainbatch = int(trainbatch)
    testbatch = int(testbatch)
    datalimit = float(datalimit)
    showparams = int(showparams)
    
    train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE = {
        "mnist": misc.data.get_mnist,
        "mnist-corrupt": misc.data.get_mnist_corrupt,
        "cifar10": misc.data.get_cifar10,
        "cifar10-corrupt": misc.data.get_cifar10_corrupt,
        "emnist": misc.data.get_emnist,
        "emnist-corrupt": misc.data.get_emnist_corrupt,
        "cs_trans": misc.data.get_circlesqr_translate,
        "cs_magnify": misc.data.get_circlesqr_magnify,
        "cs_shrink": misc.data.get_circlesqr_shrink,
        "sqrquad": misc.data.get_sqrquadrants,
    }[dataset](**dataset_kwargs)
    
    model = [Model, Cnn][classic](CHANNELS, NUM_CLASSES)
    
    if showparams:
    
        print_("Model parameters: %d" % model.paramcount(), silent)
    
        if input("Continue? [y/n] ") != "y":
            raise SystemExit
    
    model = model.to(device)
    dataloader, validloader, testloader = misc.data.create_trainvalid_split(0.2, datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch)
    
    lossf = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    lowest = float("inf")
      
    for epoch in iterepochs(epochs):
        
        c = s = n = 0.0
        
        model.train()
        for i, X, y, bar in iter_dataloader(dataloader, device, silent):
            
            yh = model(X)
            loss = lossf(yh, y)
            
            c += loss.item()
            n += 1.0
            s += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % cycle == 0:
                bar.set_description("[Epoch %d] %.3f" % (epoch, s/n))
        
        model.eval()
        
        with torch.no_grad():
            
            v = w = m = 0.0
            
            for i, X, y, bar in iter_dataloader(validloader, device, silent=True):
                
                yh = model(X)
                v += lossf(yh, y).item()
                w += (torch.argmax(yh, dim=1) == y).float().mean().item()
                m += 1
            
            print_(" -- <VERR> %.3f" % (w/m), silent)
            
            scheduler.step(v/m)
            
            testscore = n = 0.0
            
            for i, X, y, bar in iter_dataloader(testloader, device, silent=True):
                
                yh = model(X)
                n += 1.0
                testscore += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            testscore /= n
            
            print_(" -- <TEST> %.3f" % testscore, silent)
    
    return testscore

def print_(s, silent):
    if not silent:
        print(s)

def iterepochs(end):
    i = 0
    while i != end:
        yield i
        i += 1

def iter_dataloader(dataloader, device, silent):
    bar = tqdm.tqdm(dataloader, ncols=80, disable=silent)
    for i, (X, y) in enumerate(bar):
        yield i, X.to(device), y.to(device), bar

@misc.main
def _main(repeat=1, **kwargs):

    try:
        repeat = int(repeat)
        
        func = lambda silent: main(silent=silent, **kwargs)

        if repeat > 1:
            out = []
            bar = tqdm.tqdm(range(repeat), ncols=80)
            for i in bar:
                result = func(silent=True)
                bar.set_description("Score: %.3f, Stdev: %.3f" % (statistics.mean(out), statistics.stdev(out)))
                out.append(result)
            print(out)
        else:
            func(silent=False)
    
    except KeyboardInterrupt:
        pass
