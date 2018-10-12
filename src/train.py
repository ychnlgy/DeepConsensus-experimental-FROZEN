#!/usr/bin/python3

import torch, tqdm, time, numpy

import misc, models

class Model(models.Savable):
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            # 28 -> 14
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 7 -> 4
            torch.nn.Conv2d(32, 16, 3, padding=1, groups=16),
            torch.nn.AvgPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 4 -> 1
            torch.nn.Conv2d(16, 8, 3, padding=1, groups=8),
            torch.nn.AvgPool2d(4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            
            models.Reshape(8),
            models.DenseNet(
                headsize = 8,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            )
        )
    
    def forward(self, X):
        return self.net(X)

def main(dataset, split=0.9, trainbatch=100, testbatch=100, cycle=10, datalimit=1.0, rest=0, epochs=-1, device="cuda", silent=0, showparams=0, **dataset_kwargs):
    
    split = float(split)
    epochs = int(epochs)
    cycle = int(cycle)
    trainbatch = int(trainbatch)
    testbatch = int(testbatch)
    rest = float(rest)
    datalimit = float(datalimit)
    showparams = int(showparams)
    
    train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE = {
        "mnist": misc.data.get_mnist,
        "mnist-corrupt": misc.data.get_mnist_corrupt,
        "cifar10": misc.data.get_cifar10,
        "cifar10-corrupt": misc.data.get_cifar10_corrupt,
        "cs_trans": misc.data.get_circlesqr_translate,
        "cs_magnify": misc.data.get_circlesqr_magnify,
        "cs_shrink": misc.data.get_circlesqr_shrink,
    }[dataset](**dataset_kwargs)
    
    model = Model(CHANNELS, NUM_CLASSES)
    discr = models.Discriminator(model, models.DenseNet(
        headsize = model.paramcount(),
        bodysize = 1024,
        tailsize = 1,
        layers = 2,
        dropout = 0.2,
        bias = True
    ))
    
    if showparams:
    
        print_(" === PARAMETERS === ", silent)
        print_("Model        : %d" % model.paramcount(), silent)
        print_("Discriminator: %d" % discr.paramcount(), silent)
    
        if input("Continue? [y/n] ") != "y":
            raise SystemExit
    
    model = model.to(device)
    dataloader, validloader, testloader = misc.data.create_trainvalid_split(split, datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch)
    
    iter_validloader = infinite(iter_dataloader(validloader, device, silent=True))
    
    lossf = torch.nn.CrossEntropyLoss().to(device)
    lossd = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    discoptim = torch.optim.Adam(discr.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    lowest = float("inf")
    
    PRETRAIN = 3
    
    for epoch in range(PRETRAIN):
        
        c = s = n = 0.0
        v = w = m = 0.0
        
        for i, X, y, bar in iter_dataloader(dataloader, device, silent):
            
            # Update the model
            
            model.train()
            
            yh = model(X)
            loss = lossf(yh, y)
            
            c += loss.item()
            n += 1.0
            s += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the discriminator
            
            model.eval()
            
            j, Xv, yv, barv = next(iter_validloader)
            yh = model(Xv)
            loss2 = lossf(yh, yv)
            v += loss2.item()
            m += 1.0
            w += (torch.argmax(yh, dim=1) == yv).float().mean().item()
            
            loss3 = lossd(loss2, loss)
            discoptim.zero_grad()
            loss3.backward()
            discoptim.step()
            
            if i % cycle == 0:
                bar.set_description("[Pretraining %d/%d] %.3f (%.3f validation)" % (epoch+1, PRETRAIN, s/n, w/m))
        
    for epoch in iterepochs(epochs):
        
        c = s = n = 0.0
        v = w = m = 0.0
        
        for i, X, y, bar in iter_dataloader(dataloader, device, silent):
            
            # Update the model
            
            model.train()
            
            yh = model(X)
            loss1 = lossf(yh, y)
            
            c += loss.item()
            n += 1.0
            s += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            optimizer.zero_grad()
            (loss1 + discr()).backward() # NOTE: new loss
            optimizer.step()
            
            # Update the discriminator
            
            model.eval()
            
            j, Xv, yv, barv = next(iter_validloader)
            yh = model(Xv)
            loss2 = lossf(yh, yv)
            v += loss2.item()
            m += 1.0
            w += (torch.argmax(yh, dim=1) == yv).float().mean().item()
            
            loss3 = lossd(loss2, loss1)
            discoptim.zero_grad()
            loss3.backward()
            discoptim.step()
            
            if i % cycle == 0:
                bar.set_description("[Epoch %d] %.3f (%.3f validation)" % (epoch, s/n, w/m))
        
        scheduler.step(w/m)
        
        with torch.no_grad():
            
            testscore = n = 0.0
            
            for i, X, y, bar in iter_dataloader(testloader, device, silent):
                
                yh = model(X)
                n += 1.0
                testscore += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            testscore /= n
            
            print_("<TEST > %.5f" % testscore, silent)
    
        time.sleep(rest)
    
    return testscore

def infinite(iterator):
    while True:
        for i in iterator:
            yield i

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
                bar.set_description("Score: %.3f" % result)
                out.append(result)
            print(out)
        else:
            func(silent=False)
    
    except KeyboardInterrupt:
        pass
