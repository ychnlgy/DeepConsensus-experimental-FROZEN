#!/usr/bin/python3

import torch, tqdm, time, numpy

import misc, models

class Model(models.Savable):
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            # 28 -> 14
            torch.nn.Conv2d(channels, 128, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 14 -> 7
            torch.nn.Conv2d(128, 64, 3, padding=1, groups=2),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.Conv2d(64, 32, 3, padding=1, groups=2),
            torch.nn.AvgPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 1
            torch.nn.Conv2d(32, 16, 3, padding=1, groups=1),
            torch.nn.AvgPool2d(4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            models.Reshape(16),
            models.DenseNet(
                headsize = 16,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            )
        )
    
    def forward(self, X):
        return self.net(X)

def discriminator_loss(loss1, loss2, discr):
    dscr = discr()
    #print(dscr, (loss1 - loss2).abs())
    return (loss2 - dscr).abs()

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
    discr = models.Discriminator(model, torch.nn.Sequential(
    
        torch.nn.Conv1d(CHANNELS, 64, 5, stride=2, padding=2),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(64),
        
        torch.nn.Conv1d(64, 128, 5, stride=2, padding=2, groups=64),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(128),
        
        torch.nn.Conv1d(128, 64, 5, stride=2, padding=2, groups=64),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(64),
        
        torch.nn.Conv1d(64, 32, 5, stride=2, padding=2, groups=32),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(32),
        
        torch.nn.Conv1d(32, 16, 5, stride=2, padding=2, groups=16),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(16),
        
        torch.nn.Conv1d(16, 8, 5, stride=2, padding=2, groups=8),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(8),
        
        torch.nn.Conv1d(8, 1, 5, stride=2, padding=2, groups=1),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(1),
        
        models.Reshape(425),
        
        models.DenseNet(
            headsize = 425,
            bodysize = 1024,
            tailsize = 1,
            layers = 2,
            dropout = 0.2,
            bias = True
        ),
    ))
    
    if showparams:
    
        print_(" === PARAMETERS === ", silent)
        print_("Model        : %d" % model.paramcount(), silent)
        print_("Discriminator: %d" % discr.paramcount(), silent)
    
        if input("Continue? [y/n] ") != "y":
            raise SystemExit
    
    model = model.to(device)
    discr = discr.to(device)
    dataloader, validloader, testloader = misc.data.create_trainvalid_split(split, datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch)
    
    iter_validloader = infinite(iter_dataloader, validloader, device, silent=True)
    
    lossf = torch.nn.CrossEntropyLoss().to(device)
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
            loss1 = lossf(yh, y)
            
            c += loss1.item()
            n += 1.0
            s += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer.step()
            
            # Update the discriminator
            
            model.eval()
            
            j, Xv, yv, barv = next(iter_validloader)
            yh = model(Xv)
            loss2 = lossf(yh, yv)
            v += loss2.item()
            m += 1.0
            w += (torch.argmax(yh, dim=1) == yv).float().mean().item()
            
            loss3 = discriminator_loss(loss1, loss2, discr)
            discoptim.zero_grad()
            loss3.backward()
            discoptim.step()
            
            if i % cycle == 0:
                bar.set_description("[Pretrain %d/%d] %.3f (%.3f verr)" % (epoch+1, PRETRAIN, s/n, w/m))
        
    for epoch in iterepochs(epochs):
        
        c = s = n = 0.0
        v = w = m = 0.0
        dloss = 0.0
        
        for i, X, y, bar in iter_dataloader(dataloader, device, silent):
            
            for p in range(5):
                # Update the model
                model.train()
                
                yh = model(X)
                loss1 = lossf(yh, y)
                
                c += loss1.item()
                n += 1.0
                s += (torch.argmax(yh, dim=1) == y).float().mean().item()
                
                optimizer.zero_grad()
                (loss1 + 0.001*discr()).backward(retain_graph=True) # NOTE: new loss
                optimizer.step()
            
            # Update the discriminator
            
            model.eval()
            
            j, Xv, yv, barv = next(iter_validloader)
            yh = model(Xv)
            loss2 = lossf(yh, yv)
            v += loss2.item()
            w += (torch.argmax(yh, dim=1) == yv).float().mean().item()
            loss3 = discriminator_loss(loss1, loss2, discr)
            dloss += loss3.item()
            discoptim.zero_grad()
            loss3.backward()
            discoptim.step()
            if i % cycle == 0:
                bar.set_description("[Epoch %d] %.3f (%.3f verr, %.3f dloss)" % (epoch, s/n, w/n, dloss/n))
        
        scheduler.step(w/n)
        
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

def infinite(fn, *args, **kwargs):
    while True:
        for i in fn(*args, **kwargs):
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
    if silent:
        bar = dataloader
    else:
        bar = tqdm.tqdm(dataloader, ncols=80)
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
