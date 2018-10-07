#!/usr/bin/python3

import torch, tqdm, time, numpy

import misc

from Models import DistillationNetwork28or32, Cnn28or32

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

def select_netsize(imagesize, DistillNets, Cnns):
    return [
        {imsz:N for N in Nets for imsz in N.expected_imagesizes()}[imagesize]
        for Nets in [DistillNets, Cnns]
    ]

def _main(dataset, trainbatch, testbatch, cycle, classic, paramid, datalimit=1.0, rest=0, epochs=-1, device="cuda", silent=0, **dataset_kwargs):
    
    epochs = int(epochs)
    cycle = int(cycle)
    trainbatch = int(trainbatch)
    testbatch = int(testbatch)
    rest = float(rest)
    classic = int(classic)
    datalimit = float(datalimit)
    
    train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE = {
        "mnist": misc.data.get_mnist,
        "mnist-corrupt": misc.data.get_mnist_corrupt,
        "cifar10": misc.data.get_cifar10,
        "cifar10-corrupt": misc.data.get_cifar10_corrupt,
        "cs_trans": misc.data.get_circlesqr_translate,
        "cs_magnify": misc.data.get_circlesqr_magnify,
        "cs_shrink": misc.data.get_circlesqr_shrink,
    }[dataset](**dataset_kwargs)
    
    model = select_netsize(
        IMAGESIZE,
        [DistillationNetwork28or32],
        [Cnn28or32]
    )[classic](paramid, NUM_CLASSES, CHANNELS)
    
    print_("Model parameters: %d" % model.paramcount(), silent)
    
    if not silent:
        if input("Continue? [y/n] ") != "y":
            raise SystemExit
    
    model = model.to(device)
    dataloader, validloader, testloader = misc.data.create_trainvalid_split(datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch)
    
    lossf = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    lowest = float("inf")
    
    for epoch in iterepochs(epochs):
        
        model.train()
        
        c = s = n = 0.0
        
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
                bar.set_description("[E %d] %.3f" % (epoch, s/n))
        
        model.eval()
        
        with torch.no_grad():
            
            c = s = n = 0.0
            
            for i, X, y, bar in iter_dataloader(validloader, device, silent):
                
                yh = model(X)
                loss = lossf(yh, y)
                c += loss.item()
                n += 1.0
                s += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            c /= n
            s /= n
            
            scheduler.step(c)
            
            print_("<SCORE> %.5f" % s, silent)
            
            testscore = n = 0.0
            
            for i, X, y, bar in iter_dataloader(testloader, device, silent):
                
                yh = model(X)
                n += 1.0
                testscore += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            testscore /= n
            
            print_("<TEST > %.5f" % testscore, silent)
    
        time.sleep(rest)
    
    return testscore

@misc.main
def main(repeat=1, **kwargs):

    try:
        repeat = int(repeat)
        
        func = lambda silent: _main(silent=silent, **kwargs)

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
