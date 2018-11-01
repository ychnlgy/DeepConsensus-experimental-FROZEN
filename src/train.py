#!/usr/bin/python3

import torch, tqdm, time, numpy, statistics

import misc, models, resnet

from distillnet import Model
from resnet import Model as Cnn

from deepfool import deepfool

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

def save_image(name, image):
    image = image.squeeze().cpu().detach().numpy()
    pyplot.imshow(image, cmap="gray")
    pyplot.savefig(name)
    pyplot.clf()

def collect_answer(model, image):
    im = image.view(1, 1, 64, 64)
    yh = torch.nn.functional.softmax(model(im), dim=1)
    val, idx = yh.squeeze().max(dim=0)
    print(idx.item(), val.item())

def main(modelf, dataset, epochs, fool=0, classic=0, trainbatch=100, testbatch=300, cycle=1, datalimit=1.0, device="cuda", silent=0, showparams=0, **dataset_kwargs):

    fool = int(fool)
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
        "mnist64": misc.data.get_mnist64,
        "mnist64-corrupt": misc.data.get_mnist64_corrupt,
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
    
    if fool:
        trainbatch = 1
        testbatch = 1
        model.load(modelf)
    
    dataloader, validloader, testloader = misc.data.create_trainvalid_split(0.2, datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch)
    
    if fool:
        model.eval()
        images = iter(testloader)
        
        perturb_amt = []
        for i in tqdm.tqdm(range(fool), desc="Fooling network", ncols=80):
            image, label = next(images)
            #save_image("%d-%d-original.png" % (i, label.item()), image)
            image = image.to(device).squeeze(0)
            r_tot, loop_i, label_fool, k_i, pert_image = deepfool(image, model, NUM_CLASSES)
            
            #save_image("%d-%d-perturb.png" % (i, k_i.item()), pert_image)
            
            #print(label.item(), loop_i)
            #collect_answer(model, image)
            #collect_answer(model, pert_image)
            
            perturb_amt.append(float(numpy.mean(numpy.abs(r_tot))))
        
        mean = statistics.mean(perturb_amt)
        stdd = statistics.stdev(perturb_amt)
        
        print("Pertubation norm1 mean: %.3f, standard deviation: %.3f" % (mean, stdd))
        
        raise SystemExit(0)
        
    lossf = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    highest = 0
      
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
            
#                misc.debug.ALLOW_PRINTING = i < 20
#                misc.debug.println("")
#                misc.debug.println(y[0])
            
                yh = model(X)
                v += lossf(yh, y).item()
                w += (torch.argmax(yh, dim=1) == y).float().mean().item()
                m += 1
            
            w /= m
            print_(" -- <VERR> %.3f" % w, silent)
            
            print("Saving to %s..." % modelf)
            model.save(modelf)
            
            scheduler.step(v/m)
            
            testscore = n = 0.0
            
            #input("Test begins")
            
            for i, X, y, bar in iter_dataloader(testloader, device, silent=True):
            
#                misc.debug.ALLOW_PRINTING = i < 20
#                misc.debug.println("")
#                misc.debug.println(y[0])
            
                yh = model(X)
                n += 1.0
                testscore += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            
            testscore /= n
            
            print_(" -- <TEST> %.3f" % testscore, silent)
            
            #input("Test ends")
            
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

@misc.main(__name__)
def _main(repeat=1, **kwargs):

    try:
        repeat = int(repeat)
        
        func = lambda silent: main(silent=silent, **kwargs)

        if repeat > 1:
            out = []
            bar = tqdm.tqdm(range(repeat), ncols=80)
            for i in bar:
                result = func(silent=True)
                out.append(result)
                bar.set_description("Score: %.3f" % result)
            print(out)
        else:
            func(silent=False)
    
    except KeyboardInterrupt:
        pass
