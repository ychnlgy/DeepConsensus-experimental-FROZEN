#!/usr/bin/python3

import torch, tqdm, time, numpy, statistics, random

import misc, models, resnet

import scipy.misc

from deepconsensus import Model, ModelCnn
from deepconsensus_small import ModelCnn as ModelCnnSmall
from deepconsensus_exposed import ModelCnn as ModelExposed
from resnet import Model as ResNet
from cnn import Cnn
from cnn_small import Cnn as CnnSmall
from groupy_models.mnist.mnist import Net as P4MCnn

from deepfool import deepfool

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

def save_image(name, image):
    image = image.squeeze().cpu().detach().numpy()
    scipy.misc.imsave(name, image)

def collect_answer(model, image):
    im = image.view(1, 1, 64, 64)
    yh = torch.nn.functional.softmax(model(im), dim=1)
    val, idx = yh.squeeze().max(dim=0)
    print(idx.item(), val.item())

MODEL_INIT = None

def main(
    modelf,
    dataset,
    epochs,
    sameinit=0,
    testset="",
    swaptest=0,
    swaptrain=0,
    combinedataset=0,
    combinetestset=0,
    alpha=0,
    useconsensus=0,
    layers=1,
    squash=0,
    usetanh=0,
    optout=1,
    useprototype=1,
    usenorm=0,
    normp=2,
    fool=0,
    foolname="",
    foolsamples=10,
    modelid=0,
    trainbatch=100,
    testbatch=100,
    cycle=5,
    datalimit=1.0,
    device="cuda",
    silent=0,
    showparams=0,
    usefake=0,
    collectconfidence=0,
    **kwargs):

    combinedataset = int(combinedataset)
    combinetestset = int(combinetestset)
    swaptrain = int(swaptrain)
    swaptest = int(swaptest)
    normp = float(normp)
    fool = int(fool)
    modelid = int(modelid)
    epochs = int(epochs)
    cycle = int(cycle)
    trainbatch = int(trainbatch)
    testbatch = int(testbatch)
    datalimit = float(datalimit)
    showparams = int(showparams)
    usefake = int(usefake)
    foolsamples = int(foolsamples)
    sameinit = int(sameinit)
    collectconfidence = int(collectconfidence)
    
    DATASETS = {
        "mnist": misc.data.get_mnist,
        "mnist-corrupt": misc.data.get_mnist_corrupt,
        "mnist64": misc.data.get_mnist64,
        "mnist64-corrupt": misc.data.get_mnist64_corrupt,
        "mnist64-quadrants": misc.data.get_mnist64quads,
        "mnist-rgb": misc.data.get_mnistrgb,
        "mnist-rgb-corrupt": misc.data.get_mnistrgb_corrupt,
        "mnist-rot": misc.data.get_mnist_rot,
        "fashion": misc.data.get_fashionmnist,
        "fashion64-corrupt": misc.data.get_fashionmnist64_corrupt,
        "svhn": misc.data.get_svhn,
        "svhn-corrupt": misc.data.get_svhn_corrupt,
        "stl10": misc.data.get_stl10,
        "stl9-64stretch": misc.data.get_stl9_64stretch,
        "cifar10": misc.data.get_cifar10,
        "cifar10-corrupt": misc.data.get_cifar10_corrupt,
        "cifar1064": misc.data.get_cifar1064,
        "cifar1064-corrupt": misc.data.get_cifar1064_corrupt,
        "cifar9-64stretch": misc.data.get_cifar9_64stretch,
        "emnist": misc.data.get_emnist,
        "emnist-corrupt": misc.data.get_emnist_corrupt,
        "emnist64-corrupt": misc.data.get_emnist64_corrupt,
        "cs_trans": misc.data.get_circlesqr_translate,
        "cs_magnify": misc.data.get_circlesqr_magnify,
        "cs_shrink": misc.data.get_circlesqr_shrink,
        "sqrquad": misc.data.get_sqrquadrants,
    }
    
    train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE = DATASETS[dataset](**kwargs)
    
    if swaptrain:
        train_dat, test_dat, train_lab, test_lab = test_dat, train_dat, test_lab, train_lab
    
    if testset:
    
        if combinedataset:
            train_dat = torch.cat([train_dat, test_dat], dim=0)
            train_lab = torch.cat([train_lab, test_lab], dim=0)
    
        _train_dat, _train_lab, test_dat, test_lab, _classes, _channels, _imagesize = DATASETS[testset](**kwargs)
        assert _classes == NUM_CLASSES
        assert _channels == CHANNELS
        assert _imagesize == IMAGESIZE
        
        if combinetestset:
            test_dat = torch.cat([_train_dat, test_dat], dim=0)
            test_lab = torch.cat([_train_lab, test_lab], dim=0)
        elif swaptest:
            test_dat = _train_dat
            test_lab = _train_lab
    
    model = [Model, ResNet, ModelCnn, Cnn, P4MCnn, ModelCnnSmall, CnnSmall, ModelExposed][modelid](CHANNELS, NUM_CLASSES, IMAGESIZE, 
        useconsensus = useconsensus,
        layers = layers,
        squash = squash,
        usetanh = usetanh,
        optout = optout,
        useprototype = useprototype,
        usenorm = usenorm,
        p = normp,
        alpha = alpha
    )
    
    if sameinit:
        global MODEL_INIT
        if MODEL_INIT is None:
            MODEL_INIT = model.state_dict()
        else:
            model.load_state_dict(MODEL_INIT)
    
    print_("Using %s" % str(model.__class__), silent)
    
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
    
    if collectconfidence:
        
        model.eval()
        model.load(modelf)
        
        testscores = []
        confs = []
        
        with torch.no_grad():
            for X, y in testloader:
                X = X.to(device)
                y = y.to(device)
                
                yh = model(X)
                choices = torch.argmax(yh, dim=1).squeeze()
                confidences = torch.nn.functional.softmax(yh, dim=1).squeeze()
                confs.extend([c[i] for i, c in zip(choices, confidences)])
                testscores.append((choices == y).float().mean().item())
            
            print("Score: %f, std: %f" % (statistics.mean(testscores), statistics.stdev(testscores)))
            print("Confidence: %f, std: %f" % (statistics.mean(confs), statistics.stdev(confs)))
        
        raise SystemExit
    
    if fool:
        model.eval()
        images = iter(testloader)
        
        perturb_amt = []
        saved = []
        
        def get_score(im):
            pred = model(im).squeeze()
            choice = torch.argmax(pred).item()
            confid = torch.nn.functional.softmax(pred, dim=0)[choice].item()
            return "%d.%.3f" % (choice, confid)
        
        for i in tqdm.tqdm(range(fool), desc="Fooling network", ncols=80, disable=silent):
            image, label = next(images)
            #save_image("%d-%d-original.png" % (i, label.item()), image)
            image = image.to(device).squeeze(0)
            r_tot, loop_i, label_fool, k_i, pert_image = deepfool(image, model, NUM_CLASSES)
            
            im = image.unsqueeze(0)
            choice = get_score(im)
            pertch = get_score(pert_image.view(im.size()))
            saved.append((image, pert_image, choice, pertch))
            
            #save_image("%d-%d-perturb.png" % (i, k_i.item()), pert_image)
            
            #print(label.item(), loop_i)
            #collect_answer(model, image)
            #collect_answer(model, pert_image)
            r_tot = torch.from_numpy(r_tot).to(device)
            perturb_amt.append(float(r_tot.norm(p=2)/image.norm(p=2)))
        
        mean = statistics.mean(perturb_amt)
        stdd = statistics.stdev(perturb_amt)
        
        print("Pertubation density: %.3f, standard deviation: %.3f" % (mean, stdd))
        
        for i in range(foolsamples):
            image, pert_image, choice, pertch = saved[i]
            image = image.permute(1, 2, 0)
            pert_image = pert_image.squeeze(0).permute(1, 2, 0)
            
            save_image("im%d.%s.%s-original.png" % (i, choice, foolname), image)
            save_image("im%d.%s.%s-perturbed.png" % (i, pertch, foolname), pert_image)
        
        raise SystemExit(0)
        
    lossf = torch.nn.CrossEntropyLoss().to(device)#torch.nn.MultiMarginLoss().to(device) # 
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    
    highest = 0
    
    FAKE = torch.zeros(1, CHANNELS, *IMAGESIZE).to(device)
    
    if usefake:
        get_fake = lambda X: FAKE.repeat(len(X), 1, 1, 1) + random.randint(0, 1)
    else:
        get_fake = lambda X: X
    
    for epoch in iterepochs(epochs):
        
        c = s = n = 0.0
        
        model.train()
        for i, X, y, bar in iter_dataloader(dataloader, device, silent):
            
            X = get_fake(X)
        
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
                X = get_fake(X)
                
                yh = model(X)
                v += lossf(yh, y).item()
                w += (torch.argmax(yh, dim=1) == y).float().mean().item()
                m += 1
            
            w /= m
            print_(" -- <VERR> %.3f" % w, silent)
            
            print_("Saving to %s..." % modelf, silent)
            model.save(modelf)
            
            scheduler.step(v/m)
            
            testscore = n = 0.0
            
            #input("Test begins")
            bar = iter_dataloader(testloader, device, silent=True)
            
#            i, X, y, _ = next(bar)
##            
#            #with torch.enable_grad():
#            yh = model(X)
#            
#            if type(model) is Model:
#                model.eval_layers(y)
#                weights = model.get_layereval().to(device)
#                model.clear_layereval()
#                model.set_layerweights(weights)
            
#            else:
#                optimizer.zero_grad()
#                lossf(yh, y).backward() # let the resnet learn from this.
#                optimizer.step()
            
            for i, X, y, _ in bar:
            
#                misc.debug.ALLOW_PRINTING = i < 5
#                misc.debug.println("")
#                misc.debug.println(y[0])
                X = get_fake(X)
            
                yh = model(X)
                
                if type(model) is ModelExposed:
                    model.eval_layers(y)
                
                n += 1.0
                testscore += (torch.argmax(yh, dim=1) == y).float().mean().item()
            
            
            testscore /= n
            
            print_(" -- <TEST> %.3f" % testscore, silent)
            
            if type(model) is ModelExposed:
                print_(model.get_layereval(), silent)
                model.clear_layereval()
            
#            misc.debug.ALLOW_PRINTING = True
#            
#            misc.debug.println(model(FAKE)[0])
#            misc.debug.println(model(FAKE + 1)[0])
#            
#            misc.debug.ALLOW_PRINTING = False
            
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
