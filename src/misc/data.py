import tqdm, numpy, random, torch, torchvision, matplotlib, os

from .geometry import generate_circle, generate_square
from .corrupt import corrupt
from .wscipy import scipy
import scipy.misc # because spock2 doesn't have skimage

import torch.utils.data
import torchvision.datasets

DIR = os.path.dirname(__file__)
ROOT = os.path.join(DIR, "..", "..", "data")

def create_trainvalid_split(p, datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch):
    assert 0 <= datalimit <= 1
    n = int(len(train_dat) * datalimit)
    indices = numpy.arange(n)
    numpy.random.shuffle(indices)
    split = int(round(p*n))
    trainidx = torch.from_numpy(indices[split:n])
    valididx = torch.from_numpy(indices[:split])
    
    dataset = torch.utils.data.TensorDataset(train_dat[trainidx], train_lab[trainidx])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=trainbatch, shuffle=True)
    
    validset = torch.utils.data.TensorDataset(train_dat[valididx], train_lab[valididx])
    validloader = torch.utils.data.DataLoader(validset, batch_size=testbatch, shuffle=True)
    
    testset = torch.utils.data.TensorDataset(test_dat, test_lab)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testbatch, shuffle=True)
    return dataloader, validloader, testloader

def get_mnist(download=0):
    
    download = int(download)
    
    NUM_CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = (28, 28)
    
    train = torchvision.datasets.MNIST(root=ROOT, train=True, download=download)
    trainData = train.train_data.view(-1, 1, *IMAGESIZE).float()/255.0
    trainLabels = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.MNIST(root=ROOT, train=False, download=download)
    testData = test.test_data.view(-1, 1, *IMAGESIZE).float()/255.0
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist(download), **kwargs)

def make_corrupt(original, corrupt_train=False, corrupt_test=True, **kwargs):
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = original
    if int(corrupt_train):
        trainData = make_data_corrupt(trainData, kwargs)
    if int(corrupt_test):
        testData = make_data_corrupt(testData, kwargs)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def make_data_corrupt(data, kwargs):
    N, C, W, H = data.size()
    out = [
        corrupt(
            im.permute(1, 2, 0).squeeze().numpy(),
            **kwargs
        ) for im in tqdm.tqdm(data, desc="Corrupting test set", ncols=80)
    ]
    out = numpy.array(out)
    out = torch.from_numpy(out).float()
    out = out.view(N, W, H, C).permute(0, 3, 1, 2)
    return out/255.0 # numpy converts 1.0 -> 255.0 automatically...

def get_emnist(split, download=0): # recommended to use split = "letters"
    
    download = int(download)
    
    NUM_CLASSES = {
        "byclass": 62,
        "bymerge": 47,
        "balanced": 47,
        "letters": 26,
        "digits": 10,
        "mnist": 10
    }[split]
    CHANNELS = 1
    IMAGESIZE = (28, 28)
    
    train = torchvision.datasets.EMNIST(root=ROOT, split=split, train=True, download=download)
    trainData = train.train_data.view(-1, 1, *IMAGESIZE).float()/255.0
    trainData = trainData.transpose(-1, -2)
    trainLabels = torch.LongTensor(train.train_labels) - 1 # make it 0-indexed
    
    test = torchvision.datasets.EMNIST(root=ROOT, split=split, train=False, download=download)
    testData = test.test_data.view(-1, 1, *IMAGESIZE).float()/255.0
    testData = testData.transpose(-1, -2)
    testLabels = torch.LongTensor(test.test_labels) - 1 # make it 0-indexed

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_emnist_corrupt(split, download=0, **kwargs):
    return make_corrupt(get_emnist(split, download), **kwargs)

def get_cifar10(download=0):
    
    download = int(download)
    
    NUM_CLASSES = 10
    CHANNELS = 3
    IMAGESIZE = (32, 32)
    
    train = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=download)
    trainData = torch.from_numpy(train.train_data).contiguous().view(-1, IMAGESIZE[0], IMAGESIZE[1], CHANNELS).permute(0, 3, 1, 2).float()/255.0
    trainLabels = torch.LongTensor(train.train_labels)

    test = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=download)
    testData = torch.from_numpy(test.test_data).contiguous().view(-1, IMAGESIZE[0], IMAGESIZE[1], CHANNELS).permute(0, 3, 1, 2).float()/255.0
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_cifar10_corrupt(download=0, **kwargs):
    return make_corrupt(get_cifar10(download), **kwargs)

def get_circlesqr_translate(samples=400):
    
    samples = int(samples)
    
    NUM_CLASSES = 2
    CHANNELS = 1
    IMAGELEN = 28
    IMAGESIZE = (IMAGELEN, IMAGELEN)
    
    RADIUS = 2
    
    MAP = [
        generate_circle,
        generate_square
    ]
    
    f = lambda m, n: random.randint(m+RADIUS, n-RADIUS-1)
    g = lambda s, d, sz: torch.from_numpy(d).view(s, *sz)
    
    def generate_set(samples, x0, x1, y0, y1):
        dat = [None] * samples
        lab = [None] * samples
        
        for i in range(samples):
            lab[i] = random.randint(0, 1)
            origin = [f(x0, x1), f(y0, y1)]
            dat[i] = MAP[lab[i]](origin, RADIUS, IMAGELEN)
        
        dat = numpy.array(dat)
        lab = numpy.array(lab)
        
        dat = g(samples, dat, [CHANNELS, IMAGELEN, IMAGELEN]).float()
        lab = g(samples, lab, []).long()
        return dat, lab
    
    half = IMAGELEN // 2
    train_dat, train_lab = generate_set(samples, 0, half, 0, half)
    test_dat, test_lab = generate_set(samples, half, IMAGELEN, half, IMAGELEN)
    
    return train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_sqrquadrants(samples=6000):
    
    samples = int(samples)
    
    NUM_CLASSES = 4
    CHANNELS = 1
    IMAGELEN = 28
    IMAGESIZE = (IMAGELEN, IMAGELEN)
    HALF = IMAGELEN // 2
    
    g = lambda s, d, sz: torch.from_numpy(d).view(s, *sz)
    
    def generate_set(samples):
        dat = [None] * samples
        lab = [None] * samples
        
        for i in range(samples):
            lab[i] = random.randint(0, 3)
            qx, qy = divmod(lab[i], 2)
            dx = qx * HALF
            dy = qy * HALF
            r = random.randint(1, 5)
            ox = r + random.randint(0, HALF-2*r-1)
            oy = r + random.randint(0, HALF-2*r-1)
            origin = (ox + dx, oy + dy)
            dat[i] = generate_square(origin, r, IMAGELEN)
        
        dat = numpy.array(dat)
        lab = numpy.array(lab)
        
        dat = g(samples, dat, [CHANNELS, IMAGELEN, IMAGELEN]).float()
        lab = g(samples, lab, []).long()
        return dat, lab
    
    half = IMAGELEN // 2
    train_dat, train_lab = generate_set(samples)
    test_dat, test_lab = generate_set(samples)
    
    return train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_circlesqr_magnify(*args, **kwargs):
    return get_circlesqr_resize(0, *args, **kwargs)

def get_circlesqr_shrink(*args, **kwargs):
    return get_circlesqr_resize(1, *args, **kwargs)

def get_circlesqr_resize(shrink, samples=400):
    
    samples = int(samples)
    
    NUM_CLASSES = 2
    CHANNELS = 1
    IMAGELEN = 28
    IMAGESIZE = (IMAGELEN, IMAGELEN)
    
    CENTER = (IMAGELEN//2, IMAGELEN//2)
    
    MAP = [
        generate_circle,
        generate_square
    ]
    
    g = lambda s, d, sz: torch.from_numpy(d).view(s, *sz)
    
    def generate_set(samples, r0, r1):
        dat = [None] * samples
        lab = [None] * samples
        
        for i in range(samples):
            lab[i] = random.randint(0, 1)
            rad = random.randint(r0, r1-1)
            dat[i] = MAP[lab[i]](CENTER, rad, IMAGELEN)
        
        dat = numpy.array(dat)
        lab = numpy.array(lab)
        
        dat = g(samples, dat, [CHANNELS, IMAGELEN, IMAGELEN]).float()
        lab = g(samples, lab, []).long()
        return dat, lab
    
    quart = IMAGELEN // 4
    
    radii = [(3, quart-2), (quart+2, IMAGELEN//2)]
    train_dat, train_lab = generate_set(samples, *radii[shrink])
    test_dat, test_lab = generate_set(samples, *radii[1-shrink])
    
    return train_dat, train_lab, test_dat, test_lab, NUM_CLASSES, CHANNELS, IMAGESIZE

def unittest():

    from matplotlib import pyplot
    
#    td, tl, sd2, sl, n, c, i = get_sqrquadrants()
#    
#    for im, lb in zip(td[:10], tl[:10]):
#        im = im.squeeze().numpy()
#        print(lb)
#        pyplot.imshow(im, cmap="gray")
#        pyplot.show()
#        pyplot.clf()
    
    td, tl, sd2, sl, n, c, i = get_mnist(download=1)
    #td, tl, sd, sl, n, c, i = get_mnist_corrupt(download=0, minmag=1, maxmag=1, mintrans=0, maxtrans=0, minrot=0, maxrot=0, alpha=0.5, beta=1.0, sigma=0)
    
#    print("Showing train data")
#    
    for i in range(10000):
        label = tl[i].item()
        if label != 9:
            continue
        im = td[i]
        im = im.squeeze().numpy()
        pyplot.imshow(im, cmap="gray")
        pyplot.show()
        pyplot.clf()
    
#    print("Showing test data")
#    
#    N = 100
#    
#    indices = numpy.arange(len(sd))
#    numpy.random.shuffle(indices)
#    indices = indices[:N]
#    
#    for im, cls in zip(sd[indices], sl[indices]):
#        label = cls.item()
#        if label != 7:
#            continue
#        im = im.permute(1, 2, 0).squeeze().numpy()
#        pyplot.imshow(im, cmap="gray", vmin=0, vmax=1)
#        pyplot.show()
#        pyplot.clf()
