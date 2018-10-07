import tqdm, numpy, random, torch, torchvision, matplotlib, os

from .geometry import generate_circle, generate_square
from .corrupt import corrupt
from .wscipy import scipy
import scipy.misc # because spock2 doesn't have skimage

import torch.utils.data
import torchvision.datasets

DIR = os.path.dirname(__file__)
ROOT = os.path.join(DIR, "..", "..", "data")

def create_trainvalid_split(datalimit, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch):
    assert 0 <= datalimit <= 1
    n = int(len(train_dat) * datalimit)
    indices = numpy.arange(n)
    numpy.random.shuffle(indices)
    p = 0.2
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
    
    test = torchvision.datasets.MNIST(root=ROOT, train=False, transform=torchvision.transforms.ToTensor(), download=download)
    testData = test.test_data.view(-1, 1, *IMAGESIZE).float()/255.0
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist(download), kwargs)

def make_corrupt(original, kwargs):
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = original
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
    return make_corrupt(get_cifar10(download), kwargs)

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
    
    td, tl, sd2, sl, n, c, i = get_cifar10(download=0)
    td, tl, sd, sl, n, c, i = get_cifar10_corrupt(download=0, minmag=0.5, maxmag=1.5, mintrans=-5, maxtrans=5, minrot=-20, maxrot=20)
    
#    print("Showing train data")
#    
#    for im in td[:3]:
#        im = im.squeeze().numpy()
#        pyplot.imshow(im, cmap="gray")
#        pyplot.show()
#        pyplot.clf()
    
    print("Showing test data")
    
    for ims in zip(sd[:10], sd2[:10]):
        for im in ims:
            im = im.permute(1, 2, 0).squeeze().numpy()
            pyplot.imshow(im, cmap="gray")
            pyplot.show()
            pyplot.clf()
