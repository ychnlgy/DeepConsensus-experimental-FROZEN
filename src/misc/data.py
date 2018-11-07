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
    
    dataloader = create_loader(train_dat[trainidx], train_lab[trainidx], trainbatch)
    validloader = create_loader(train_dat[valididx], train_lab[valididx], testbatch)
    testloader = create_loader(test_dat, test_lab, testbatch)
    
    return dataloader, validloader, testloader

def create_loader(dat, lab, batch):
    dataset = torch.utils.data.TensorDataset(dat, lab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return dataloader

def get_fashionmnist(download=0):
    download = int(download)
    NUM_CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = (32, 32)
    
    fashion_ROOT = os.path.join(ROOT, "fashion")
    
    train = torchvision.datasets.FashionMNIST(root=fashion_ROOT, train=True, download=download)
    trainData = train.train_data.view(-1, 1, 28, 28).float()/255.0
    trainData = convert_size(trainData, IMAGESIZE)
    trainLabels = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.FashionMNIST(root=fashion_ROOT, train=False, download=download)
    testData = test.test_data.view(-1, 1, 28, 28).float()/255.0
    testData = convert_size(testData, IMAGESIZE)
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_fashionmnist64(download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_fashionmnist(download)
    trainData = convert_size(trainData, IMAGESIZE)
    testData = convert_size(testData, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_fashionmnist64_corrupt(download=0, **kwargs):
    return make_corrupt(get_fashionmnist64(download), **kwargs)

def get_svhn(download=0, **kwargs):
    download = int(download)
    NUM_CLASSES = 10
    CHANNELS = 3
    IMAGESIZE = (32, 32)
    
    svhn_ROOT = os.path.join(ROOT, "svhn")
    
    train = torchvision.datasets.SVHN(root=svhn_ROOT, split="train", download=download)
    train_data, train_labels = pillow_to_numpy(train)
    trainData = train_data.permute(0, 3, 1, 2).float()/255.0
    trainLabels = torch.LongTensor(train_labels)
    
    test = torchvision.datasets.SVHN(root=svhn_ROOT, split="test", download=download)
    test_data, test_labels = pillow_to_numpy(test)
    testData = test_data.permute(0, 3, 1, 2).float()/255.0
    testLabels = torch.LongTensor(test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def pillow_to_numpy(dataset):
    data, labs = [], []
    for d, l in tqdm.tqdm(dataset, desc="Converting to numpy array", ncols=80):
        data.append(numpy.array(d))
        labs.append(l)
    data = torch.from_numpy(numpy.array(data))
    return data, labs

def get_mnist(download=0):
    
    download = int(download)
    
    NUM_CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = (32, 32)
    
    train = torchvision.datasets.MNIST(root=ROOT, train=True, download=download)
    trainData = train.train_data.view(-1, 1, 28, 28).float()/255.0
    trainData = convert_size(trainData, IMAGESIZE)
    trainLabels = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.MNIST(root=ROOT, train=False, download=download)
    testData = test.test_data.view(-1, 1, 28, 28).float()/255.0
    testData = convert_size(testData, IMAGESIZE)
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist64(download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_mnist(download)
    trainData = convert_size(trainData, IMAGESIZE)
    testData = convert_size(testData, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist64_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist64(download), **kwargs)

def get_mnistrgb(download=0, r=1, g=1, b=1):
    r, g, b = float(r), float(g), float(b)
    CHANNELS = 3
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, _, IMAGESIZE = get_mnist(download)
    trainData = trainData.repeat(1, 3, 1, 1)
    testData = torch.cat([testData*v for v in [r, g, b]], dim=1)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnistrgb_corrupt(download=0, r=1, g=1, b=1, **kwargs):
    return make_corrupt(get_mnistrgb(download, r, g, b), **kwargs)

def convert_size(data, size):
    N, C, W, H = data.size()
    X, Y = size
    CX = (X - W)//2
    CY = (Y - H)//2
    out = torch.zeros(N, C, *size)
    out[:,:,CX:CX+W,CY:CY+H] = data
    return out

def get_mnist64quads(download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_mnist(download)
    NUM_CLASSES *= 4
    trainLabels *= 4
    testLabels  *= 4
    trainData, trainLabels = convert_quads(trainData, trainLabels, IMAGESIZE)
    testData, testLabels = convert_quads(testData, testLabels, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def convert_quads(data, labels, size):
    N, C, W, H = data.size()
    X, Y = size
    quad = torch.LongTensor(N).random_(0, 4) # Integers in [0,4)
    dx = (quad % 2) * X//2
    dy = (quad //2) * Y//2
    out = torch.zeros(N, C, X, Y)
    for i in range(N):
        x = dx[i].item()
        y = dy[i].item()
        out[i,:,x:x+W, y:y+H] = data[i]
    return out, labels+quad

def get_mnist_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist(download), **kwargs)

def make_corrupt(original, corrupt_train=False, corrupt_test=True, **kwargs):
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = original
    
    train_kwargs, test_kwargs = process_corrupt_kwargs(kwargs)
    
    if int(corrupt_train):
        trainData = make_data_corrupt(trainData, train_kwargs)
    if int(corrupt_test):
        testData = make_data_corrupt(testData, test_kwargs)
    
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

TRAIN_KEY = "TRAIN_"

def process_corrupt_kwargs(kwargs):
    train_kwargs = {}
    test_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith(TRAIN_KEY):
            train_kwargs[k[len(TRAIN_KEY):]] = v
        else:
            test_kwargs[k] = v
    return train_kwargs, test_kwargs

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
    IMAGESIZE = (32, 32)
    
    train = torchvision.datasets.EMNIST(root=ROOT, split=split, train=True, download=download)
    trainData = train.train_data.view(-1, 1, 28, 28).float()/255.0
    trainData = convert_size(trainData, IMAGESIZE)
    trainData = trainData.transpose(-1, -2)
    trainLabels = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.EMNIST(root=ROOT, split=split, train=False, download=download)
    testData = test.test_data.view(-1, 1, 28, 28).float()/255.0
    testData = convert_size(testData, IMAGESIZE)
    testData = testData.transpose(-1, -2)
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_emnist_corrupt(split, download=0, **kwargs):
    return make_corrupt(get_emnist(split, download), **kwargs)

def get_emnist64(split, download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_emnist(split, download)
    trainData = convert_size(trainData, IMAGESIZE)
    testData = convert_size(testData, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_emnist64_corrupt(split, download=0, **kwargs):
    return make_corrupt(get_emnist64(split, download), **kwargs)

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

def get_cifar1064(download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_cifar10(download)
    trainData = convert_size(trainData, IMAGESIZE)
    testData = convert_size(testData, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

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
    
    #td, tl, sd, sl, n, c, i = get_svhn(download=0)
    td, tl, sd, sl, n, c, i = get_emnist64_corrupt(
        split = "balanced",
        download=0,
        minmag=1, maxmag=1,
        mintrans=0, maxtrans=0,
        minrot=0, maxrot=0,
        minalpha=1, maxalpha=1,
        minbeta=1, maxbeta=1,
        minsigma=0, maxsigma=0,
        mingauss=0, maxgauss=0
    )
    
#    print("Showing train data")
#    
#    for i in range(10000):
#        label = tl[i].item()
#        #if label != 9:
#        #    continue
#        im = td[i]
#        im = im.squeeze().numpy()
#        pyplot.imshow(im, cmap="gray")
#        pyplot.show()
#        pyplot.clf()
#    
#    print("Showing test data")
    
    N = 100
    
    indices = numpy.arange(len(sd))
    numpy.random.shuffle(indices)
    indices = indices[:N]
    
    import models
    
    for img, cls in zip(sd[indices], sl[indices]):
        label = cls.item()
        print(label)
        im = img.permute(1, 2, 0).squeeze().numpy()
        pyplot.imshow(im, cmap="gray", vmin=0, vmax=1)
        pyplot.show()
        pyplot.clf()
