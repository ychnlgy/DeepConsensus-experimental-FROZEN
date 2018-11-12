import numpy, tqdm
from os import path

TRAIN = "mnist_all_rotation_normalized_float_train_valid.amat"
TEST = "mnist_all_rotation_normalized_float_test.amat"

DIR = path.dirname(__file__)
DATA = path.join(DIR, "..", "..", "data", "mnist-rot")

def get_trainset(folder=DATA):
    return parse_amat(path.join(folder, TRAIN))

def get_testset(folder=DATA):
    return parse_amat(path.join(folder, TEST))

def parse_amat(f):
    dats = []
    labs = []
    for d, l in tqdm.tqdm(_parse_amat(f), desc="Loading amat file", ncols=80):
        dats.append(d)
        labs.append(l)
    data = numpy.array(dats).reshape((-1, 28, 28))
    return data, numpy.array(labs)

def _parse(f):
    with open(f) as fi:
        for line in fi:
            yield line.rstrip().split(" ")

def _str2float(arr):
    return [float(v) for v in arr]

def _parse_amat(f):
    for dataline in _parse(f):
        if dataline:
            data = _str2float(dataline[:-1])
            label = float(dataline[-1])
            yield data, label

if __name__ == "__main__":
    dats, labs = parse_amat(TRAIN)
    
    from matplotlib import pyplot
    
    pyplot.imshow(dats[5], cmap="gray", vmin=0, vmax=1)
    pyplot.show()
