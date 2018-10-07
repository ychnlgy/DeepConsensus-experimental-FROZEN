#!/usr/bin/python3

import numpy, json
from os import path
from matplotlib import pyplot
from scipy.stats import gaussian_kde

RESOLUTION = 100

RESULTS_DIR = path.join(path.dirname(__file__), "..", "results")
DATA_DIR = path.join(RESULTS_DIR, "data")

DATASETS = [
#    (
#        "circlesqr",
#        [
#            "rand",
#            "c350k-trans",
#            "c350k-magnify",
#            "c350k-shrink",
#            "d350k-trans",
#            "d350k-magnify",
#            "d350k-shrink"
#        ]
#    ),
    (
        "mnist-corrupt-distilled",
        [
            #"rand",
            #"d30k0.05m02r02t",
            #"d30k0.10m05r04t",
            "d30k0.20m10r08t",
            "d30k0.30m15r12t"
        ]
    ),
]

def process_results(datasets):
    for dataset_name, files in datasets:
        paths = [
            path.join(DATA_DIR, dataset_name, f + ".json")
            for f in files
        ]
        datas = [json.load(open(p, "r")) for p in paths]
        lab_ys = list(zip(files, datas))
        outfile = path.join(RESULTS_DIR, dataset_name + ".png")
        yield outfile, lab_ys

def create_densityplot(ys):
    xs = numpy.linspace(0, 1, num=RESOLUTION)
    ys = gaussian_kde(ys)(xs)
    return ys
    
def plot(outfile, lab_ys):
    labels = [lab for lab, y in lab_ys]
    ys = [create_densityplot(y) for lab, y in lab_ys]

    N = len(ys)
    tik = [""] * (N*2+1)
    out = numpy.zeros((N*2+1, RESOLUTION))
    for i in range(N):
        out[i*2+1] = ys[i]/numpy.max(ys[i])
        tik[i*2+1] = labels[i]
    
    pyplot.title("Test scores of 100 random initializations after 50 epochs", fontsize=11)
    pyplot.yticks(range(N*2+1), tik)
    im = pyplot.imshow(out, cmap="hot", interpolation="bilinear", aspect="auto", vmin=0, vmax=1)
    cb = pyplot.colorbar(im)
    cb.set_label("Fraction of initializations")
    pyplot.legend(labels)
    pyplot.xlabel("Accuracy (%)")
    pyplot.savefig(outfile, bbox_inches="tight")
    pyplot.clf()

if __name__ == "__main__":
    for args in process_results(DATASETS):
        plot(*args)
