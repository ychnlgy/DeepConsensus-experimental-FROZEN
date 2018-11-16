ROOT = "results/abalations/data"

import os, json, collections

ORDER = [
    "cosine-similarity",
    "norm-2",
    "linear-tanh",
    "c prototypes",
    "no h()",
]

def load_file(f):
    d = json.load(open(f))
    return [(k, d[k]) for k in ORDER]

def analyze_fname(fname):
    fname = os.path.splitext(fname)[0]
    dset, part = fname.split("-")
    ptype = part[0]
    pval = float(part[1:])
    return dset, ptype, pval

def load_folder(f):
    for fname in os.listdir(f):
        fi = os.path.join(f, fname)
        try:
            data = load_file(fi)
        except:
            print("Skipped", fi)
            continue
        dset, ptype, pval = analyze_fname(fname)
        yield dset, ptype, pval, data

import misc, statistics
from matplotlib import pyplot

NUM_PTYPES = 5

def gather_ptype(ptype):
    pdata = sorted(ptype)
    x = []
    y = [[] for i in range(NUM_PTYPES)]
    e = [[] for i in range(NUM_PTYPES)]
    l = []
    for xi, yg in pdata:
        x.append(xi)
        for i, (yl, yv) in enumerate(yg):
            y[i].append(statistics.mean(yv))
            e[i].append(statistics.stdev(yv))
            l.append(yl)
    return x, y, e, l

LABELS = [
    ("t", "Translation (pixels)"),
    ("m", "Magnification (fraction)"),
    ("g", "Noise (standard deviation)"),
    ("s", "Blur (standard deviation)")
]

def gather_dset(dset):
    for i, (key, k) in enumerate(LABELS):
        yield i, k, gather_ptype(dset[key])
    
@misc.main
def plot_folder(f=ROOT):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for dset, ptype, pval, data in load_folder(f):
        groups[dset][ptype].append((pval, data))
    
    DSETS = ["mnist"]
    
    pyplot.rcParams["font.family"] = "serif"
    fig, axes = pyplot.subplots(nrows=len(DSETS), ncols=4, sharey=True)
    fig.set_size_inches(16, 5)
    
    
    for i, dsetname in enumerate(DSETS):
        for j, ptype, (x, y, e, l) in gather_dset(groups[dsetname]):
            for yg, eg, lg in zip(y, e, l):
                if "h()" in lg:
                    lg = "absence of $h(\\cdot)$"
                axes[j].errorbar(x, yg, yerr=eg, label=lg, fmt="x--")
                axes[j].set_xlabel(ptype)
                
#    x, y, e, l = gather_ptype(groups["mnist"]["g"])
#    for yg, eg, lg in zip(y, e, l):
#        pyplot.errorbar(x, yg, yerr=eg, label=lg, fmt="x--")
    
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    pyplot.ylim([-0.1, 1.1])
    pyplot.savefig("abalation_results.png", bbox_inches="tight")
    
