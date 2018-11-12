import os, json, statistics, collections
from matplotlib import pyplot

import misc

MNIST = ("MNIST", "abcmno")
FASHION = ("FashionMNIST", "def")
EMNIST = ("EMNIST", "ghipqr")
CIFAR10 = ("CIFAR10", "jkl")

DSETS = [MNIST, EMNIST]

DC = "dc"
CN = "cnn"
SMALL = "-small"

SPLIT = [
    [("Translation", "pixels", [0, 5, 10, 15, 20]), ("Magnification", "fraction", [1, 1.25, 1.5, 1.75, 2.0])],
    [("Rotation", "degrees", [0, 15, 30, 45, 60]), ("Noise", "standard deviation", [0, 15, 30, 45, 60])],
    [("Blur", "standard deviation", [0, 0.4, 0.8, 1.2, 1.6]), ("degraded signal", "fraction", [1, 0.85, 0.7, 0.55, 0.4])]
]

def iter_split():
    i = 0
    for grp in SPLIT:
        for perturb in grp:
            if perturb[0] not in ["degraded signal", "Rotation"]:
                yield i, perturb
                i += 1

# ex. MNIST -> translation -> DC -> miu -> listof means
empty = lambda: [0] * 5
GROUPS = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(empty))))

MIU = "miu"
STD = "std"

def plot():
    pyplot.rcParams["font.family"] = "serif"
    num_dsets = len(DSETS)
    num_perturbs = 4
    fig, axes = pyplot.subplots(nrows=num_dsets, ncols=num_perturbs, sharey=True)
    fig.set_size_inches(16, 10)
    for c, (dsetname, _) in enumerate(DSETS):
        dset = GROUPS[dsetname]
        for i, (pname, punit, paxis) in iter_split():
            perturb = dset[pname]
            plt = axes[c, i]
            plt.tick_params(labelbottom=False)
            for mtype in [DC, CN, DC+SMALL, CN+SMALL]:
                mdata = perturb[mtype]
                p = {DC:"bo", CN:"rx", DC+SMALL: "bs", CN+SMALL: "r^"}[mtype]
                plt.errorbar(paxis, mdata[MIU], yerr=mdata[STD], fmt=p + "--")
    
    for c in range(num_dsets):
        y = 0.52
        if c == 1:
            y = 0.55
        axes[c, 0].set_title(DSETS[c][0], rotation="vertical", x=-0.3, y=y)
    
        axes[c, 0].set_ylabel("Accuracy")

    for i, (pname, punit, paxis) in iter_split():
        plt = axes[-1, i]
        plt.set_xlabel("%s (%s)" % (pname, punit))
        plt.tick_params(labelbottom=True)
        plt.set_xticks(paxis)
    
    axes[0, 0].legend(["DeepConsensus", "Base CNN", "DeepConsensus-Small", "Base CNN-Small"], bbox_to_anchor=[0.7, 0.2])
    
    pyplot.savefig("results.png", bbox_inches="tight")

def get_perturb_properties(perturbname):
    for grp in SPLIT:
        for perturb in grp:
            if perturb[0] == perturbname:
                return perturb[1:]
    assert False

def set_grp_baseline(folder):
    for dset in DSETS:
        for modeltype in [DC, CN]:
            fname = folder % (dset[1][:3], modeltype)
            jdata = json.load(open(fname))
            miu = statistics.mean(jdata)
            std = statistics.stdev(jdata)
            
            for suffix in ["", SMALL]:
                for dname in dset[1]:
                    for ptype in [0, 1]:
                        add_group(dname, 0, miu, std, perturbtype=ptype, modeltype=modeltype+suffix)

def get_type(i):
    if i <= 4:
        return 0, DC, i
    elif i <= 8:
        return 0, CN, i - 4
    elif i <= 12:
        return 1, DC, i - 8
    else:
        return 1, CN, i - 12

def index_group(dname):
    for dsetname, dsetgrp in DSETS:
        if dname in dsetgrp:
            i = dsetgrp.index(dname)
            if i < 3:
                return dsetname, SPLIT[i], ""
            else:
                return dsetname, SPLIT[i-3], SMALL
    assert False

def add_group(dname, i, miu, std, perturbtype=None, modeltype=None):
    dsetname, split, suffix = index_group(dname)
    if perturbtype is None:
        perturbtype, modeltype, i = get_type(i)
        modeltype += suffix
    perturbname = split[perturbtype][0]
    grp = GROUPS[dsetname][perturbname][modeltype]
    grp[MIU][i] = miu
    grp[STD][i] = std

@misc.main
def main():
    
    data = "abcghimnopqr"
    
    indx = list(range(1, 17))
    
    folder = "results/nov9/%s-%s.json"
    
    set_grp_baseline(folder)
    
    for dname in data:
        for i in indx:
            fname = folder % (dname, i)
            
            try:
                jdata = json.load(open(fname))
            except:
                print("Skipping %s" % fname)
                continue
            
            miu = statistics.mean(jdata)
            std = statistics.stdev(jdata)
            add_group(dname, i, miu, std)
    plot()
#            pyplot.errorbar(["%s%s" % (dname, i)], [miu], yerr=[std], fmt="o")
#    
#        pyplot.savefig("%s.png" % dname)
#        pyplot.clf()
