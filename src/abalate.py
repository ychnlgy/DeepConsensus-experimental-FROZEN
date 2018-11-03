import json, os, statistics

from matplotlib import pyplot

import misc

def load(f):
    print(f)
    return json.load(open(f, "r"))

@misc.main
def plot():

    ROOT = "results/abalations/results/"
    
    for fold in os.listdir(ROOT):
    
        folder = os.path.join(ROOT, fold)

        outname = os.path.basename(folder)
        fnames = sorted(os.listdir(folder))
        files  = [os.path.join(folder, f) for f in fnames]
        jsons  = [load(f) for f in files]
        labels = [f[0] for f in fnames]
        for i in range(len(fnames)):
            data = jsons[i]
            if data:
                mean = statistics.mean(data)
                stdd = statistics.stdev(data)
                pyplot.errorbar([labels[i]], [mean], yerr=[stdd], fmt="o")
        pyplot.ylim([-0.1, 1.1])
        pyplot.savefig(outname + ".png")
        pyplot.clf()
