import os, json, statistics
import scipy.stats
from matplotlib import pyplot, lines

def load(jf):
    return json.load(open(jf))

def analyze_folder(folder):
    c = os.path.join(folder, "c.json")
    d = os.path.join(folder, "d.json")
    c, d = load(c), load(d)
    return scipy.stats.ttest_ind(d, c, equal_var=False), c, d

import misc

@misc.main
def analyze():
    pyplot.rcParams["font.family"] = "serif"

    ROOT = "results/trials50x/results/"
    
    for ftype in sorted(os.listdir(ROOT)):
        print(ftype)
        ttest, c, d = analyze_folder(os.path.join(ROOT, ftype))
        print(ttest)
        miu_c = statistics.mean(c)
        miu_d = statistics.mean(d)
        std_c = statistics.stdev(c)
        std_d = statistics.stdev(d)
        pyplot.errorbar([ftype + " res", ftype + " dc+res"], [miu_c, miu_d], yerr=[std_c, std_d], fmt="o--")
    
    pyplot.xticks(rotation=30)
    pyplot.ylabel("Accuracy")
    pyplot.savefig("resnet-trails50x.png", bbox_inches="tight")
