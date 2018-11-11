import os, json, statistics
from matplotlib import pyplot

import misc

@misc.main
def main():
    
    data = ["d", "e"]
    
    indx = list(range(1, 17))
    
    folder = "results/nov9/%s-%s.json"
    
    for dname in data:
        for i in indx:
            fname = folder % (dname, i)
            jdata = json.load(open(fname))
            
            miu = statistics.mean(jdata)
            std = statistics.stdev(jdata)
            pyplot.errorbar(["%s%s" % (dname, i)], [miu], yerr=[std], fmt="o")
    
        pyplot.savefig("%s.png" % dname)
        pyplot.clf()
