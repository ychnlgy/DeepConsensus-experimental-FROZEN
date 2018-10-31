#!/usr/bin/python3

import statistics, json, glob, os
from matplotlib import pyplot

import misc, sample

def load(f):
    with open(f, "r") as fi:
        return json.load(fi)

def load_folder(folder):
    files = glob.glob(os.path.join(folder, "*.json"))
    datas = [load(f) for f in files]
    return files, datas

def process_file(f):
    basename = os.path.basename(f)[:-5] # no .json
    cls, epochs, label = basename.split("-")
    return int(cls == "c"), label

def match_data(labels, data):
    out = {}
    for (c, label), d in zip(labels, data):
        if label not in out:
            out[label] = [None, None]
        out[label][c] = process_data(d)
    return out

def process_data(data):
#    mean = statistics.mean(data)
#    stdd = statistics.stdev(data)
#    return mean, stdd
    return data

@misc.main
def main():
    files, datas = load_folder("results")
    labels = [process_file(f) for f in files]
    data = match_data(labels, datas)
    
    out = "../data/figures/scores.png"
    sample.boxplots(labels, datas, out)
        
