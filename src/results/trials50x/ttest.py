import os, json
import scipy.stats

def load(jf):
    return json.load(open(jf))

def analyze_folder(folder):
    c = os.path.join(folder, "c.json")
    d = os.path.join(folder, "d.json")
    return load(c), load(d)
