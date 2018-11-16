import statistics, json
import misc
import scipy.stats

ROOT = "results/variance_analysis/%s-%s.json"
C_VAR = ROOT % ("c", "var")
C_CONST = ROOT % ("c", "const")
D_VAR = ROOT % ("d", "var")
D_CONST = ROOT % ("d", "const")

def load(f):
    data = json.load(open(f))
    print(f)
    print(statistics.stdev(data))
    return data

def levene(f1, f2):
    return scipy.stats.levene(load(f1), load(f2))

print(levene(C_VAR, C_CONST))
print(levene(D_VAR, D_CONST))
