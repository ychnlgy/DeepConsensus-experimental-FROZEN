import scipy.misc, os, collections, numpy

ORI = "original.png"
PER = "perturbed.png"

def pair(folder):
    files = os.listdir(folder)
    sep = collections.defaultdict(dict)
    for f in files:
        fid, lab = f.split("-")
        im = scipy.misc.imread(os.path.join(folder, f))
        sep[fid][lab] = im
    
    h = 2*im.shape[0]
    w = len(files)//2 * im.shape[1]
    if len(im.shape) == 3:
        c = im.shape[2]
        size = (w, h, c)
        out = numpy.zeros((w, h, c))
    else:
        size = (w, h)
    out = numpy.zeros(size)
    for i, (k, v) in enumerate(sep.items()):
        out[i * im.shape[1]: (i+1)*im.shape[1], :im.shape[0]] = v[ORI]
        out[i * im.shape[1]: (i+1)*im.shape[1], im.shape[0]:] = v[PER]
    scipy.misc.imsave("merged.png", out)

if __name__ == "__main__":
    import sys
    pair(sys.argv[1])
