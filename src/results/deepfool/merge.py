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
    
    w = 2*im.shape[0]
    h = len(files)//2 * im.shape[1]
    if len(im.shape) == 3:
        c = im.shape[2]
        size = (w, h, c)
        out = numpy.zeros((w, h, c))
    else:
        size = (w, h)
    out = numpy.zeros(size)
    for i, (k, v) in enumerate(sep.items()):
        out[:im.shape[0], i * im.shape[1]: (i+1)*im.shape[1]] = v[ORI]
        out[im.shape[0]:, i * im.shape[1]: (i+1)*im.shape[1]] = v[PER]
    scipy.misc.imsave("merged.png", out)

if __name__ == "__main__":
    import sys
    pair(sys.argv[1])
