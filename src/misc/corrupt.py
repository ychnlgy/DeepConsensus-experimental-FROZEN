import random, numpy

from .wscipy import scipy
import scipy.misc

from .util import hardmap

def corrupt(im, minmag, maxmag, minrot, maxrot, mintrans, maxtrans):

    minmag, maxmag, minrot, maxrot, mintrans, maxtrans = hardmap(
        float, 
        minmag, maxmag, minrot, maxrot, mintrans, maxtrans
    )

    w, h = im.shape[:2]
    originalshape = im.shape
    ox, oy = w//2, h//2
    im = randomresize(im, minmag, maxmag)
    im = randomrotate(im, minrot, maxrot)
    px, py = randommove(ox, oy, mintrans, maxtrans)
    return draw(im, px, py, w, h, originalshape)

def randomresize(im, minmag, maxmag):
    scale = random.random() * (maxmag - minmag) + minmag
    return scipy.misc.imresize(im, scale)

def randommove(ox, oy, mintrans, maxtrans):
    px = ox + random.randint(mintrans, maxtrans)
    py = oy + random.randint(mintrans, maxtrans)
    return px, py

def randomrotate(im, minrot, maxrot):
    degree = random.random() * (maxrot - minrot) + minrot
    return scipy.misc.imrotate(im, degree)

def draw(im, px, py, w, h, originalshape):
    out = numpy.zeros(originalshape)
    x, y = im.shape[:2]
    hx, hy = x//2, y//2
    xa = max(0, px-hx)
    xb = min(w, px-hx+x)
    ya = max(0, py-hy)
    yb = min(h, py-hy+y)
    dx = xb - xa
    dy = yb - ya
    sx = xa - (px-hx)
    sy = ya - (py-hy)
    out[xa:xa+dx, ya:ya+dy] = im[sx:sx+dx, sy:sy+dy]
    return out
