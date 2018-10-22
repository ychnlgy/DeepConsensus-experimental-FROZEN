import random, numpy

from .wscipy import scipy
import scipy.misc
import scipy.ndimage

from .util import hardmap

def corrupt(im, minmag, maxmag, minrot, maxrot, mintrans, maxtrans, alpha, beta, sigma):

    minmag, maxmag, minrot, maxrot, mintrans, maxtrans, alpha, beta, sigma = hardmap(
        float, 
        minmag, maxmag, minrot, maxrot, mintrans, maxtrans, alpha, beta, sigma
    )

    w, h = im.shape[:2]
    originalshape = im.shape
    ox, oy = w//2, h//2
    im = randomresize(im, minmag, maxmag)
    im = randomrotate(im, minrot, maxrot)
    px, py = randommove(ox, oy, mintrans, maxtrans)
    im = add_noise(im, alpha)
    im = reduce_colorgrad(im, beta)
    im = gaussian_blur(im, sigma)
    return draw(im, px, py, w, h, originalshape)
    
def rand_select(v0, vf):
    return random.random() * (vf - v0) + v0

def gaussian_blur(im, sigma):
    sigma = rand_select(0, sigma)
    return scipy.ndimage.filters.gaussian_filter(im, sigma=sigma)

def add_noise(im, alpha):
    alpha = rand_select(alpha, 1.0)
    noise = numpy.random.rand(*im.shape) * 255.0
    return alpha * im + (1 - alpha) * noise

def reduce_colorgrad(im, beta):
    beta = rand_select(beta, 1.0)
    mean = numpy.mean(im)
    diff = im - mean
    return diff * beta + mean

def randomresize(im, minmag, maxmag):
    scale = rand_select(minmag, maxmag)
    return scipy.misc.imresize(im, scale)

def randommove(ox, oy, mintrans, maxtrans):
    px = ox + random.randint(mintrans, maxtrans)
    py = oy + random.randint(mintrans, maxtrans)
    return px, py

def randomrotate(im, minrot, maxrot):
    degree = rand_select(minrot, maxrot)
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
