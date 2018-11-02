import random, numpy

from .wscipy import scipy
import scipy.misc
import scipy.ndimage

from .util import hardmap

def corrupt(im, minmag=1, maxmag=1, minrot=0, maxrot=0, mintrans=0, maxtrans=0, minalpha=1, maxalpha=1, minbeta=1, maxbeta=1, minsigma=0, maxsigma=0, mingauss=0, maxgauss=0, **kwargs):

    minmag, maxmag, minrot, maxrot, mintrans, maxtrans, minalpha, maxalpha, minbeta, maxbeta, minsigma, maxsigma, mingauss, maxgauss = hardmap(
        float, 
        minmag, maxmag, minrot, maxrot, mintrans, maxtrans, minalpha, maxalpha, minbeta, maxbeta, minsigma, maxsigma, mingauss, maxgauss
    )

    w, h = im.shape[:2]
    originalshape = im.shape
    ox, oy = w//2, h//2
    im = randomresize(im, minmag, maxmag)
    im = randomrotate(im, minrot, maxrot)
    px, py = randommove(ox, oy, mintrans, maxtrans)
    im = draw(im, px, py, w, h, originalshape)
    im = add_noise(im, minalpha, maxalpha)
    im = reduce_colorgrad(im, minbeta, maxbeta)
    im = gaussian_blur(im, minsigma, maxsigma)
    im = gaussian_noise(im, mingauss, maxgauss)
    return im
    
def rand_select(v0, vf):
    return random.random() * (vf - v0) + v0

def gaussian_blur(im, low, high):
    sigma = rand_select(low, high)
    return scipy.ndimage.filters.gaussian_filter(im, sigma=sigma)

def gaussian_noise(im, low, high):
    sigma = rand_select(low, high)
    noise = numpy.random.normal(scale=sigma, size=im.shape)
    assert noise.shape == im.shape
    return im + noise

def add_noise(im, low, high):
    alpha = rand_select(low, high)
    noise = numpy.random.rand(*im.shape) * 255.0
    return alpha * im + (1 - alpha) * noise

def reduce_colorgrad(im, low, high):
    #return im * 
    beta = rand_select(low, high)
    mean = numpy.mean(im)
    diff = im - mean
    return diff * beta + mean

def randomresize(im, minmag, maxmag):
    scale = rand_select(minmag, maxmag)
    return scipy.misc.imresize(im, scale, interp="nearest")

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
