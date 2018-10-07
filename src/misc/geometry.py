import numpy

def generate_circle(origin, radius, l):
    out = numpy.zeros((l, l))
    x = numpy.arange(-radius, radius+1)
    p = numpy.sqrt(radius**2 - x**2)

    ox, oy = origin

    x = x.astype(numpy.int32) + ox
    y = numpy.rint(p).astype(numpy.int32)
    
    y1 = oy + y
    y2 = oy - y
    
    out[x, y1] = 1
    out[x, y2] = 1
    return out
    
def generate_square(origin, radius, l):
    out = numpy.zeros((l, l))
    
    ox, oy = origin
    
    s = numpy.arange(-radius, radius+1)
    out[s+ox, oy+radius] = 1
    out[s+ox, oy-radius] = 1
    out[ox-radius, s+oy] = 1
    out[ox+radius, s+oy] = 1
    return out
