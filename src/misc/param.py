def convert2d(p):
    if type(p) in [tuple, list]:
        return p
    else:
        return (p, p)
