#!/usr/bin/python3

import os

import misc

CMD_FORM = "python3 -W ignore ./train.py modelf={dset}.torch repeat=10 dataset={dset}64-corrupt epochs=30 modelid={{modelid}} min{{t}}={{v}} max{{t}}={{v}}"
CMD = None
CMDS = []

EPS = 1e-4

def vrange(vmin, vmax, vstep):
    while abs(vmin - vmax) > EPS:
        yield vmin
        vmin += vstep

def do_tv(t, vmin, vmax, vstep):
    for modelid in [2, 3]:
        for v in vrange(vmin, vmax+vstep, vstep):
            CMDS.append(CMD.format(modelid=modelid, t=t, v=v))

@misc.main
def main(gid, dset, start, **kwargs):

    global CMD
    CMD = CMD_FORM.format(dset=dset)
    
    start = int(start)

    add = " ".join(["=".join(d) for d in kwargs.items()])
    
    if add:
        CMD += " " + add

    if gid == "1":
        do_tv("trans", 5, 20, 5)
        do_tv("mag", 1.25, 2.0, 0.25)
    elif gid == "2":
        do_tv("rot", 15, 60, 15)
        do_tv("gauss", 15, 60, 15)
    elif gid == "3":
        do_tv("sigma", 0.4, 1.6, 0.4)
        do_tv("beta", 0.85, 0.4, -0.15)
    else:
        raise AssertionError
    
    CMDS = CMDS[start:]
    
    for cmd in CMDS:
        print(cmd)
    
    for cmd in CMDS:
        os.system(cmd)
