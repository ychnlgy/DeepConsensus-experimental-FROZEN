#!/usr/bin/python3

import os

import misc

@misc.main
def main(f):
    cmds = list(misc.util.parse(f))
    for cmd in cmds:
        print("{0}\n>> {1}\n{0}".format("="*60, cmd))
        os.system(cmd)
