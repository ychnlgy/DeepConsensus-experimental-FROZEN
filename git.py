#!/usr/bin/python3

import os, sys

os.system('''\
git add . && git commit -m "%s" && git push''' % " ".join(sys.argv[1:])
)
