from matplotlib import pyplot

NOISE = [0.4177, 0.3209, 0.2573, 0.6001, 0.8855, 0.9538, 0.9782, 0.9781]
TRANS = [0.4129, 0.4994, 0.9276, 0.9461, 0.7934, 0.8580, 0.5282, 0.5869]
MAG = [0.2486, 0.4175, 0.4021, 0.3792, 0.4299, 0.4026, 0.3407, 0.3586]
pyplot.plot(NOISE)
pyplot.plot(TRANS)
pyplot.plot(MAG)

pyplot.legend(["Gaussian noise (30 s)", "Translation (20 pixels)", "Magnification (2x)"])
pyplot.ylabel("Accuracy (%)")
pyplot.xlabel("Layer")

pyplot.savefig("layereval.png")
