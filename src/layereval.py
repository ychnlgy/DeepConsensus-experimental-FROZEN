from matplotlib import pyplot

# EMNIST, balanced

TOT_NOISE = 0.411
MIU_NOISE = [0.029521275846723545, 0.2312233983439968, 0.39707445844690853, 0.38127658627134686]
STD_NOISE = [0.018709274756350206, 0.04309041687002444, 0.04498906140328897, 0.046520283515651344]

TOT_BLUR = 0.689
MIU_BLUR = [0.15563829811884367, 0.4480851037388152, 0.6647340431809425, 0.6520744679456062]
STD_BLUR = [0.0390682530346692, 0.04982915168657338, 0.04919060064523528, 0.0492885728229028]

TOT_TRANS = 85.3
MIU_TRANS = [0.5646276462902414, 0.7502127479999623, 0.7581382802826293, 0.7923935975166078]
STD_TRANS = [0.050201408533735045, 0.042299634406440516, 0.04766462365641103, 0.042679180094108864]

TOT_ROT = 59.6
MIU_ROT = [0.3034574480608423, 0.3511170224147908, 0.5578723372297084, 0.590797870875673]
STD_ROT = [0.05005091935972197, 0.04783356872933732, 0.047670267178763104, 0.04590040166737066]

TOT_MAG = 69.6
MIU_MAG = [0.1557978729103157, 0.460797869620171, 0.5838297852810393, 0.6605851073214348]
STD_MAG = [0.033843009868595876, 0.05439901612627148, 0.05312897181556344, 0.047540337537934914]

pyplot.rcParams["font.family"] = "serif"

pyplot.errorbar(NOISE)
pyplot.errorbar(TRANS)
pyplot.errorbar(MAG)

pyplot.legend(["Noise (30 std)", "Blur (1.2 std)", "Translation (20 pixels)", "Rotation (45 degrees)", "Magnification (1.5)"])
pyplot.ylabel("Accuracy (%)")
pyplot.xlabel("Layer")

pyplot.savefig("layereval.png")
