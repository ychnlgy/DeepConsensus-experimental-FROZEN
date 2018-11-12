# All tests made after 30 epochs
# rStand is the standard MNIST-rot

# 126K parameter Group Equivariant P4M Network: for translation, rotation and mirror equivariance.
GC_TRANS4 = [0.29339999228715896, 0.2156999945640564, 0.23819999322295188]
GC_ROTA45 = [0.7066999810934067, 0.6831999826431274, 0.7002999794483185]
GC_ROTA90 = [0.2294999997317791, 0.20759999953210354, 0.17400000020861625]
GC_MIRROR = [0.40409999758005144, 0.3839999990165234, 0.3924999985098839]
GC_rStand = [0.9295400021076202, 0.9255200008153915, 0.9219000022411347]

# 132K parameter DeepConsensus
DC_TRANS4 = [0.9951999813318253, 0.9918999737501144, 0.9926999765634537]
DC_ROTA45 = [0.675799983739853, 0.6371999871730805, 0.6725999814271927]
DC_ROTA90 = [0.16370000012218952, 0.19499999977648258, 0.1718000001460314]
DC_MIRROR = [0.3682000008225441, 0.3999999985098839, 0.37430000156164167]
DC_rStand = [0.9493000013828278, 0.9492200008630752, 0.9476400017738342]

from matplotlib import pyplot, lines
import misc, statistics

@misc.main
def main():
    pyplot.rcParams["font.family"] = "serif"
    for group, gc, dc in [
        ("Translation (4 pixels)", GC_TRANS4, DC_TRANS4),
        ("Rotation (45 degrees)", GC_ROTA45, DC_ROTA45),
        ("Rotation (90 degrees)", GC_ROTA90, DC_ROTA90),
        ("Mirror", GC_MIRROR, DC_MIRROR),
        ("Standard MNIST-rot", GC_rStand, DC_rStand)
    ]:
        gc_miu = statistics.mean(gc)
        gc_std = statistics.stdev(gc)
        dc_miu = statistics.mean(dc)
        dc_std = statistics.stdev(dc)
        
        pyplot.errorbar([dc_miu], [group], xerr=[dc_std], fmt="bo")
        pyplot.errorbar([gc_miu], [group], xerr=[gc_std], fmt="rx")
    
    dc = lines.Line2D([], [], color="b", marker="o", label="DeepConsensus")
    gc = lines.Line2D([], [], color="r", marker="x", label="P4M Group Equivariant CNN")
    
    pyplot.legend(handles=[dc, gc])
    
    #pyplot.xticks(rotation=90)
    pyplot.xlabel("Accuracy")
    pyplot.savefig("p4m-results.png", bbox_inches="tight")


