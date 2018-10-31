from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][0], markeredgecolor='blue')
    setp(bp['fliers'][1], markeredgecolor='red')
    setp(bp['medians'][1], color='red')

def boxplots(labels, data, outf):
    ## Some fake data to plot
    #A= [[1, 2, 5,],  [7, 2]]
    #B = [[5, 7, 2, 2, 5], [7, 2, 5]]
    #C = [[3,2,5,7], [6, 7, 3]]

    fig = figure()
    ax = axes()
    hold(True)

    for i, d in enumerate(data):
        bp = boxplot(d, positions = [i*3+1, i*3+2], widths=0.6)
        setBoxColors(bp)

    # set axes limits and labels
    xlim(0,len(labels)*3)
    ylim(0,1)
    ax.set_xticklabels(labels)
    ax.set_xticks([(1.5 + (i*3)) for i in range(len(labels))])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    legend((hB, hR),('ResNet with summary-prototype', 'ResNet'))
    hB.set_visible(False)
    hR.set_visible(False)

    savefig(outf)
    show()
