
def plotPlatform(ax, points, *args, **kwargs):
    oldPoint = points[-1, :]
    for pointNum, point in enumerate(points):
        ax.plot([point[0], oldPoint[0]], 
                [point[1], oldPoint[1]], 
                [point[2], oldPoint[2]], 
                *args, **kwargs)
        oldPoint = point

def drawLinks(ax, base, platform, *args, **kwargs):
    for bp, pp in zip(base, platform):
        ax.plot([bp[0], pp[0]], 
                [bp[1], pp[1]], 
                [bp[2], pp[2]], *args, **kwargs)