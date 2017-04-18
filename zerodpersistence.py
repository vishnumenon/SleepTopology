import numpy as np

class Vertex(object):
    def __init__(self, x, y, isMax, componentBirths, currComponent=-1):
        self.x = x
        self.y = y
        self.isMax = isMax
        self.component = currComponent
        if not isMax:
            componentBirths[currComponent] = y
        self.children = []

    def addChild(self, child):
        self.children.append(child)

def getFunc0DPersistence(points):
    points = points[::2]
    componentBirths = {}
    cc = 0
    points = sorted(points, key=lambda p: p[0])

    extrema = []
    for i, p in enumerate(points):
        if i > 0 and i < len(points) - 1:
            if not points[i-1][1] <= p[1] <= points[i+1][1] or points[i-1][1] >= p[1] >= points[i+1][1]:
                extrema.append(Vertex(p[0], p[1], points[i-1][1] < p[1] > points[i+1][1], componentBirths, cc))
                cc+=1

    maxHeight = max(map(lambda p: p.y, extrema))

    if points[0][1] < points[1][1] or points[0][1] > maxHeight:
        extrema = [Vertex(points[0][0], points[0][1], points[0][1] > maxHeight, componentBirths, cc)] + extrema
        cc+=1
        maxHeight = max(maxHeight, points[0][1])

    if points[-1][1] < points[-2][1] or points[-1][1] > maxHeight:
        extrema.append(Vertex(points[-1][0], points[-1][1], points[-1][1] > maxHeight, componentBirths, cc))
        cc+=1
        maxHeight = max(maxHeight, points[-1][1])


    for i, p in enumerate(extrema):
        if p.isMax:
            if i > 0:
                p.addChild(extrema[i-1])
            if i < len(extrema) - 1:
                p.addChild(extrema[i+1])

    ppairs = []
    orderedMaxes = list(filter(lambda p: p.isMax, extrema))
    orderedMaxes = sorted(orderedMaxes, key=lambda p: p.y)
    for m in orderedMaxes:
        youngerComponent = (max(m.children, key=lambda p: componentBirths[p.component])).component
        olderComponent = (min(m.children, key=lambda p: componentBirths[p.component])).component
        for e in [e for e in extrema if e.component == youngerComponent and not e.isMax]:
            e.component = olderComponent
        if youngerComponent != olderComponent:
            ppairs.append(np.array([componentBirths[youngerComponent], m.y]))

    return np.array(ppairs)
