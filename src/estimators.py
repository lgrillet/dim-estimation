import itertools
from collections import defaultdict
from math import sqrt, floor, log

######################################
# Tools                              #
######################################

def distance(a,b):
    d = len(a)
    squared_norm = 0
    for i in range(d):
        squared_norm += (a[i]-b[i])*(a[i]-b[i])
    return sqrt(squared_norm)

def torus_distance(a,b,maxima):
    d = len(a)
    r = []
    for i in range(d):
        c = abs(a[i]-b[i])
        if (c > maxima[i]/2):
            c = maxima[i] - c
        r.append(c)
    return sqrt(sum(x*x for x in r))

######################################
# CORSUM                             #
######################################
# Compute the correlation dimension of a cloud of points
# cloud : cloud of points
# e1 > e2 : scales
# torus : True to compute in the torus of size maxima
# maxima : only used if torus is at True
# mute : True to disable every print

def corsum(cloud, e1, e2, torus=False, maxima=[], mute=False):
    if len(cloud)==0:    
        print("The cloud of points is empty.")
        return -1 # Returns error
    if torus:
        if len(cloud[0]) == len(maxima):
            dist = (lambda a, b : torus_distance(a, b, maxima))
        else:
            print("maxima does not have the appropriate size.")
            return -1 # Returns error
    else:
        dist = distance
    N=len(cloud)
    count1=0 # Number of points at distance at most e1
    count2=0 # Number of points at distance at most e2

    # compute boxes to speed up the computation
    if not mute:
        print("Computing boxes...", end="\r")
    boxes = defaultdict(set)
    shifts = [i for i in itertools.product(range(2), repeat=len(cloud[0]))]
    for j, point in enumerate(cloud):
        hashed = [floor(x/(2*e1)) for x in point]
        for shift in shifts:
            box = [x+i for x,i in zip(hashed,shift)]
            if torus:
                for k, x in enumerate(box):
                    if (x+1)*2*e1>maxima[k]:
                        box[k] = x-floor(maxima[k]/(2*e1))
            boxes[tuple(box)].add(tuple(point))
    
    # count the number of points at distance at most e1 and e2
    for j, point in enumerate(cloud):
        hashed = [round(x/(2*e1)) for x in point]
        if torus:
            for k, x in enumerate(hashed):
                if (x+1)*2*e1>maxima[k]:
                    hashed[k] = x-floor(maxima[k]/(2*e1))
        for point2 in boxes[tuple(hashed)]:
            d = dist(point, point2)
            if d<e1 and point[0]<point2[0]:
                count1+=1
                if d<e2:
                    count2+=1
        if (j+1)%1000 == 0 and not mute:
            print("Computing distances:",j+1,"/",N, "points.", end="\r")

    # review the results
    if count2==0:
        print("There are no pairs of points at distance", e2)
        if count1==0:
            print("And here are no pairs of points at distance", e1)
        return -1 # Returns error
    if not mute:
        print("There are", count1, "pairs of points at distance at most", e1)
        print("There are", count2, "pairs of points at distance at most", e2)

    # compute the correlation dimension
    dimension = log(count2/count1)/log(e2/e1)
    if not mute:
        print("The correlation dimension is", dimension)
    return dimension