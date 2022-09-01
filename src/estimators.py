import itertools
import scipy.integrate as integrate
import numpy as np
from collections import defaultdict
from math import sqrt, floor, log, acos, pi

# Implementation of different dimension estimators.

# Some tools

def norm(p):
    """Returns the L2 norm of a vector."""
    return sqrt(sum(x*x for x in p))

def distance(a,b):
    """Returns the L2 distance between two vectors."""
    return norm(a-b)

def angle(a,b):
    """Returns the angle between two vectors."""
    r = np.dot(a,b) / (norm(a)*norm(b))
    # Deals with computation errors that can happen due to float precision
    if r>1:
        r=1
    if r<-1:
        r=-1
    return acos(r)

def torus_distance(a,b,maxima):
    """Distance between two vectors in the torus of dimension
    [0,x1_max] * ... * [0,xn_max]
    where maxima = [x1_max, ..., xn_max]
    """
    d = len(a)
    if d != len(maxima):
        print("maxima does not have the right length.")
        return -1
    r = []
    for i in range(d):
        c = abs(a[i]-b[i])
        if (c > maxima[i]/2):
            c = maxima[i] - c
        r.append(c)
    return sqrt(sum(x*x for x in r))
    
def f_ANOVA(t,d):
    """Repartition of the cosinus of the angle between two vectors
    in a sphere of dimension d-1."""
    return pow(sqrt(1-t*t),d-3)

def P_ANOVA(d):
    """Integral of f_ANOVA from -1 to 1, used for renormalization."""
    f = lambda t: f_ANOVA(t,d)
    return integrate.quad(f, -1, 1)[0]

def mean_ANOVA(d):
    """Mean of the square of the difference between an angle and pi/2."""
    if d==1:
        return (pi/2)**2
    f = lambda t: f_ANOVA(t,d)*(acos(t)-pi/2)**2
    return integrate.quad(f, -1, 1)[0] / P_ANOVA(d)

# Estimators

def corsum(cloud, e1, e2, noboxes=False, pairs=-1,
           torus=False, maxima=[], mute=False):
    """Estimate the correlation dimension of a point cloud.

    Arguments:
    cloud -- list or numpy array of a point cloud
    e1 > e2 -- scales
    noboxes -- True to disable an optimization (useful for high dimensions)
    pairs -- limit the number of points to obtain this number of pairs
             (-1 by default for no limitation)
    torus -- True to measure distances in a flat torus
    maxima -- if torus==True, defines the list lengths of the torus
    mute -- True to avoid printing anything, except if there is an error
    """
    cloud = np.array(cloud)
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
    n=len(cloud)

    # count the number of pairs
    count1=0 # Number of pairs of points at distance at most e1
    count2=0 # Number of pairs of points at distance at most e2

    # naive algorithm for high dimensions
    if noboxes:
        for i, point in enumerate(cloud):
            for j, point2 in enumerate(cloud):
                d = dist(point, point2)
                if d<e1 and j<i:
                    count1+=1
                    if d<e2:
                        count2+=1
            if pairs > 0:
                if count1 >= pairs:
                    break
            if (i+1)%10 == 0 and not mute:
                print("Computing distances:",i+1,"/",n , "points.", end="\r")
    
    # faster algorithm for low dimensions
    else:
        boxes = defaultdict(set)
        shifts = [[x-1 for x in i] for i in
                  itertools.product(range(3), repeat=len(cloud[0]))]

        for i, point in enumerate(cloud):

            hashed = [floor(x/(2*e1)) for x in point]

            if torus:
                for k, x in enumerate(hashed):
                    if (x+1)*2*e1>maxima[k]:
                        hashed[k] = x-floor(maxima[k]/(2*e1))
            for l in boxes[tuple(hashed)]:
                d = dist(point, cloud[l])
                if d<e1:
                    count1+=1
                    if d<e2:
                        count2+=1
            if pairs > 0:
                if count1 >= pairs:
                    break

            for shift in shifts:
                box = [x+i for x,i in zip(hashed,shift)]
                if torus:
                    for k, x in enumerate(box):
                        if (x+1)*2*e1>maxima[k]:
                            box[k] = x-floor(maxima[k]/(2*e1))
                boxes[tuple(box)].add(i)

            if (i+1)%500 == 0 and not mute:
                print("Computing distances:",i+1,"/",n , "points.", end="\r")

    # review the results
    if pairs > 0:
        if count1 < pairs:
            print("The number of pairs has not been reached")
            return -1
    if count2==0:
        if not mute:
            print("There are no pairs of points at distance", e2)
            print("And", count1, "pairs of points at distance", e1)
        return -1 # Returns error
    if not mute:
        print("There are", count1, "pairs of points at distance at most", e1)
        print("There are", count2, "pairs of points at distance at most", e2)
        if pairs>0:
            print(i, "points were used.")

    # compute the correlation dimension
    dimension = log(count2/count1)/log(e2/e1)
    if not mute:
        print("The correlation dimension is", dimension)
    return dimension

def count_scales(cloud, scales, noboxes=False, torus=False, maxima=[], mute=False):
    """Count the number of pairs at different scales.
    Useful for the log-log plots.

    Arguments:
    cloud -- list or numpy array of a point cloud
    scales -- list of scales
    torus -- True to measure distances in a flat torus
    maxima -- if torus==True, defines the list lengths of the torus
    mute -- True to avoid printing anything, except if there is an error
    """
    cloud = np.array(cloud) 
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
    n=len(cloud)

    # count the number of pairs
    counts=[0]*len(scales)

    if noboxes:
        for i, point in enumerate(cloud):
            for j, point2 in enumerate(cloud):
                d = dist(point, point2)
                for k, scale in enumerate(scales):
                    if d<scale and j<i:
                        counts[k] += 1

            if (i+1)%10 == 0 and not mute:
                print("Computing distances:",i+1,"/",n , "points.", end="\r")
        

    else:
        boxes = defaultdict(set)
        shifts = [[x-1 for x in i] for i in 
                itertools.product(range(3), repeat=len(cloud[0]))]

        for i, point in enumerate(cloud):

            e0 = max(scales)
            hashed = [floor(x/(2*e0)) for x in point]
            if torus:
                for k, x in enumerate(hashed):
                    if (x+1)*2*e0>maxima[k]:
                        hashed[k] = x-floor(maxima[k]/(2*e0))

            for l in boxes[tuple(hashed)]:
                d = dist(point, cloud[l])

                for j, scale in enumerate(scales):
                    if d<scale:
                        counts[j] += 1

            for shift in shifts:
                box = [x+i for x,i in zip(hashed,shift)]
                if torus:
                    for k, x in enumerate(box):
                        if (x+1)*2*e0>maxima[k]:
                            box[k] = x-floor(maxima[k]/(2*e0))
                boxes[tuple(box)].add(i)

            if (i+1)%250 == 0 and not mute:
                print("Computing distances:",i+1,"/",n , "points.", end="\r")

    return counts

######################################
# ANOVA                             #
######################################
# Compute the ANOVA dimension of a cloud of points
# cloud : cloud of points
# eps : scale
# torus : True to compute in the torus of size maxima
# maxima : only used if torus is at True
# full_output: True to return number of triples in addition to the dimension
# mute : True to disable every print

def anova(cloud, eps, mute=False):
    """Estimate the dimension using ANOVA.

    Arguments:
    cloud -- list or numpy array of a point cloud
    eps -- scale
    mute -- True to avoid printing anything, except if there is an error
    """
    cloud = np.array(cloud)
    if len(cloud)==0:    
        print("The cloud of points is empty.")
        return -1 # Returns error
    dist = distance
    n=len(cloud)

    # compute boxes to speed up the computation
    if not mute:
        print("Computing boxes...", end="\r")
    boxes = defaultdict(set)
    shifts = [i for i in itertools.product(range(2), repeat=len(cloud[0]))]
    for j, point in enumerate(cloud):
        hashed = [floor(x/(2*eps)) for x in point]
        for shift in shifts:
            box = [x+i for x,i in zip(hashed,shift)]
            boxes[tuple(box)].add(tuple(point))
    
    # count the number of points at distance at most e1 and e2
    count=0 # Number of angles of points at distance at most eps
    total=0 # Sum of the (angles-pi/2)^2
    for j, point in enumerate(cloud):
        hashed = [round(x/(2*eps)) for x in point]
        for point2 in boxes[tuple(hashed)]:
            for point3 in boxes[tuple(hashed)]:
                d1 = dist(point, point2)
                d2 = dist(point, point3)
                if d1<eps and d2<eps and point[0]!=point2[0] \
                                     and point[0]!=point3[0] \
                                     and point2[0]<point3[0]:
                    count+=1
                    total+=(angle(point2-point, point3-point)-pi/2)**2
                    
        if (j+1)%10 == 0 and not mute:
            print("Computing distances:",j+1,"/",n , "points.", end="\r")

    # review the results
    if count==0:
        if not mute:
            print("There are no triples of points at distance", eps)
        return -1 # Returns error
    if not mute:
        print("The sample variance is ", total/count)
        print("There are", count, " angles at distance at most", eps)

    # compute the correlation dimension
    mean = total/count
    d=0
    var = 1
    while var > mean:
        d+=1
        var = mean_ANOVA(d)
    if mean-var > mean_ANOVA(d-1)-mean:
        dimension = d-1
    else:
        dimension = d

    if not mute:
        print("The correlation dimension is", dimension)
    return dimension