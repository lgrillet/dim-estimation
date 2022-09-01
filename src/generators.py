from ctypes import pointer
import random
from math import cos, sin, sqrt, floor, pi
import numpy as np
import time

# Algorithms used to sample points uniformly in different manifolds.

#######################################
# Tools                               #
#######################################

def squared_norm(p):
    """Returns the squared L2 norm of a vector."""
    squared_norm = 0
    for coord in p:
        squared_norm += coord*coord
    return squared_norm

def step(t):
    """An interpolation function."""
    return (3-2*t)*t*t

def interpolate(a, b, t):
    """Interpolate between a and b at time t in [0,1]."""
    return a + (b-a)*step(t)

def cloud_product(cloud1, cloud2):
    """Return the product of two point clouds."""
    if len(cloud1) != len(cloud2):
        print("The two clouds must have the same number of points.")
        return -1
    return np.concatenate((cloud1,cloud2), axis=1)

#######################################
# Samplers                            #
#######################################

def box_sample(n, dimension, maxima):
    """Samples points uniformly in [0,x1_max]*...*[0,xd_max].

    Arguments:
    n -- number of points sampled
    dimension -- ambiant dimension
    maxima -- [x1_max, x2_max, ..., xd_max]
    """
    if len(maxima) != dimension:
        print("The number of maxima is different from the dimension.\n")
        return np.array([])
    cloud = np.zeros([n,dimension])
    for i in range(n):
        point = []
        for j in range(dimension):
            point.append(random.random()*maxima[j])
        cloud[i] = point
    return cloud

def gaussian_sample(n, dimension, sigmas, maxima=-1):
    """Samples points from  a normal distribution.

    Arguments:
    n -- number of points sampled
    dimension -- ambiant dimension
    sigmas -- list of standard deviations for each direction
    maxima -- define bounds [-x1_max,x1_max]*...*[-xd_max,xd_max]
              -1 for no bound
    """
    if len(sigmas) != dimension:
        print("The number of sigmas is different from the dimension.\n")
        return np.array([])
    cloud = np.zeros([n,dimension])
    for i in range(n):
        point = []
        for j in range(dimension):
            x = random.normalvariate(0, sigmas[j])
            if maxima != -1:
                while abs(x) > maxima[j]:
                    x = random.normalvariate(0, sigmas[j])
            point.append(x)
        cloud[i] = point
    return cloud

def sphere_sample(n, dimension, radius=1, copies=1):
    """Samples points uniformly in an hypersphere.

    Arguments:
    n -- number of points sampled
    dimension -- dimension of the sphere (in ambiant dimension "dimension+1")
    radius -- radius of the sphere
    copies -- number of disjoint copies of the sphere
    """
    dimension = dimension+1
    cloud = np.zeros([n,dimension])
    for i in range(n):
        point = []
        for j in range(dimension):
            point.append(random.normalvariate(0, 1))
        norm = squared_norm(point)
        point = [radius*k/sqrt(norm) for k in point]
        point[0] = point[0] + 4*radius*random.randint(1, copies)
        cloud[i]=point
    return cloud

def clifford_torus_sample(n, dimension, radii=-1):
    """Samples points uniformly in a Clifford torus.

    Arguments:
    n -- number of points sampled
    dimension -- dimension of the torus (in ambiant dimension "2 * dimension")
    radii -- list of radii of the torus
    """
    if radii == -1:
        radii = [1]*dimension
    cloud = np.zeros([n,2*dimension])
    for i in range(n):
        point = []
        for j in range(dimension):
            angle = random.random()*2*pi
            point.append(cos(angle)*radii[j])
            point.append(sin(angle)*radii[j])
        cloud[i] = point
    return cloud

def torus_sample(n, dimension, R=2, r=1):
    """Samples points uniformly in a revolution torus.

    Arguments:
    n -- number of points sampled
    dimension -- dimension of the torus (in ambiant dimension "dimension*3/2")
    R -- big radius
    r -- small radius
    """
    if dimension%2 != 0:
        print("Error: the dimension must be even.")
        return np.array([])
    points = []
    for i in range(n):
        point = []
        count = int(dimension/2)
        while count > 0:
            u = random.random()*2*pi
            v = random.random()*2*pi
            w = random.random()
            if (R+r*cos(u))/(R+r):
                x = (R+r*cos(u))*cos(v)
                y = (R+r*cos(u))*sin(v)
                z = r*sin(u)
                point += [x,y,z]
                count -= 1
        points.append(point)
    cloud = np.array(points)
    return cloud

def implicit_sample(n, dimension, maxima, thickness, function):
    """Samples points randomly in [0,x1_max]*...*[0,xd_max] on
    the implicit manifold defined by the equation f = 0.

    Arguments:
    n -- number of points sampled
    dimension -- ambient dimension
    maxima -- [x1_max, x2_max, ..., xd_max]
    thickness -- detection range (smaller values give more
                 precise results but take longer to compute)
    function -- a function taking "dimension" numbers and
                returning "dimension+1" numbers :
                f(x), df/dx1(x), ..., df/dxd(x)
                where f is a map from R^dimension to R
    """
    cloud = np.zeros([n,dimension])
    count = 0
    while count < n:
        point=[]
        random.seed(time.time()+count)
        for j in range(dimension):
            point.append(random.random()*maxima[j])
        f, grad = function(point)
        norm = squared_norm(grad)
        if f*f<norm*thickness*thickness:
            for j in range(dimension):
                point[j] -= grad[j]*f/norm
            cloud[count]=point
            count += 1
    return cloud

#######################################
# Implicit functions                  #
#######################################

def schwarz(p):
    """Implicit function of the Schwarz surface."""
    p = [(x*pi/2)-pi for x in p]
    f = cos(p[0])+cos(p[1])+cos(p[2])
    gx = -sin(p[0])*pi/2
    gy = -sin(p[1])*pi/2
    gz = -sin(p[2])*pi/2
    return[f,[gx,gy,gz]]

# 2D Perlin noise

def perlin_2d(grid_size, x_max, y_max, seed):
    """Noisy function in R^2.

    Arguments:
    grid_size -- size of the random pattern
    x_max -- 
    """
    def function(point):
        return perlin_2d_explicit(
            point[0], 
            point[1], 
            grid_size, 
            x_max, 
            y_max, 
            seed)
    return function

def perlin_2d_explicit(x, y, grid_size, x_max, y_max, seed):
    grid_x = floor(x/grid_size)
    grid_y = floor(y/grid_size)
    frac_x = (x%grid_size)/grid_size
    frac_y = (y%grid_size)/grid_size
    a = perlin_2d_rand(grid_x  , grid_y  , x_max, y_max, seed)
    b = perlin_2d_rand(grid_x+1, grid_y  , x_max, y_max, seed)
    c = perlin_2d_rand(grid_x  , grid_y+1, x_max, y_max, seed)
    d = perlin_2d_rand(grid_x+1, grid_y+1, x_max, y_max, seed)
    r = perlin_2d_evaluate(a, b, c, d, frac_x, frac_y)
    dx, dy = perlin_2d_gradient(a, b, c, d, frac_x, frac_y)
    return r-0.5, [dx, dy]

def perlin_2d_evaluate(a, b, c, d, x, y):
    r = interpolate(interpolate(a,b,x),interpolate(c,d,x),y)
    return r

def perlin_2d_gradient(a, b, c, d, x, y):
    dx = 6*(d-c)*(x-x*x)*(3.*y*y-2.*y*y*y) + 6*(b-a)*(x-x*x)*(1-3*y*y+2*y*y*y)
    dy = 6*(d-b)*(y-y*y)*(3.*x*x-2.*x*x*x) + 6*(c-a)*(y-y*y)*(1-3*x*x+2*x*x*x)
    return [dx,dy]

def perlin_2d_rand(x, y, x_max, y_max, seed):
    random.seed(hash((x,y,seed)))
    r = random.random()
    if r<0.5:
        r -= 0.1
    else:
        r += 0.1
    if (x<=0) or (y<=0) or (x>=x_max-1) or (y>=y_max) or (x%2==0 and y%2==0):
        r=0
    elif (x%2==1 and y%2==1):
        r=1
    return r