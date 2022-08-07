import random
from math import cos, sin, sqrt, floor, pi
import numpy as np
import time

######################################
# Tools                              #
######################################

def squared_norm(a):
    squared_norm = 0
    for coord in a:
        squared_norm += coord*coord
    return squared_norm

def step(t):
    return (3-2*t)*t*t

def interpolate(a, b, t):
    return a + (b-a)*step(t)

######################################
# Sampler - Uniform Box              #
######################################
### Samples points uniformly in [0,x1_max]*...*[0,xd_max]
# n : number of points sampled
# dimension : ambiant dimension
# maxima : [x1_max, x2_max, ..., xd_max]

def box_sample(n, dimension, maxima):
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

######################################
# Sampler - Gaussian                 #
######################################
### Samples points from  a normal distribution
# n : number of points sampled
# dimension : ambiant dimension
# sigmas : list of standard deviations for each direction
# maxima : define bounds [-x1_max,x1_max]*...*[-xd_max,xd_max], -1 for no bound

def gaussian_sample(n, dimension, sigmas, maxima=-1):
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
    
######################################
# Sampler - Sphere                   #
######################################
### Samples points uniformly in an hypersphere
# n : number of points sampled
# dimension : dimension of the sphere (in ambiant dimension "dimension+1")
# radius : radius of the sphere
# copies : number of disjoint copies of the sphere

def sphere_sample(n, dimension, radius=1, copies=1):
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

######################################
# Sampler - Torus in 4D              #
######################################
# n : number of points sampled
# dimension : dimension of the torus (in ambiant dimension "2 * dimension")

def clifford_torus_sample(n, dimension, radii=-1):
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

######################################
# Sampler - Torus in 3D              #
######################################
# n : number of points sampled
# dimension : dimension of the torus (in ambiant dimension "3/2 * dimension")

def torus_sample(n, dimension, R=2, r=1, mute=False):
    if n%2 != 0:
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

######################################
# Experimental                       #
######################################
######################################
# Sampler - From implicit function   #
######################################
### Samples points randomly in [0,x1_max]*...*[0,xd_max] at a distance 
### 'thickness' of the implicit manifold defined by the equation function = 0.
###  Then bring those points closer to this manifold.
# maxima : [x1_max, x2_max, ..., xd_max]
# n : number of points sampled
# d : ambiant dimension d
# function : a function f taking d numbers and returning d+1 numbers :
#    f(x), df/dx1(x), ..., df/dxd(x)
# returns a numpy array of n d-dimensional points

def implicit_sample(maxima, thickness, n, dimension, function):
    start = time.time()
    print("Generating a cloud of", n, "points...")
    cloud = np.zeros([n,dimension])
    count = 0
    while count < n:
        point=[]
        random.seed(time.time()+count)
        for j in range(dimension):
            point.append(random.random()*maxima[j])
        f, grad = function(point)
        n = squared_norm(grad)
        if f*f<n*thickness*thickness:
            for j in range(dimension):
                point[j] -= grad[j]*f/n
            cloud[count]=point
            count += 1
            if (count%100==0):
                print("\r", count, "/", n, "points generated",end="")
    print("\nIt took", time.time()-start, "seconds\n")
    return cloud

#######################################
# Implicit function - 2D Perlin noise #
#######################################
# Takes parameters and returns a function
# f:(point)->(f(point),df/dx(point),df/dy(point))

def perlin_2d(grid_size, x_max, y_max, seed):
    def function(point):
        return perlin_2d_explicit(
            point[0], 
            point[1], 
            grid_size, 
            x_max, 
            y_max, 
            seed)
    return function

# Explicit function depending on the parameters and on x and y
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
    
#######################################
# Implicit function - 3D Perlin noise #
#######################################
# Takes parameters and returns a function
# f:(point)->(f(point),df/dx(point),df/dy(point),df/dz(point))

def perlin_3d(grid_size, x_max, y_max, z_max, seed):
    def function(point):
        return perlin_3d_explicit(
            point[0], 
            point[1], 
            point[2], 
            grid_size, 
            x_max, 
            y_max, 
            z_max, 
            seed)
    return function

# Explicit function depending on the parameters and on x and y
def perlin_3d_explicit(x, y, z, grid_size, x_max, y_max, z_max, seed):
    grid_x = floor(x/grid_size)
    grid_y = floor(y/grid_size)
    grid_z = floor(z/grid_size)
    frac_x = (x%grid_size)/grid_size
    frac_y = (y%grid_size)/grid_size
    frac_z = (z%grid_size)/grid_size
    a = perlin_3d_rand(grid_x  , grid_y  , grid_z  , x_max, y_max, z_max, seed)
    b = perlin_3d_rand(grid_x+1, grid_y  , grid_z  , x_max, y_max, z_max, seed)
    c = perlin_3d_rand(grid_x  , grid_y+1, grid_z  , x_max, y_max, z_max, seed)
    d = perlin_3d_rand(grid_x+1, grid_y+1, grid_z  , x_max, y_max, z_max, seed)
    e = perlin_3d_rand(grid_x  , grid_y  , grid_z+1, x_max, y_max, z_max, seed)
    f = perlin_3d_rand(grid_x+1, grid_y  , grid_z+1, x_max, y_max, z_max, seed)
    g = perlin_3d_rand(grid_x  , grid_y+1, grid_z+1, x_max, y_max, z_max, seed)
    h = perlin_3d_rand(grid_x+1, grid_y+1, grid_z+1, x_max, y_max, z_max, seed)
    r = perlin_3d_evaluate(a, b, c, d, e, f, g, h, frac_x, frac_y, frac_z)
    dx, dy, dz = perlin_3d_gradient(a, b, c, d, e, f, g, h, 
                                    frac_x, frac_y, frac_z)
    return r-0.5, [dx, dy, dz]

def perlin_3d_evaluate(a, b, c, d, e, f, g, h, x, y, z):
    r = interpolate(interpolate(interpolate(a,b,x), interpolate(c,d,x),y), \
                    interpolate(interpolate(e,f,x), interpolate(g,h,x),y),z)
    return r

def perlin_3d_gradient(a, b, c, d, e, f, g, h, x, y, z):
    dx =  6 * (b-a) * (x-x*x) * (1-3*y*y+2*y*y*y) * (1-3*z*z-2*z*z*z) \
        + 6 * (d-c) * (x-x*x) * (3*y*y-2*y*y*y)   * (1-3*z*z-2*z*z*z) \
        + 6 * (f-e) * (x-x*x) * (1-3*y*y+2*y*y*y) * (3*z*z-2*z*z*z)   \
        + 6 * (h-g) * (x-x*x) * (3*y*y-2*y*y*y)   * (3*z*z-2*z*z*z)
    dy =  6 * (c-a) * (y-y*y) * (1-3*x*x+2*x*x*x) * (1-3*z*z-2*z*z*z) \
        + 6 * (d-b) * (y-y*y) * (3*x*x-2*x*x*x)   * (1-3*z*z-2*z*z*z) \
        + 6 * (g-e) * (y-y*y) * (1-3*x*x+2*x*x*x) * (3*z*z-2*z*z*z)   \
        + 6 * (h-f) * (y-y*y) * (3*x*x-2*x*x*x)   * (3*z*z-2*z*z*z) 
    dz =  6 * (e-a) * (z-z*z) * (1-3*x*x+2*x*x*x) * (1-3*y*y-2*y*y*y) \
        + 6 * (f-b) * (z-z*z) * (3*x*x-2*x*x*x)   * (1-3*y*y-2*y*y*y) \
        + 6 * (g-c) * (z-z*z) * (1-3*x*x+2*x*x*x) * (3*y*y-2*y*y*y)   \
        + 6 * (h-d) * (z-z*z) * (3*x*x-2*x*x*x)   * (3*y*y-2*y*y*y)
    return [dx,dy,dz]

def perlin_3d_rand(x, y, z, x_max, y_max, z_max, seed):
    random.seed(hash((x,y,z,seed)))
    r = random.random()
    if r<0.5:
        r -= 0.1
    else:
        r += 0.1
    if (x<=0) or (y<=0) or (z<=0) or (x>=x_max-1) or (y>=y_max) or (z>=z_max):
        r=0
    return r

def schwarz():
    def function(p):
        f = cos(p[0])+cos(p[1])+cos(p[2])
        gx = -sin(p[0])
        gy = -sin(p[1])
        gz = -sin(p[2])
        return[f,[gx,gy,gz]]
    return function