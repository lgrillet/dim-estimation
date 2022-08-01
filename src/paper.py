import scipy.integrate as integrate
from scipy.special import gamma
from math import pi, sqrt, sin, sinh, asin, log

# returns the lower and upper bounds for the volume of a ball
# d : dimension
# r : radius
def volume_ball(d,r):
    lower = integrate.quad(lambda x: pow(sin(x),d-1)                 , 0, r)[0]
    upper = integrate.quad(lambda x: pow(sinh(sqrt(2)*x)/sqrt(2),d-1), 0, 2*asin(r/2))[0]
    const = 2*pow(pi,d/2)/(gamma(d/2))
    lower *= const
    upper *= const
    return [lower, upper]

# returns the gap between the range of possible mean of X and its nearest d+0.5 or d-0.5
# d : dimension
# e1 > e2 : scales
def gap(d,e1,e2):
    lower = (e1/(2*asin(e2/2))) * pow(sin(e1)/(sin(2*asin(e2/2))),d-1)
    upper1 = integrate.quad(lambda x: pow(sinh(x),d-1), 0, sqrt(2)*2*asin(e1/2))[0]
    upper2 = integrate.quad(lambda x: pow(sinh(x),d-1), 0, sqrt(2)*e2)[0]
    upper = upper1/upper2
    k = e2/e1
    lower_mean = log(lower)/log(1/k)
    upper_mean = log(upper)/log(1/k)
    if upper_mean > d+0.5 or lower_mean < d-0.5:
        return -1
    else:
        return min(d+0.5-upper_mean, lower_mean-(d-0.5))

# returns a lower bound for the number of poins needed
# d : dimension
# e1 > e2 : scale2
# V : Volume of the manifold
# delta : the probability of succes is at least 1-delta
# alpha : as in Corollary XXX
def estimate(d,e1,e2,V,delta,alpha):
    L1, U1 = volume_ball(d,e1)
    L2, U2 = volume_ball(d,e2)
    k = e2/e1
    g = gap(d,e1,e2)
    if g == -1:
        return float('inf')
    rho = delta * (1-pow(k, g/2))**2
    n1 = 1 + (1/( alpha   *rho)) * (((U1/L1)-1)**2)+sqrt((2/( alpha   *rho))*(V/L1))
    n2 = 1 + (1/((1-alpha)*rho)) * (((U2/L2)-1)**2)+sqrt((2/((1-alpha)*rho))*(V/L2))
    return max(n1,n2)

# optimizes the scales for given dimension and volume
# d : dimension
# e1 > e2 : starting scales
# V : Volume of the manifold
# delta : the probability of succes is at least 1-delta
# alpha : as in Corollary XXX
def optimize(d, e1, e2, V, delta, alpha):
    steps = [0.01]
    
    for step in steps:
        stop = False
        while not stop:
            stop = True

            a = estimate(d,e1,e2,V,delta,alpha)
            b = estimate(d,e1+step,e2,V,delta,alpha)
            c = estimate(d,e1-step,e2,V,delta,alpha)
            if b<a:
                e1 += step
                stop = False
            elif c<a:
                e1 -= step
                stop = False

            a = estimate(d,e1,e2,V,delta,alpha)
            b = estimate(d,e1,e2+step,V,delta,alpha)
            c = estimate(d,e1,e2-step,V,delta,alpha)
            if b<a:
                e2 += step
                stop = False
            elif c<a:
                e2 -= step
                stop = False

            a = estimate(d,e1,e2,V,delta,alpha)
            b = estimate(d,e1,e2,V,delta,alpha+step)
            c = estimate(d,e1,e2,V,delta,alpha-step)
            if b<a:
                alpha += step
                stop = False
            elif c<a:
                alpha -= step
                stop = False

    return a, e1, e2, alpha