import math
from scipy.special import binom



def cdf(n, x):

    if x < 0:
        return 0

    if x > n:
        return 1

    int_x = int(x) #integer part

    fact_n = float(math.factorial(n))

    sum_terms = 0

    for k in range(int_x + 1):
        sum_terms += (-1)**k * binom(n,k) * (x - k)**n  


    return sum_terms / fact_n
