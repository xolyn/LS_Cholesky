import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.random import default_rng
import matplotlib.pyplot as plt
from decomp import *

# Next, we will randomly some data points (here choose 100) using package rng and its function rng.normal with some deviation:
# Generate data point called `x'

n, b0, b1 = 100, 10, -2
x = np.linspace(0,10,num=n)
std = 1 + (x - 4)**2
y = b0 + b1 * x + std * rng.normal(0,1,n)
plt.scatter(x,y,alpha=0.5)
plt.show()

# Then transform their types into vectors, transpose them and stack the data points with the 1 vector
ones = np.ones((100,1))
xreshape=x.reshape((100,1))
A=np.concatenate((ones, xreshape),1)
# A
# ↑ Be careful to display this variable due to the size

# Then just do Cholesky decomposition by the function we’ve defined before:
coeff_ols = decomp.ols_by_chol(A,y)
#.reshape((100,1))
print(coeff_ols)
