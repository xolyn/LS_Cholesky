# Import some necessary packages:
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.random import default_rng
import matplotlib.pyplot as plt
# Set random number seeds for deviation:
rng = default_rng(443)

def backsubs(U,b):
    m = b.shape[0]
    x = np.zeros(m)
    for i in reversed(range(m)):
        x[i] = (b[i] - np.dot(U[i,i+1:m],x[i+1:m]))/U[i,i]
    return x

def forwardsubs(L,b):
    m = b.shape[0]
    x = np.zeros(m)
    for i in range(m):
        x[i] = (b[i] - np.dot(L[i,0:i],x[0:i]))/L[i,i]
    return x

def cholesky(B):
    n = B.shape[0] # number of rows
    L = np.zeros((n, n)) # initiallization of L
    for j in range(n):
        L[j,0:j] = forwardsubs(L[0:j,0:j],B[j,0:j])
        L[j,j] = np.sqrt(B[j,j] - LA.norm(L[j,0:j])**2)
    return L

def ols_by_chol(A, y):
    L = cholesky(A.T @ A)
    z = forwardsubs(L, A.T @ y)
    return backsubs(L.T, z)

