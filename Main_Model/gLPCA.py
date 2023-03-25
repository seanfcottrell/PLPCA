import numpy as np
import pandas as pd
import warnings
from constructW import constructW

warnings.filterwarnings("ignore")


def rbf(dist, t):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist / t))

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    dist_mat = np.asarray(dist_mat)
    return dist_mat

def cal_rbf_dist(data, n_neighbors, t):
    dist = Eu_dis(data)
    # print('dist : ', dist)
    n = dist.shape[0]
    # rbf_dist = rbf(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors]
        len_index_L = len(index_L)
        for j in range(len_index_L):
            # W_L[i, index_L[j]] = rbf_dist[i, index_L[j]]
            W_L[i, index_L[j]] = 1
    # W_L = np.multiply(W_L, (W_L > W_L.transpose())) + np.multiply(W_L.transpose(), (W_L.transpose() >= W_L))
    W_L = np.maximum(W_L, W_L.transpose())
    return W_L

def cal_laplace(data):
    N = data.shape[0]
    H = np.zeros_like(data)
    for i in range(N):
        H[i, i] = np.sum(data[i])
    L = H - data  # Laplacian
    return L

def gLPCA_Algorithm(xMat,laplace,garma,k,n):
    Z = -(xMat.T * xMat) + garma * laplace  # (643, 643)
    Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
    eigValIndice = np.argsort(Z_eigVals)
    n_eigValIndice = eigValIndice[0:k]
    n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
    Q = np.array(n_Z_eigVect)  # (643, 3)
    qMat = np.mat(Q)  # (643, 3)
    Y = xMat * qMat  # (20502, 3)
    return Y

def gLPCA_cal_projections(X_data, gamma1, k_d):
    # nclass = 2
    # nclass = B_data.shape[1]
    n = len(X_data)  # 500
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    # W_L = constructW(np.array(X_data), NeighborMode='KNN', WeightMode='Binary', k=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y= gLPCA_Algorithm(X_data.transpose(), M, gamma1, k_d, n)
    return Y
