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

def LSDSPCA_Algorithm(xMat,bMat,laplace,alpha,beta,gamma,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)  # (4,3)
    V = np.eye(n)  # (500, 500)
    vMat = np.mat(V)  # (500, 500)
    for m in range(0, 10):
        Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat + gamma * laplace  # (643, 643)
        # cal Q
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)
        # cal V
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)
        vMat = np.mat(VV)
        qMat = np.mat(Q)
        # cal Y
        Y = xMat * qMat
        # cal A
        A = bMat * qMat

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def LSDSPCA_cal_projections(X_data,B_data,alpha1, beta1, gamma1,k_d):
    # nclass = 2
    nclass = B_data.shape[1]
    n = len(X_data)
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    # W_L = constructW(np.array(X_data), NeighborMode='KNN', WeightMode='Binary', k=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y = LSDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), M, alpha1, beta1, gamma1, k_d, nclass, n)
    return Y