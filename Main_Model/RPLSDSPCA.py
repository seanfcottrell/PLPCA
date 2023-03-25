import numpy as np
import pandas as pd
import warnings
from scipy import linalg
import os
import time
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import roc_auc_score,confusion_matrix,normalized_mutual_info_score,matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from constructW import constructW

warnings.filterwarnings("ignore")


def rbf(dist, t):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist**2 / t))

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

def heat_kernel_cal_rbf_dist(data, n_neighbors, t):
    dist = Eu_dis(data)
    # print('dist : ', dist)
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors]
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = rbf_dist[i, index_L[j]]
            #W_L[i, index_L[j]] = 1
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

def norm_laplace(data, L):
    N = data.shape[0]
    D = np.zeros_like(data)
    for i in range(N):
        D[i, i] = np.sum(data[i])
    norm_L = linalg.fractional_matrix_power(D, -1/2) * L * linalg.fractional_matrix_power(D, -1/2) #normalized Laplacian
    return norm_L

def cal_persistent_laplace(W_L, alpha1, alpha2, alpha3, alpha4,
                            alpha5, alpha6):
    n = W_L.shape[0]
    np.fill_diagonal(W_L,0)

    L = cal_laplace(W_L)
    #print("Laplace: ", L)

    np.fill_diagonal(L, 1000000000) #Make sure diagonal is excluded from maximal and minimal value consideration
    min_l = np.min(L[np.nonzero(L)]) #Establish Min Value
    #print("min: ", min_l)
    np.fill_diagonal(L, -1000000000)
    max_l = np.max(L[np.nonzero(L)]) #Establish Max Value
    #print("max: ", max_l)

    d = max_l - min_l
    #print("d: ", d)

    L = cal_laplace(W_L)
    PL = np.zeros((7,n,n))
    for k in range(1,7):
        PL[k,:,:] = np.where(L < (k/6*d + min_l), 1, 0) 
        #print("Threshold for k = ", k, ": ", k/6*d + min_l)
        np.fill_diagonal(PL[k,:,:],0)
        PL[k,:,:] = cal_laplace(PL[k,:,:])
        #print(PL[k,:,:])

    P_L = alpha1 * PL[1] + alpha2 * PL[2] + alpha3 * PL[3] + alpha4 * PL[4] + alpha5 * PL[5] + alpha6 * PL[6]
     
    return P_L

def RLSDSPCA_Algorithm(xMat,bMat,laplace,zeta,beta,gamma,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)
    V = np.eye(n)
    vMat = np.mat(V)
    E = np.ones((xMat.shape[0],xMat.shape[1]))
    E = np.mat(E)
    C = np.ones((xMat.shape[0],xMat.shape[1]))
    C = np.mat(C)
    laplace = np.mat(laplace)
    miu = 1
    for m in range(0, 10):
        Z = (-(miu/2) * ((E - xMat + C/miu).T * (E - xMat + C/miu))) - (zeta * bMat.T * bMat) + beta * vMat + gamma * laplace
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
        Y = (xMat - E - C/miu) * qMat
        # cal A
        A = bMat * qMat
        # cal AA
        AA = xMat - Y * qMat.T - C/miu
        # cal E
        for i in range(E.shape[1]):
            E[:,i] = (np.max((1 - 1.0 / (miu * np.linalg.norm(AA[:,i]))),0)) * AA[:,i]
        # cal C
        C = C + miu * (E - xMat + Y * qMat.T)
        # cal miu
        miu = 1.2 * miu

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def RPLSDSPCA_cal_projections(X_data,B_data,zeta1, beta1, gamma1, k_d, 
                                alpha1, alpha2, alpha3, alpha4,
                                alpha5, alpha6):
    # nclass = 4
    nclass = B_data.shape[1]
    # print('B_data.shape = ', B_data.shape[1])
    n = len(X_data)
    dist = Eu_dis(X_data)
    max_dist = np.max(dist)
    W_L = heat_kernel_cal_rbf_dist(X_data, n_neighbors=9, t=max_dist**(1.5))
    #W_L = heat_kernel_cal_rbf_dist(X_data, n_neighbors=9, t=1)
    # print(type(np.array(X_data)))
    # W_L = constructW(np.array(X_data), NeighborMode='KNN', WeightMode='Binary', k=9, t=1)
    R = W_L
    M = cal_persistent_laplace(R, alpha1, alpha2, alpha3, alpha4,
                                alpha5, alpha6)
    Y = RLSDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), M, zeta1, beta1, gamma1, k_d, nclass, n)
    return Y