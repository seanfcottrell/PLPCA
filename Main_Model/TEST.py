import numpy as np 
import sys

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

def normalize_laplacian(L):
    N = L.shape[0]
    A = np.copy(L)
    np.fill_diagonal(A,0)
    A = -1*A
    D = np.zeros_like(L)
    for i in range(N):
        D[i, i] = np.sum(A[i]) + 1e-15
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
    norm_L = D_inv_sqrt * L * D_inv_sqrt # Normalized Laplacian
    return norm_L

def cal_persistent_laplace(W_L, alpha1, alpha2, alpha3, alpha4,
                            alpha5, alpha6, alpha7, alpha8,alpha9, alpha10):
    n = W_L.shape[0]
    np.fill_diagonal(W_L,0)
    print("Weighted Adjacency:", W_L)

    L = cal_laplace(W_L)
    print("Weighted Laplacian:", L)

    np.fill_diagonal(L, 1000000000) #Make sure diagonal is excluded from maximal and minimal value consideration
    min_l = np.min(L[np.nonzero(L)]) #Establish Min Value
    print("min:", min_l)
    np.fill_diagonal(L, -1000000000)
    max_l = np.max(L[np.nonzero(L)]) #Establish Max Value
    print("max:", max_l)

    d = max_l - min_l
    print("d:", d)

    L = cal_laplace(W_L)
    PL = np.zeros((11,n,n))
    for k in range(1,11):
        PL[k,:,:] = np.where(L > (k/10*d + min_l), 0, 1) #need to only account for NONZERO ELEMENTS SO THIS IS WRONG
        np.fill_diagonal(PL[k,:,:],0)
        print("Threshold: ", k/10*d + min_l)
        PL[k,:,:] = cal_laplace(PL[k,:,:])
    
        print(PL[k,:,:] == PL[k-1,:,:])

    P_L = alpha1 * PL[1] + alpha2 * PL[2] + alpha3 * PL[3] + alpha4 * PL[4] + alpha5 * PL[5] + alpha6 * PL[6] + alpha7 * PL[7]
    + alpha8 * PL[8] + alpha9 * PL[9] + alpha10 * PL[10]
     
    return P_L

import pandas as pd
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler

alpha1 = 0.1
alpha2 = 0.2
alpha3 = 0.3
alpha4 = 0.4
alpha5 = 0.5
alpha6 = 0.6
alpha7 = 0.7
alpha8 = 0.8
alpha9 = 0.9
alpha10=1
#alpha10 = float(sys.argv[1])

import warnings
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #COAD_sample1_filepath = rootPath + "/data/COAD_sample1.csv"
    #COAD_sample1 = pd.read_csv(COAD_sample1_filepath)
    #COAD_sample1 = COAD_sample1.values

    #COAD_sample2_filepath = rootPath + "/data/COAD_sample2.csv"
    #COAD_sample2 = pd.read_csv(COAD_sample2_filepath)
    #COAD_sample2 = COAD_sample2.values

    #COAD_sample3_filepath = rootPath + "/data/COAD_sample3.csv"
    #COAD_sample3 = pd.read_csv(COAD_sample3_filepath)
    #COAD_sample3 = COAD_sample3.values

    COAD_sample4_filepath = rootPath + "/data/COAD_sample4.csv"
    COAD_sample4 = pd.read_csv(COAD_sample4_filepath)
    COAD_sample4 = COAD_sample4.values

    #X_original_G_sample = np.vstack((COAD_sample1, COAD_sample2, COAD_sample3, COAD_sample4))
    #X_original = X_original_G_sample
    X_original = COAD_sample4
    # X_original_G_sample_PD = pd.DataFrame(X_original_G_sample)
    # datapath1 = rootPath + "/data/COAD_sample.csv"
    # X_original_G_sample_PD.to_csv(datapath1, index=False)

    # X_filepath = rootPath + "/data/COAD_GE.csv"
    # X_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\X_original_G.csv'
    # X_original = pd.read_csv(X_filepath)
    # X_original = X_original.values
    sc = MinMaxScaler()
    # fit_transform Parameters
    # ----------
    # X: numpy array of shape[n_samples, n_features]
    X_original = sc.fit_transform(X_original)
    X_original = X_original.transpose()
    X = np.mat(X_original)

    n = len(X)
    dist = Eu_dis(X)
    max_dist = np.max(dist)
    W_L = heat_kernel_cal_rbf_dist(X, n_neighbors=9, t=max_dist)
    #W_L = heat_kernel_cal_rbf_dist(X_data, n_neighbors=9, t=1.2)
    # print(type(np.array(X_data)))
    # W_L = constructW(np.array(X_data), NeighborMode='KNN', WeightMode='Binary', k=9, t=1)
    R = W_L
    cal_persistent_laplace(R, alpha1, alpha2, alpha3, alpha4,
                        alpha5, alpha6, alpha7, alpha8,alpha9, alpha10)
