import numpy as np
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

def RgLPCA_Algorithm(xMat,laplace,garma,k):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    # A = np.random.rand(c, k)  # (4,3)
    # V = np.eye(n)  # (500, 500)
    # I = np.eye(n)
    # I = np.mat(I)
    # vMat = np.mat(V)  # (500, 500)
    E = np.ones((xMat.shape[0],xMat.shape[1]))
    E = np.mat(E)
    C = np.ones((xMat.shape[0],xMat.shape[1]))
    C = np.mat(C)
    laplace = np.mat(laplace)
    miu = 1
    for m in range(0, 10):
        Z = (-(miu/2) * ((E - xMat + C/miu).T * (E - xMat + C/miu)))  + garma * laplace  # (643, 643)
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)  # (643, 3)
        # q = np.linalg.norm(Q, ord=2, axis=1)
        # qq = 1.0 / (q * 2)
        # VV = np.diag(qq)  # (643, 643)
        # vMat = np.mat(VV)  # (643, 643)
        qMat = np.mat(Q)  # (643, 3)
        Y = (xMat - E - C/miu) * qMat  # (20502, 3)
        A = xMat - Y * qMat.T - C/miu
        for i in range(E.shape[1]):
            E[:,i] = (np.max((1 - 1.0 / (miu * np.linalg.norm(A[:,i]))),0)) * A[:,i]
        C = C + miu * (E - xMat + Y * qMat.T)
        miu = 1.2 * miu

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def RgLPCA_cal_projections(X_data,gamma1, k_d):
    # nclass = 2
    # nclass = B_data.shape[1]
    n = len(X_data)  # 500
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    # W_L = constructW(np.array(X_data), NeighborMode='KNN', WeightMode='Binary', k=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y = RgLPCA_Algorithm(X_data.transpose(), M, gamma1, k_d)
    return Y
