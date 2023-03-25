import numpy as np
import warnings

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

def SDSPCA_Algorithm(xMat,bMat,alpha,beta,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)  # (4,3)
    V = np.eye(n)  # (500, 500)
    vMat = np.mat(V)  # (500, 500)
    for m in range(0, 10):
        Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat # (643, 643)
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)  # (643, 3)
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)  # (643, 643)
        vMat = np.mat(VV)  # (643, 643)
        qMat = np.mat(Q)  # (643, 3)
        Y = xMat * qMat  # (20502, 3)
        A = bMat * qMat  # (4, 3)

        # obj1 = (np.linalg.norm(xMat - Y * qMat.T, ord='fro')) ** 2 + alpha * (
        #     np.linalg.norm(bMat - A * qMat.T, ord='fro')) ** 2 + beta * np.trace(qMat.T * vMat * qMat) + garma * np.trace(qMat.T * laplace * qMat)
        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def SDSPCA_cal_projections(X_data,B_data,alpha1, beta1, k_d):
    # nclass = 2
    nclass = B_data.shape[1]
    n = len(X_data)  # 500
    Y = SDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), alpha1, beta1, k_d, nclass, n)
    return Y
