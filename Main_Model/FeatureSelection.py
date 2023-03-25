import numpy as np
import pandas as pd
import warnings
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

warnings.filterwarnings("ignore")

def selectfeature(Y_Feature,Num):
    abs_Y_Feature = abs(Y_Feature)
    # print('abs_Y_Feature = ',abs_Y_Feature)
    ind = np.argsort(-abs_Y_Feature)
    Y_Feature_reverse = abs_Y_Feature[np.argsort(-abs_Y_Feature)]
    count = 0
    for i in range(0,Num):
        if Y_Feature_reverse[i] > 0:
            count +=1
        else:
            break
    number = count
    index = ind[0:count]
    return number,index