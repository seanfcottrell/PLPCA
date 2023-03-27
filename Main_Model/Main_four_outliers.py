import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from constructW import constructW
from PCA import PCA_cal_projections
from gLPCA import gLPCA_cal_projections
from gLSPCA import gLSPCA_cal_projections
from RgLPCA import RgLPCA_cal_projections
from SDSPCA import SDSPCA_cal_projections
from LSDSPCA import LSDSPCA_cal_projections
from RPLSDSPCA import RPLSDSPCA_cal_projections
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")
def trans(yy):
    nn = len(yy)
    yy_array = np.zeros((nn,2))
    for i in range(nn):
        if (yy[i] == 1):
            yy_array[i, 1] = 1
        else:
            yy_array[i, 0] = 1
    return yy_array
if __name__ == '__main__':
    X_filepath = rootPath + "/data/gaussxor_four_outliers.csv"
    X_original_data = pd.read_csv(X_filepath)
    X_original_data = X_original_data.values
    # print(X_original_data)
    X_original = X_original_data[:, 1:]
    X_original = X_original.transpose()
    Y_gnd4class4 = X_original_data[:, 0]
    # print(X_original.shape)
    # print(Y_gnd4class4)
    Y_gnd4class4_array = trans(Y_gnd4class4)
    Y_gnd4class4_array = Y_gnd4class4_array.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4_array)

    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []
    knc = KNeighborsClassifier(n_neighbors=1)

    # RPLSDSPCA
    alpha1 = 3
    alpha2 = 2 
    alpha3 = 1.5
    alpha4 = 3
    alpha5 = 1
    alpha6 = 3 
    print("======================================RPLSDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = RPLSDSPCA_cal_projections(x_train, y_train, 1e-5, 1e4, 0.1, k,
                                                    alpha1,
                                                    alpha2, alpha3, alpha4, alpha5, alpha6)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('RPLSDSPCA accuracy_mean = ', accuracy_mean)
    print('RPLSDSPCA accuracy_var = ', accuracy_var)
    print('RPLSDSPCA accuracy_std = ', accuracy_std)
    write_file = open('/mnt/home/cottre61/THESIS/parameter_tuning_results/RPLSDSPCA_accuracy_four_outliers.csv','a+')
    write_file.write('k = 6: Alpha1: %.2f, Alpha2: %.2f, Alpha3: %.2f, Acc: %.6f\n'%(alpha1,alpha2,alpha3,accuracy_mean))
    write_file.close()

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('RPLSDSPCA precision_mean = ', precision_mean)
    print('RPLSDSPCA precision_var = ', precision_var)
    print('RPLSDSPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('RPLSDSPCA recall_mean = ', recall_mean)
    print('RPLSDSPCA recall_var = ', recall_var)
    print('RPLSDSPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('RPLSDSPCA f1_mean = ', f1_mean)
    print('RPLSDSPCA f1_var = ', f1_var)
    print('RPLSDSPCA f1_std = ', f1_std)

