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
from RLSDSPCA import RLSDSPCA_cal_projections
from RPLSDSPCA import RPLSDSPCA_cal_projections
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    COAD_sample1_filepath = rootPath + "/data/COAD_sample1.csv"
    COAD_sample1 = pd.read_csv(COAD_sample1_filepath)
    COAD_sample1 = COAD_sample1.values

    COAD_sample2_filepath = rootPath + "/data/COAD_sample2.csv"
    COAD_sample2 = pd.read_csv(COAD_sample2_filepath)
    COAD_sample2 = COAD_sample2.values

    COAD_sample3_filepath = rootPath + "/data/COAD_sample3.csv"
    COAD_sample3 = pd.read_csv(COAD_sample3_filepath)
    COAD_sample3 = COAD_sample3.values

    COAD_sample4_filepath = rootPath + "/data/COAD_sample4.csv"
    COAD_sample4 = pd.read_csv(COAD_sample4_filepath)
    COAD_sample4 = COAD_sample4.values

    X_original_G_sample = np.vstack((COAD_sample1, COAD_sample2, COAD_sample3, COAD_sample4))
    X_original = X_original_G_sample
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

    Y_filepath = rootPath + "/data/COADSampleCategory1.csv"
    # Y_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\gnd4class_4_GAI.csv'
    Y_gnd4class4 = pd.read_csv(Y_filepath)
    Y_gnd4class4 = Y_gnd4class4.values.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4)
    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []
    knc = KNeighborsClassifier(n_neighbors=1)

    # RPLSDSPCA
    print("======================================RPLSDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()

    #float(sys.argv[1])
    alpha1 = 0.5
    alpha2 = 5
    alpha3 = 2
    alpha4 = 2
    alpha5 = 2
    alpha6 = 2
    
    alpha = 1e5
    beta = 60
    gamma = 3

    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0 

        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = RPLSDSPCA_cal_projections(x_train, y_train, alpha, beta, gamma, k, alpha1, alpha2, alpha3, alpha4,
                                                    alpha5, alpha6)

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
