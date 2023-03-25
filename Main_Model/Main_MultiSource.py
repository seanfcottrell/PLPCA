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
from RLSDSPCA import RLSDSPCA_cal_projections
from RPLSDSPCA import RPLSDSPCA_cal_projections
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    CHOL_sample_filepath = rootPath + "/data/CHOL_sample.csv"
    CHOL_sample = pd.read_csv(CHOL_sample_filepath)
    CHOL_sample = CHOL_sample.values

    HNSCC_sample1_filepath = rootPath + "/data/HNSCC_sample1.csv"
    HNSCC_sample1 = pd.read_csv(HNSCC_sample1_filepath)
    HNSCC_sample1 = HNSCC_sample1.values

    HNSCC_sample2_filepath = rootPath + "/data/HNSCC_sample2.csv"
    HNSCC_sample2 = pd.read_csv(HNSCC_sample2_filepath)
    HNSCC_sample2 = HNSCC_sample2.values

    HNSCC_sample3_filepath = rootPath + "/data/HNSCC_sample3.csv"
    HNSCC_sample3 = pd.read_csv(HNSCC_sample3_filepath)
    HNSCC_sample3 = HNSCC_sample3.values

    HNSCC_sample4_filepath = rootPath + "/data/HNSCC_sample4.csv"
    HNSCC_sample4 = pd.read_csv(HNSCC_sample4_filepath)
    HNSCC_sample4 = HNSCC_sample4.values

    HNSCC_sample5_filepath = rootPath + "/data/HNSCC_sample5.csv"
    HNSCC_sample5 = pd.read_csv(HNSCC_sample5_filepath)
    HNSCC_sample5 = HNSCC_sample5.values

    PAAD_sample1_filepath = rootPath + "/data/PAAD_sample1.csv"
    PAAD_sample1 = pd.read_csv(PAAD_sample1_filepath)
    PAAD_sample1 = PAAD_sample1.values

    PAAD_sample2_filepath = rootPath + "/data/PAAD_sample2.csv"
    PAAD_sample2 = pd.read_csv(PAAD_sample2_filepath)
    PAAD_sample2 = PAAD_sample2.values

    Normal_sample_filepath = rootPath + "/data/Normal_sample.csv"
    Normal_sample = pd.read_csv(Normal_sample_filepath)
    Normal_sample = Normal_sample.values

    X_original_G_sample = np.vstack((HNSCC_sample1, HNSCC_sample2, HNSCC_sample3, HNSCC_sample4, HNSCC_sample5,
                                     CHOL_sample, PAAD_sample1, PAAD_sample2, Normal_sample))
    X_original = X_original_G_sample
    # X_original_G_sample_PD = pd.DataFrame(X_original_G_sample)
    # datapath1 = rootPath + "/data/X_original_GAI1.csv"
    # X_original_G_sample_PD.to_csv(datapath1, index=False)

    # X_filepath = rootPath + "/data/X_original_G.csv"
    # # X_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\X_original_G.csv'
    # X_original = pd.read_csv(X_filepath)
    # X_original = X_original.values
    sc = MinMaxScaler()
    # fit_transform Parameters
    # ----------
    # X: numpy array of shape[n_samples, n_features]
    X_original = sc.fit_transform(X_original)
    X_original = X_original.transpose()

    Y_filepath = rootPath + "/data/gnd4class_4_GAI.csv"
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

    alpha1 = .5
    alpha2 = 1
    alpha3 = 0
    alpha4 = 5
    alpha5 = 1
    alpha6 = 5

    alpha = float(sys.argv[1]) #1e4
    beta = float(sys.argv[2]) #.5
    gamma = float(sys.argv[3]) #1e1

    # RPLSDSPCA
    print("======================================RPLSDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=143, random_state=per)
            Y_train_pro = RPLSDSPCA_cal_projections(x_train, y_train, alpha, beta, gamma, k,
                                                    alpha1, alpha2, alpha3, alpha4,
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
    write_file = open('/mnt/home/cottre61/THESIS/parameter_tuning_results/RPLSDSPCA_accuracy_multisource_alphabetagamma.csv','a+')
    write_file.write('f1 test, k = 6: Alpha: %.3f, Beta: %.3f, Gamma: %.3f, acc: %.6f\n'%(alpha,beta,gamma,accuracy_mean))
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
    write_file = open('/mnt/home/cottre61/THESIS/parameter_tuning_results/RPLSDSPCA_f1_multisource_alphabetagamma.csv','a+')
    write_file.write('k = 6: Alpha: %.3f, Beta: %.3f, Gamma: %.3f, f1: %.6f\n'%(alpha,beta,gamma,f1_mean))
    write_file.close()
