import numpy as np
import matplotlib.pyplot as plt

#################################################################################################################
### Other Functions

def CreateAMatrix(X_matrix):
    if X_matrix.ndim < 2:
        X_matrix = X_matrix.reshape(-1,1)
    A_matrix = np.hstack((X_matrix, np.ones([X_matrix.shape[0],1])))
    return A_matrix

def CreatezVector(w_vector,b):
    z_vector = np.append(w_vector,b).reshape(-1,1)
    return z_vector

def LoadTrainingAndTestData(filename_X_train,filename_X_test,filename_y_train,filename_y_test):
    X_train = np.genfromtxt(filename_X_train, delimiter=',')
    X_test = np.genfromtxt(filename_X_test, delimiter=',')
    y_train = np.genfromtxt(filename_y_train, delimiter=',')
    y_test = np.genfromtxt(filename_y_test, delimiter=',') 
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return X_train,X_test,y_train,y_test

def LinearRegressionPredictor(X_train,X_test,y_train,y_test):
    A_matrix_train = CreateAMatrix(X_train)
    A_matrix_test = CreateAMatrix(X_test)
    z_vector_LinR = np.linalg.solve(np.dot(A_matrix_train.T,A_matrix_train),np.dot(A_matrix_train.T,y_train))
    y_pred_LinR = np.dot(A_matrix_test,z_vector_LinR)
    mse_LinR = np.mean((y_test-y_pred_LinR)**2)
    return z_vector_LinR,y_pred_LinR,mse_LinR

def QuickSelect(X_test_distances_from_X_train_shuffled,k):
    if X_test_distances_from_X_train_shuffled.shape[0] == 1:
        return X_test_distances_from_X_train_shuffled[0]
    pivot = X_test_distances_from_X_train_shuffled[X_test_distances_from_X_train_shuffled.shape[0]//2]
    pivots_list = X_test_distances_from_X_train_shuffled[X_test_distances_from_X_train_shuffled == pivot]
    lower_than_pivot_list = X_test_distances_from_X_train_shuffled[X_test_distances_from_X_train_shuffled < pivot]
    higher_than_pivot_list = X_test_distances_from_X_train_shuffled[X_test_distances_from_X_train_shuffled > pivot]
    if k < lower_than_pivot_list.shape[0]:
        return QuickSelect(lower_than_pivot_list,k)
    elif k < lower_than_pivot_list.shape[0]+pivots_list.shape[0]:
        return pivots_list[0]
    else:
        return QuickSelect(higher_than_pivot_list,k-len(lower_than_pivot_list)-len(pivots_list))

def FindkthSmallestDistance(X_test_distances_from_X_train,k):
    X_test_distances_from_X_train_shuffled = np.random.permutation(X_test_distances_from_X_train)
    kth_smallest_distance = QuickSelect(X_test_distances_from_X_train_shuffled,k)
    return kth_smallest_distance

def MostCommonLabel(KNN_labels):
    unique_labels_list, count_of_labels_list = np.unique(KNN_labels, return_counts=True)
    most_common_label = unique_labels_list[np.argmax(count_of_labels_list)]
    return most_common_label

def PredictUsingKNN(X_train,X_test_point,y_train,K):
    X_test_distances_from_X_train = np.linalg.norm(X_test_point-X_train,axis=1)
    kth_smallest_distance = FindkthSmallestDistance(X_test_distances_from_X_train,K)
    kth_smallest_distances_indices = np.where(X_test_distances_from_X_train <= kth_smallest_distance)[0]
    if kth_smallest_distances_indices.shape[0] > K:
        kth_smallest_distances_indices_sorted = kth_smallest_distances_indices[np.argsort(X_test_distances_from_X_train[kth_smallest_distances_indices])]
        kth_smallest_distances_indices = kth_smallest_distances_indices_sorted[:K]
    KNN = X_train[kth_smallest_distances_indices]
    KNN_labels = y_train[kth_smallest_distances_indices]
    y_test_point = np.mean(KNN_labels)
    return y_test_point

def KNNClassifier(X_train,X_test,y_train,y_test,K_list):
    X_test_list = X_test.tolist()
    for i in range(K_list.shape[0]):
        K = K_list[i]
        y_pred_KNN = np.array([[PredictUsingKNN(X_train,X_test_point,y_train,K)] for X_test_point in X_test_list])
        if i == 0:
            list_of_y_pred_KNN = []
            list_of_mse_KNN = []
        list_of_y_pred_KNN.append(np.array([[PredictUsingKNN(X_train,X_test_point,y_train,K)] for X_test_point in X_test_list]))
        list_of_mse_KNN.append(np.mean((y_test-y_pred_KNN)**2))
    return list_of_y_pred_KNN,list_of_mse_KNN

def PlotMSE(mse_LinR,list_of_mse_KNN,K_list,data_type_str):
    plt.axhline(y=mse_LinR,color='green',label='Linear Regression')
    plt.plot(K_list,list_of_mse_KNN,color='blue',label='KNN')
    plt.xlabel('K value')
    plt.ylabel('MSE')
    plt.title('MSE vs K value For Data Type '+data_type_str)
    plt.legend()
    plt.show()
    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train_F = 'Data/X_train_F.csv'
filename_X_test_F = 'Data/X_test_F.csv'
filename_y_train_F = 'Data/y_train_F.csv'
filename_y_test_F = 'Data/y_test_F.csv'
K_list = np.arange(1,10)

#################################################################################################################
### Main Code

X_train_F,X_test_F,y_train_F,y_test_F = LoadTrainingAndTestData(filename_X_train_F,filename_X_test_F,filename_y_train_F,filename_y_test_F)
z_vector_LinR_F,y_pred_LinR_F,mse_LinR_F = LinearRegressionPredictor(X_train_F,X_test_F,y_train_F,y_test_F)
list_of_y_pred_KNN_F,list_of_mse_KNN_F = KNNClassifier(X_train_F,X_test_F,y_train_F,y_test_F,K_list)
PlotMSE(mse_LinR_F,list_of_mse_KNN_F,K_list,'F')

