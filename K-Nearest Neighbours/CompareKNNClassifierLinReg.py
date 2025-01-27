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
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return X_train,X_test,y_train,y_test

def CreateAdditionalTestingDatapoints(X_test,num_points):
    X_test_min = np.min(X_test)
    X_test_max = np.max(X_test)
    step_size = (X_test_max - X_test_min)/num_points
    X_test_extra_points = np.arange(X_test_min,X_test_max,step_size).reshape(-1,1)
    X_test_full_range = np.unique(np.vstack((X_test,X_test_extra_points)))
    return X_test_full_range
    
def PrepareTrainingAndTestData(filename_X_train_D,filename_X_test_D,filename_y_train_D,filename_y_test_D,filename_X_train_E,filename_X_test_E,filename_y_train_E,filename_y_test_E,num_points):
    X_train_D,X_test_D,y_train_D,y_test_D = LoadTrainingAndTestData(filename_X_train_D,filename_X_test_D,filename_y_train_D,filename_y_test_D)
    X_train_E,X_test_E,y_train_E,y_test_E = LoadTrainingAndTestData(filename_X_train_E,filename_X_test_E,filename_y_train_E,filename_y_test_E)
    X_test_full_range_D = CreateAdditionalTestingDatapoints(X_test_D,num_points)
    X_test_full_range_E = CreateAdditionalTestingDatapoints(X_test_E,num_points)
    return X_train_D,X_test_D,y_train_D,y_test_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_D,X_test_full_range_E

def LinearRegressionPredictor(X_train,X_test,y_train,y_test,X_test_full_range):
    A_matrix_train = CreateAMatrix(X_train)
    A_matrix_test = CreateAMatrix(X_test)
    z_vector_LinR = np.linalg.solve(np.dot(A_matrix_train.T,A_matrix_train),np.dot(A_matrix_train.T,y_train))
    y_pred_LinR = np.dot(A_matrix_test,z_vector_LinR)
    mse_LinR = np.mean((y_test-y_pred_LinR)**2)
    A_matrix_test_full_range = CreateAMatrix(X_test_full_range)
    y_pred_full_range = np.dot(A_matrix_test_full_range,z_vector_LinR)
    return z_vector_LinR,y_pred_LinR,mse_LinR,y_pred_full_range

def ComputeLinearRegressionPredictors(X_train_D,X_test_D,y_train_D,y_test_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_D,X_test_full_range_E):
    z_vector_LinR_D,y_pred_LinR_D,mse_LinR_D,y_pred_LinR_full_range_D = LinearRegressionPredictor(X_train_D,X_test_D,y_train_D,y_test_D,X_test_full_range_D)
    z_vector_LinR_E,y_pred_LinR_E,mse_LinR_E,y_pred_LinR_full_range_E = LinearRegressionPredictor(X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_E)
    return z_vector_LinR_D,y_pred_LinR_D,mse_LinR_D,y_pred_LinR_full_range_D,z_vector_LinR_E,y_pred_LinR_E,mse_LinR_E,y_pred_LinR_full_range_E

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

def KNNClassifier(X_train,X_test,y_train,y_test,K_list,X_test_full_range):
    X_test_list = X_test.tolist()
    X_test_list_full_range = X_test_full_range.tolist()
    for i in range(K_list.shape[0]):
        K = K_list[i]
        y_pred_KNN = np.array([[PredictUsingKNN(X_train,X_test_point,y_train,K)] for X_test_point in X_test_list])
        if i == 0:
            list_of_y_pred_KNN = []
            list_of_y_pred_KNN_full_range = []
            list_of_mse_KNN = []
        list_of_y_pred_KNN.append(np.array([[PredictUsingKNN(X_train,X_test_point,y_train,K)] for X_test_point in X_test_list]))
        list_of_y_pred_KNN_full_range.append(np.array([[PredictUsingKNN(X_train,X_test_point_full_range,y_train,K)] for X_test_point_full_range in X_test_list_full_range]))
        list_of_mse_KNN.append(np.mean((y_test-y_pred_KNN)**2))
    return list_of_y_pred_KNN,list_of_y_pred_KNN_full_range,list_of_mse_KNN

def ComputeKNNClassifiers(X_train_D,X_test_D,y_train_D,y_test_D,X_test_full_range_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_E,K_list):
    list_of_y_pred_KNN_D,list_of_y_pred_KNN_full_range_D,list_of_mse_KNN_D = KNNClassifier(X_train_D,X_test_D,y_train_D,y_test_D,K_list,X_test_full_range_D)
    list_of_y_pred_KNN_E,list_of_y_pred_KNN_full_range_E,list_of_mse_KNN_E = KNNClassifier(X_train_E,X_test_E,y_train_E,y_test_E,K_list,X_test_full_range_E)
    return list_of_y_pred_KNN_D,list_of_y_pred_KNN_full_range_D,list_of_mse_KNN_D,list_of_y_pred_KNN_E,list_of_y_pred_KNN_full_range_E,list_of_mse_KNN_E

def PlotOutputs(X_test,X_test_full_range,y_test,y_pred_LinR_full_range,list_of_y_pred_KNN_full_range,K_list,data_type_str):
    plt.scatter(X_test,y_test,color='green',label='Original Data')
    plt.plot(X_test_full_range,y_pred_LinR_full_range,color='blue',label='Using Linear Regression')
    for i in range(K_list.shape[0]):
        plt.scatter(X_test_full_range,list_of_y_pred_KNN_full_range[i],label='Using KNN (K='+str(K_list[i])+')')
        plt.xlabel('Inputs, x')
        plt.ylabel('Outputs, y')
        plt.title('Outputs, y vs inputs, x For Data Type '+data_type_str)
        plt.legend()
    plt.show()
    return

def PlotMSE(mse_LinR,list_of_mse_KNN,K_list,data_type_str):
    plt.axhline(y=mse_LinR,color='green',label='Linear Regression')
    plt.plot(K_list,list_of_mse_KNN,color='blue',label='KNN')
    plt.xlabel('K value')
    plt.ylabel('MSE')
    plt.title('MSE vs K value For Data Type '+data_type_str)
    plt.legend()
    plt.show()
    return

def CreatePlots(X_test_D,X_test_full_range_D,y_test_D,y_pred_LinR_full_range_D,mse_LinR_D,X_test_E,X_test_full_range_E,y_test_E,y_pred_LinR_full_range_E,mse_LinR_E,list_of_y_pred_KNN_full_range_D,list_of_mse_KNN_D,list_of_y_pred_KNN_full_range_E,list_of_mse_KNN_E,K_list):
    PlotOutputs(X_test_D,X_test_full_range_D,y_test_D,y_pred_LinR_full_range_D,list_of_y_pred_KNN_full_range_D,K_list,'D')
    PlotOutputs(X_test_E,X_test_full_range_E,y_test_E,y_pred_LinR_full_range_E,list_of_y_pred_KNN_full_range_E,K_list,'E')
    PlotMSE(mse_LinR_D,list_of_mse_KNN_D,K_list,'D')
    PlotMSE(mse_LinR_E,list_of_mse_KNN_E,K_list,'E')
    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train_D = 'Data/X_train_D.csv'
filename_X_test_D = 'Data/X_test_D.csv'
filename_y_train_D = 'Data/y_train_D.csv'
filename_y_test_D = 'Data/y_test_D.csv'
filename_X_train_E = 'Data/X_train_E.csv'
filename_X_test_E = 'Data/X_test_E.csv'
filename_y_train_E = 'Data/y_test_E.csv'
filename_y_test_E = 'Data/y_test_E.csv'
num_points = 1000
K_list = np.arange(1,10)

#################################################################################################################
### Main Code

X_train_D,X_test_D,y_train_D,y_test_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_D,X_test_full_range_E = PrepareTrainingAndTestData(filename_X_train_D,filename_X_test_D,filename_y_train_D,filename_y_test_D,filename_X_train_E,filename_X_test_E,filename_y_train_E,filename_y_test_E,num_points)
z_vector_LinR_D,y_pred_LinR_D,mse_LinR_D,y_pred_LinR_full_range_D,z_vector_LinR_E,y_pred_LinR_E,mse_LinR_E,y_pred_LinR_full_range_E = ComputeLinearRegressionPredictors(X_train_D,X_test_D,y_train_D,y_test_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_D,X_test_full_range_E)
list_of_y_pred_KNN_D,list_of_y_pred_KNN_full_range_D,list_of_mse_KNN_D,list_of_y_pred_KNN_E,list_of_y_pred_KNN_full_range_E,list_of_mse_KNN_E = ComputeKNNClassifiers(X_train_D,X_test_D,y_train_D,y_test_D,X_test_full_range_D,X_train_E,X_test_E,y_train_E,y_test_E,X_test_full_range_E,K_list)
CreatePlots(X_test_D,X_test_full_range_D,y_test_D,y_pred_LinR_full_range_D,mse_LinR_D,X_test_E,X_test_full_range_E,y_test_E,y_pred_LinR_full_range_E,mse_LinR_E,list_of_y_pred_KNN_full_range_D,list_of_mse_KNN_D,list_of_y_pred_KNN_full_range_E,list_of_mse_KNN_E,K_list)











