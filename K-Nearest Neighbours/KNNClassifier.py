import numpy as np

#################################################################################################################
### Other Functions

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

#################################################################################################################
### Main Functions

def KNNClassifier(X_train,X_test,y_train,K):
    X_test_list = X_test.tolist()
    y_pred = np.array([[PredictUsingKNN(X_train,X_test_point,y_train,K)] for X_test_point in X_test_list])
    print("************************************************************")
    print('################ KNN (Using K ='+str(K)+') ################')
    print('Input:',X_test)
    print('Predicted Output:',y_pred)
    print("************************************************************")
    return y_pred

#################################################################################################################
### Variables

X_train = np.array([[-1,-1],[-1,1],[-1,2],[1,-1],[1,1],[1,2],[2,-1],[2,1],[2,2]])
X_test = np.array([[0,-10],[0,10],[10,0],[-10,0]])
y_train = np.array([[1],[-1],[1],[1],[1],[-1],[-1],[-1],[1]])
K = 6

#################################################################################################################
### Main Code

y_pred = KNNClassifier(X_train,X_test,y_train,K)



