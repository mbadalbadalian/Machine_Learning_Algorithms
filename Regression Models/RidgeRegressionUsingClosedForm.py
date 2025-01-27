import numpy as np

#################################################################################################################
### Other Functions

#CreateAMatrix Function
def CreateAMatrix(X_matrix):
    A_matrix = np.hstack((X_matrix, np.ones([X_matrix.shape[0],1])))
    return A_matrix

#CreatezVector Function
def CreatezVector(w_vector,b):
    z_vector = np.append(w_vector,b).reshape(-1,1)
    return z_vector

#################################################################################################################
### Main Functions

#GetParameterVectorUsingRidgeRegressionUsingClosedForm Function
def GetParameterVectorUsingRidgeRegressionUsingClosedForm(X_matrix,y_vector,lambda_val):
    A_matrix = CreateAMatrix(X_matrix)
    
    #Find parameter vector
    z_vector = np.linalg.solve(np.dot(A_matrix.T,A_matrix)+2*lambda_val*A_matrix.shape[0]*np.eye(A_matrix.shape[1]),np.dot(A_matrix.T,y_vector))
    w_vector = z_vector[:-1,:]
    b = z_vector[-1,:]
    print("Q3 (Using Closed Form Solution):")
    print("Ridge Regression Optimal w vector:")
    print(w_vector)
    print("Ridge Regression Optimal b value:")
    print(b)
    print()
    print("***********************************************************************************")
    return w_vector,b

#################################################################################################################
### Variables

X_matrix = np.array([[1, 2],[3, 4],[5, 6]])
y_vector = np.array([[2],[4],[6]])
lambda_val = 5

#################################################################################################################
### Main Code

w_vector,b = GetParameterVectorUsingRidgeRegressionUsingClosedForm(X_matrix,y_vector,lambda_val)
