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

#ComputeRidgeRegressionGradient Function
def ComputeRidgeRegressionGradient(X_matrix,y_vector,w_vector,b,lambda_val):
    #Compute prediction values
    y_pred = np.dot(X_matrix,w_vector).reshape(-1,1) + b*np.ones([X_matrix.shape[0],1])
    
    #Compute the gradient
    gradient_w = (1/X_matrix.shape[0])*np.dot(X_matrix.T,y_pred-y_vector) + 2*lambda_val*w_vector
    gradient_b = (1/X_matrix.shape[0])*np.dot(np.ones([X_matrix.shape[0],1]).T,y_pred-y_vector)
    return gradient_w,gradient_b

#ComputeError Function
def ComputeError(X_matrix,y_vector,w_vector,b):
	error = (1/(2*X_matrix.shape[0]))*np.dot((np.dot(X_matrix,w_vector) + b*np.ones([X_matrix.shape[0],1]) - y_vector).T,(np.dot(X_matrix,w_vector) + b*np.ones([X_matrix.shape[0],1]) - y_vector))
	return error

#################################################################################################################
### Main Functions

#GetParameterVectorUsingRidgeRegressionWithGradientDescent Function
def GetParameterVectorUsingRidgeRegressionWithGradientDescent(X_matrix,y_vector,w0_vector,b0,max_pass,learning_rate,tol,lambda_val):
    #Initialize parameters
    w_vector = w0_vector
    b = b0
    it_num = 0
    diff_of_wt_and_wtminus1 = tol+1    
    
    #Loop until the w_vector changes no more than tol after an iteration 
    while (np.linalg.norm(diff_of_wt_and_wtminus1) > tol):
        it_num = it_num + 1
        
        #Store training error
        if it_num == 1:
            training_error_list = ComputeError(X_matrix,y_vector,w_vector,b)
        else:
            training_error_list = np.append(training_error_list,ComputeError(X_matrix,y_vector,w_vector,b))
        
        #Compute the gradient
        gradient_w,gradient_b = ComputeRidgeRegressionGradient(X_matrix,y_vector,w_vector,b,lambda_val)
        wtminus1_vector = w_vector.copy()

        #Update parameter vectors
        w_vector = w_vector - learning_rate*gradient_w
        b = b - learning_rate*gradient_b
        diff_of_wt_and_wtminus1 = w_vector - wtminus1_vector 
        if it_num >= max_pass:
            break
    
    #Store error
    training_error_list = training_error_list.reshape(-1,1)
    it_list = np.arange(1,it_num+1).reshape(-1,1)
    training_error_and_it_num_mat = np.hstack((it_list,training_error_list))
    print("Q4 (Using Gradient Descent):")
    print("Ridge Regression Optimal w vector:")
    print(w_vector)
    print("Ridge Regression Optimal b value:")
    print(b)
    print("Total number of iterations:")
    print(it_num)
    print("[Iteration Number, Training Error]:")
    print(training_error_and_it_num_mat)
    print("***********************************************************************************")
    return w_vector,b,training_error_and_it_num_mat

#################################################################################################################
### Variables

X_matrix = np.array([[1, 2],[3, 4],[5, 6]])
y_vector = np.array([[2],[4],[6]])
w0_vector = np.zeros([X_matrix.shape[1],1])
b0 = 0
max_pass = 1e6
learning_rate = 1e-6
tol = 1e-6
lambda_val = 5

#################################################################################################################
### Main Code

w_vector,b,training_error_and_it_num_mat = GetParameterVectorUsingRidgeRegressionWithGradientDescent(X_matrix,y_vector,w0_vector,b0,max_pass,learning_rate,tol,lambda_val)
