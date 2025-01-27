import numpy as np
import time
import matplotlib.pyplot as plt
import sys

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

#LoadTrainingAndTestData Function
def LoadTrainingAndTestData(filename_X_train,filename_X_test,filename_y_train,filename_y_test):
	X_train = np.genfromtxt(filename_X_train, delimiter=',')
	X_test = np.genfromtxt(filename_X_test, delimiter=',')
	y_train = np.genfromtxt(filename_y_train, delimiter=',')
	y_test = np.genfromtxt(filename_y_test, delimiter=',')
	return X_train,X_test,y_train,y_test

#StandardizeTrainingAndTestData Function
def StandardizeTrainingAndTestData(X_train,X_test,y_train,y_test):
	X_train_std = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
	X_test_std = (X_test - np.mean(X_test, axis=0))/np.std(X_test, axis=0)
	y_train_std = (y_train - np.mean(y_train, axis=0))/np.std(y_train, axis=0)
	y_test_std = (y_test - np.mean(y_test, axis=0))/np.std(y_test, axis=0)
	return X_train_std,X_test_std,y_train_std,y_test_std

#PrepareTrainingAndTestData Function
def PrepareTrainingAndTestData(filename_X_train,filename_X_test,filename_y_train,filename_y_test):
	X_train,X_test,y_train,y_test = LoadTrainingAndTestData(filename_X_train,filename_X_test,filename_y_train,filename_y_test)
	X_train = X_train.T
	X_test = X_test.T
	y_train = y_train.reshape(-1,1) 
	y_test = y_test.reshape(-1,1)
	X_train_std,X_test_std,y_train_std,y_test_std = StandardizeTrainingAndTestData(X_train,X_test,y_train,y_test)
	return X_train_std,y_train_std,X_test_std,y_test_std,X_train,y_train,X_test,y_test

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
#ComputeLoss Function
def ComputeLoss(X_matrix,y_vector,w_vector,b,lamda):
	loss = (1/(2*X_matrix.shape[0]))*np.dot((np.dot(X_matrix,w_vector) + b*np.ones([X_matrix.shape[0],1]) - y_vector).T,(np.dot(X_matrix,w_vector) + b*np.ones([X_matrix.shape[0],1]) - y_vector)) + lamda*np.dot(w_vector.T,w_vector)
	return loss

#GetParameterVectorUsingRidgeRegressionUsingClosedForm Function
def GetParameterVectorUsingRidgeRegressionUsingClosedForm(X_matrix_train,y_vector_train,X_matrix_test,y_vector_test,lambda_val_list):
	#Initialize parameters
	A_matrix_train = CreateAMatrix(X_matrix_train)
	for i in range(lambda_val_list.shape[0]):
		start_time = time.perf_counter()
		lambda_val = lambda_val_list[i]
		z_vector = np.linalg.solve(np.dot(A_matrix_train.T,A_matrix_train)+2*lambda_val*A_matrix_train.shape[0]*np.eye(A_matrix_train.shape[1]),np.dot(A_matrix_train.T,y_vector_train))
		w_vector = z_vector[:-1,:]
		b = z_vector[-1,:]
		end_time = time.perf_counter()
		if i == 0:
			training_error_list = ComputeError(X_matrix_train,y_vector_train,w_vector,b)
			training_loss_list = ComputeLoss(X_matrix_train,y_vector_train,w_vector,b,lambda_val)
			testing_error_list = ComputeError(X_matrix_test,y_vector_test,w_vector,b)
			runtime_list = np.array([end_time - start_time])
		else:
			training_error_list = np.append(training_error_list,ComputeError(X_matrix_train,y_vector_train,w_vector,b))
			training_loss_list = np.append(training_loss_list,ComputeLoss(X_matrix_train,y_vector_train,w_vector,b,lambda_val)) 
			testing_error_list = np.append(testing_error_list,ComputeError(X_matrix_test,y_vector_test,w_vector,b))
			runtime_list = np.append(runtime_list,end_time-start_time)
	list_of_error = lambda_val_list.reshape(-1,1)
	list_of_error = np.hstack((list_of_error,training_error_list.reshape(-1,1)))
	list_of_error = np.hstack((list_of_error,training_loss_list.reshape(-1,1)))
	list_of_error = np.hstack((list_of_error,testing_error_list.reshape(-1,1)))
	list_of_error = np.hstack((list_of_error,runtime_list.reshape(-1,1)))
	return w_vector,b,list_of_error

#GetParameterVectorUsingRidgeRegressionWithGradientDescent Function
def GetParameterVectorUsingRidgeRegressionWithGradientDescent(X_matrix_train,y_vector_train,X_matrix_test,y_vector_test,max_pass,learning_rate,tol,lambda_val_list):
	#Initialize parameters
	w0_vector = np.zeros([X_matrix_train.shape[1],1])
	b0 = 0
	list_of_training_error_and_loss_matrices = []
	runtime_list = np.zeros(lambda_val_list.shape[0])

	#Run for every lambda value
	for i in range(lambda_val_list.shape[0]):
		lambda_val = lambda_val_list[i]
		w_vector = w0_vector
		b = b0
		it_num = 0
		diff_of_wt_and_wtminus1 = tol+1
		
		#Loop until the w_vector changes no more than tol after an iteration
		start_time = time.time()
		while (np.linalg.norm(diff_of_wt_and_wtminus1) > tol):
			it_num = it_num + 1
			
			#Store training error
			if it_num == 1:
				training_error_list = ComputeError(X_matrix_train,y_vector_train,w_vector,b)
				training_loss_list = ComputeLoss(X_matrix_train,y_vector_train,w_vector,b,lambda_val)
			else:
				training_error_list = np.append(training_error_list,ComputeError(X_matrix_train,y_vector_train,w_vector,b))
				training_loss_list = np.append(training_loss_list,ComputeLoss(X_matrix_train,y_vector_train,w_vector,b,lambda_val))
			
			#Compute the gradient
			gradient_w,gradient_b = ComputeRidgeRegressionGradient(X_matrix_train,y_vector_train,w_vector,b,lambda_val)
			wtminus1_vector = w_vector.copy()

			#Update parameter vectors
			w_vector = w_vector - learning_rate*gradient_w
			b = b - learning_rate*gradient_b
			diff_of_wt_and_wtminus1 = w_vector - wtminus1_vector 
			if it_num >= max_pass:
				break
		end_time = time.time()

		#Store runtime
		runtime_list[i] = end_time - start_time 

		#Store error
		training_error_list = training_error_list.reshape(-1,1)
		training_error_and_loss_matrix = np.zeros([training_error_list.shape[0],3]) 
		training_error_and_loss_matrix[:,0] = np.arange(1,it_num+1).reshape(-1)
		training_error_and_loss_matrix[:,1] = training_error_list.reshape(-1) 
		training_error_and_loss_matrix[:,2] = training_loss_list.reshape(-1)  
		if i == 0:
			list_of_training_error_and_loss_matrices = [training_error_and_loss_matrix]

			list_of_test_error = ComputeError(X_matrix_test,y_vector_test,w_vector,b)[0]
		else:
			list_of_test_error = np.append(list_of_test_error,ComputeError(X_matrix_test,y_vector_test,w_vector,b))
			list_of_training_error_and_loss_matrices.append(training_error_and_loss_matrix)
	return w_vector,b,list_of_training_error_and_loss_matrices,list_of_test_error,runtime_list

#PrintOutputs Function
def PrintOutputs(list_of_error_closed_form,list_of_training_error_and_loss_matrices_gradient_descent_std,list_of_test_error_gradient_descent_std,runtime_list_gradient_descent_std,list_of_training_error_and_loss_matrices_gradient_descent_regular,list_of_test_error_gradient_descent_regular,runtime_list_gradient_descent_regular):
	print("************************************************************")
	print("Ridge Regression using Closed Form (Standardized):")
	print("[Lambda, Training Error, Training Loss, Testing Error, Runtime]:")
	print(list_of_error_closed_form)
	print("************************************************************")
	print("Ridge Regression using Gradient Descent (Standardized):")
	for i in range(len(lambda_val_list)):
		lamda_val = lambda_val_list[i]
		print("################## Lambda =",lamda_val,"##################")
		print("[Iteration Number, Training Error, Training Loss]:")
		
		#Print for more detailed, less convenience
		'''
		for j in range(list_of_training_error_and_loss_matrices_gradient_descent_std[i].shape[0]):
			if j == 0:
				print(list_of_training_error_and_loss_matrices_gradient_descent_std[i][j,:])
			else:
				print(list_of_training_error_and_loss_matrices_gradient_descent_std[i])
		'''

		#Print for less details, more convenience
		#'''
		print(list_of_training_error_and_loss_matrices_gradient_descent_std[i])
		#'''

		print("Testing Error:")
		print(list_of_test_error_gradient_descent_std[i])
		print("Runtime:")
		print(runtime_list_gradient_descent_std[i],"s")
		print()
	print("************************************************************")
	print("Ridge Regression using Gradient Descent (Regular):")
	list_of_training_error_and_loss_matrices_gradient_descent_regular,list_of_test_error_gradient_descent_regular,runtime_list_gradient_descent_regular
	for i in range(len(lambda_val_list)):
		lamda_val = lambda_val_list[i]
		print("################## Lambda =",lamda_val,"##################")
		print("[Iteration Number, Training Error, Training Loss]:")
		
		#Print for more detailed, less convenience
		'''
		for j in range(list_of_training_error_and_loss_matrices_gradient_descent_regular[i].shape[0]):
			if j == 0:
				print(list_of_training_error_and_loss_matrices_gradient_descent_regular[i][j,:])
			else:
				print(list_of_training_error_and_loss_matrices_gradient_descent_regular[i])
		'''

		#Print for less details, more convenience
		#'''
		print(list_of_training_error_and_loss_matrices_gradient_descent_regular[i])
		#'''

		print("Testing Error:")
		print(list_of_test_error_gradient_descent_regular[i])
		print("Runtime:")
		print(runtime_list_gradient_descent_regular[i],"s")
		print()
	print("************************************************************")
	return

#CreatePlot Function
def CreatePlot(list_of_training_error_and_loss_matrices_gradient_descent,lambda_val_list,learning_rate,tol,data_mode_str):
	for i in range(len(list_of_training_error_and_loss_matrices_gradient_descent)):
		training_error_matrix = list_of_training_error_and_loss_matrices_gradient_descent[i]
		lambda_val = lambda_val_list[i]
		plt.plot(training_error_matrix[:,0],training_error_matrix[:,2],color='green',label='Original Data')
		plt.xlabel('Iteration Number')
		plt.ylabel('Training Loss')
		plt.title('Ridge Regression Gradient Descent Training Loss ('+str(data_mode_str)+') for Lamda='+str(lambda_val)+' LR='+str(learning_rate)+', Tol='+str(tol))
		plt.show()
	return

#################################################################################################################
### Main Functions

def CompareClosedFormAndGradientDescentRidgeRegression(filename_X_train,filename_X_test,filename_y_train,filename_y_test,lambda_val_list,max_pass,learning_rate,tol):
	X_train_std,y_train_std,X_test_std,Y_test_std,X_train_regular,y_train_regular,X_test_regular,y_test_regular = PrepareTrainingAndTestData(filename_X_train,filename_X_test,filename_y_train,filename_y_test)
	w_vector_closed_form,b_closed_form,list_of_error_closed_form = GetParameterVectorUsingRidgeRegressionUsingClosedForm(X_train_std,y_train_std,X_test_std,Y_test_std,lambda_val_list)
	w_vector_gradient_descent_std,b_gradient_descent_std,list_of_training_error_and_loss_matrices_gradient_descent_std,list_of_test_error_gradient_descent_std,runtime_list_gradient_descent_std = GetParameterVectorUsingRidgeRegressionWithGradientDescent(X_train_std,y_train_std,X_test_std,Y_test_std,max_pass,learning_rate,tol,lambda_val_list)
	w_vector_gradient_descent_regular,b_gradient_descent_regular,list_of_training_error_and_loss_matrices_gradient_descent_regular,list_of_test_error_gradient_descent_regular,runtime_list_gradient_descent_regular = GetParameterVectorUsingRidgeRegressionWithGradientDescent(X_train_regular,y_train_regular,X_test_regular,y_test_regular,max_pass,learning_rate,tol,lambda_val_list)
	PrintOutputs(list_of_error_closed_form,list_of_training_error_and_loss_matrices_gradient_descent_std,list_of_test_error_gradient_descent_std,runtime_list_gradient_descent_std,list_of_training_error_and_loss_matrices_gradient_descent_regular,list_of_test_error_gradient_descent_regular,runtime_list_gradient_descent_regular)
	CreatePlot(list_of_training_error_and_loss_matrices_gradient_descent_std,lambda_val_list,learning_rate,tol,'Standardized')
	CreatePlot(list_of_training_error_and_loss_matrices_gradient_descent_regular,lambda_val_list,learning_rate,tol,'Regular')
	return 

#################################################################################################################
### Variables

filename_X_train = 'Data/housing_X_train.csv'
filename_X_test = 'Data/housing_X_test.csv'
filename_y_train = 'Data/housing_Y_train.csv'
filename_y_test = 'Data/housing_Y_test.csv'
#max_pass = 1e6
#learning_rate = 1e-3
#tol = 1e-6
max_pass = 1e6
learning_rate = 1e-3
tol = 1e-6
lambda_val_list = np.array([0,10])
output_file = 'A1_E2_Q5.txt'

#################################################################################################################
### Main Code
CompareClosedFormAndGradientDescentRidgeRegression(filename_X_train,filename_X_test,filename_y_train,filename_y_test,lambda_val_list,max_pass,learning_rate,tol)

