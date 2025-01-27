import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVC
import sys

#################################################################################################################
### Other Functions

#LoadTrainingAndTestData Function
def LoadTrainingData(filename_X_train,filename_y_train):
    X_train = np.genfromtxt(filename_X_train, delimiter=',')
    y_train = np.genfromtxt(filename_y_train, delimiter=',')
    
    #Reshape data to make it a column vector
    y_train = y_train.reshape(-1,1)
    return X_train,y_train

#LoadTrainingAndTestData Function
def LoadTrainingAndTestingData(filename_X_train,filename_y_train,filename_X_test,filename_y_test):
    X_train = np.genfromtxt(filename_X_train, delimiter=',')
    y_train = np.genfromtxt(filename_y_train, delimiter=',')
    X_test = np.genfromtxt(filename_X_test, delimiter=',')
    y_test = np.genfromtxt(filename_y_test, delimiter=',')

    #Reshape data to make it a column vector
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return X_train,y_train,X_test,y_test

#StandardizeTrainingAndTestData Function
def StandardizeTrainingData(X_train):
	X_train_std = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
	return X_train_std

#PrepareTrainingData Function
def PrepareTrainingData(filename_X_train,filename_y_train):
	X_train,y_train = LoadTrainingData(filename_X_train,filename_y_train)
	y_train = y_train.reshape(-1) 
	return X_train,y_train

#PrepareTrainingAndTestingData Function
def PrepareTrainingAndTestingData(filename_X_train,filename_y_train,filename_X_test,filename_y_test):
	X_train,y_train,X_test,y_test = LoadTrainingAndTestingData(filename_X_train,filename_y_train,filename_X_test,filename_y_test)
	y_train = y_train.reshape(-1) 
	return X_train,y_train,X_test,y_test

#ComputeSoftMarginSupportVectorMachines Function
def ComputeSoftMarginSupportVectorMachinesTrain(X_train,y_train):
    soft_margin_support_vector_machines_model = SVC(C=1,kernel='linear').fit(X_train,y_train)
    w_vector_SM_SVM = soft_margin_support_vector_machines_model.coef_.T
    b_SM_SVM = soft_margin_support_vector_machines_model.intercept_
    y_pred_SM_SVM = soft_margin_support_vector_machines_model.predict(X_train).reshape(-1,1)
    return soft_margin_support_vector_machines_model,w_vector_SM_SVM,b_SM_SVM,y_pred_SM_SVM

#Sign Function
def Sign(input_matrix):
    output_matrix = np.where(input_matrix > 0, 1, -1)
    return output_matrix

#ReplaceZerosWithMinusOnes Function
def ReplaceZerosWithMinusOnes(X_train,y_train,w_vector_SVM,b_SM_SVM):
    #Take inner produce of each point with coefficient vector and scale by sign(y_train) function (turns 0 into -1)
    scaled_inner_product = (np.dot(X_train,w_vector_SVM)+b_SM_SVM)*Sign(y_train.reshape(-1,1))
    num_values_less_or_equal_1 = np.sum(scaled_inner_product<= 1)
    return scaled_inner_product,num_values_less_or_equal_1

#ComputeParameterVectorFromSupportVectors Function
def ComputeParameterVectorFromSupportVectors(support_vector_machines_model,X_train,y_train):
    support_vectors_SVM = support_vector_machines_model.support_vectors_
    #Count number of support vectors
    num_support_vectors_SVM = len(support_vectors_SVM)
    support_vectors_SVM_indices = np.where((X_train[:, None] == support_vectors_SVM).all(-1).any(-1))[0]
    #Get associated SVM outputs
    support_vectors_SVM_outputs = y_train[support_vectors_SVM_indices].reshape(-1,1)
    #Get duel coefficients and SVM parameter vector
    duel_coefficients_SVM = support_vector_machines_model.dual_coef_.T
    parameter_vector_SVM = np.dot(support_vectors_SVM.T,duel_coefficients_SVM)
    return support_vectors_SVM,num_support_vectors_SVM,support_vectors_SVM_indices,support_vectors_SVM_outputs,duel_coefficients_SVM,parameter_vector_SVM

#ComputeLogisticRegression Function
def ComputeLogisticRegressionTrainAndTest(X_train,y_train,X_test,y_test):
    logistic_regression_model = sm.Logit(y_train,X_train).fit_regularized()
    parameter_vector_LogR = logistic_regression_model.params.reshape(-1,1)
    y_pred_LogR_train = logistic_regression_model.predict(X_train).reshape(-1,1)
    y_pred_LogR_test = logistic_regression_model.predict(X_test).reshape(-1,1)
    zero_one_loss_LogR = ZeroOneLoss(y_test,y_pred_LogR_test)
    return logistic_regression_model,parameter_vector_LogR,y_pred_LogR_train,y_pred_LogR_test,zero_one_loss_LogR

#ComputeSoftMarginSupportVectorMachines Function
def ComputeSoftMarginSupportVectorMachinesTrainAndTest(X_train,y_train,X_test,y_test):
    soft_margin_support_vector_machines_model = SVC(C=1,kernel='linear').fit(X_train,y_train)
    w_vector_SM_SVM = soft_margin_support_vector_machines_model.coef_.T
    b_SM_SVM = soft_margin_support_vector_machines_model.intercept_
    y_pred_SM_SVM_train = soft_margin_support_vector_machines_model.predict(X_train).reshape(-1,1)
    y_pred_SM_SVM_test = soft_margin_support_vector_machines_model.predict(X_test).reshape(-1,1)
    zero_one_loss_SM_SVM = ZeroOneLoss(y_test,y_pred_SM_SVM_test)
    return soft_margin_support_vector_machines_model,w_vector_SM_SVM,b_SM_SVM,y_pred_SM_SVM_train,y_pred_SM_SVM_test,zero_one_loss_SM_SVM

#ComputeHardMarginSupportVectorMachines Function
def ComputeHardMarginSupportVectorMachinesTrainAndTest(X_train,y_train,X_test,y_test):
    hard_margin_support_vector_machines_model = SVC(C=sys.float_info.max,kernel='linear').fit(X_train,y_train)
    w_vector_HM_SVM = hard_margin_support_vector_machines_model.coef_.T
    b_HM_SVM = hard_margin_support_vector_machines_model.intercept_
    y_pred_HM_SVM_train = hard_margin_support_vector_machines_model.predict(X_train).reshape(-1,1)
    y_pred_HM_SVM_test = hard_margin_support_vector_machines_model.predict(X_test).reshape(-1,1)
    zero_one_loss_HM_SVM = ZeroOneLoss(y_test,y_pred_HM_SVM_test)
    return hard_margin_support_vector_machines_model,w_vector_HM_SVM,b_HM_SVM,y_pred_HM_SVM_train,y_pred_HM_SVM_test,zero_one_loss_HM_SVM

#ZeroOneLoss Function
def ZeroOneLoss(y_test,y_pred):
    y_pred = y_pred.round()
    zero_one_loss = np.sum(y_test!=y_pred)/y_test.shape[0]
    #zero_one_loss = np.sum([1 for y_test,y_pred in zip(y_test,y_pred) if y_test != y_pred])/y_test.shape[0]
    return zero_one_loss

#PrintOutputs Function
def PrintOutputs(num_values_less_or_equal_1_SM_SVM_A,num_support_vectors_SM_SVM_A,support_vectors_SM_SVM_A_indices,support_vectors_SM_SVM_A_outputs,parameter_vector_SM_SVM_A,parameter_vector_LogR,zero_one_loss_LogR_B,w_vector_SM_SVM_B,b_SM_SVM_B,zero_one_loss_SM_SVM_B,num_support_vectors_SM_SVM_B,support_vectors_SM_SVM_B_indices,support_vectors_SM_SVM_B_outputs):
    print("************************************************************")
    print("################ Dataset A Using Soft-Margin Support Vector Machines ################")
    print("Number of values <= 1:")
    print(num_values_less_or_equal_1_SM_SVM_A)
    print("Number of support vectors used:")
    print(num_support_vectors_SM_SVM_A)
    print()
    print("Parameter vector obtained:")
    print(parameter_vector_SM_SVM_A)
    print("as linear combination of support vectors at indices")
    print(support_vectors_SM_SVM_A_indices)
    print("with labels (transposed)")
    print(support_vectors_SM_SVM_A_outputs)
    print("************************************************************")
    print("################ Dataset B Using Logistic Regression ################")
    print("Parameter vector obtained using model")
    print(parameter_vector_LogR)
    print("Empirical Prediction Accuracy (Using 0-1 Loss)")
    print(zero_one_loss_LogR_B)
    print("************************************************************")
    print("################ Dataset B Using Soft-Margin Support Vector Machines ################")
    print("Parameter vector obtained using model")
    print(w_vector_SM_SVM_B)
    print("Bias obtained using model")
    print(b_SM_SVM_B)
    print("Empirical Prediction Accuracy (Using 0-1 Loss)")
    print(zero_one_loss_SM_SVM_B)
    print("Number of support vectors used:")
    print(num_support_vectors_SM_SVM_B)
    print()
    print("Parameter vector obtained:")
    print('Same as parameter vector obtained from model')
    print("Obtained as linear combination of support vectors at indices")
    print(support_vectors_SM_SVM_B_indices)
    print("with labels")
    print(support_vectors_SM_SVM_B_outputs.T[0])
    print("************************************************************")

    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train_A = 'Data/X_train_A.csv'
filename_y_train_A = 'Data/Y_train_A.csv'
filename_X_train_B = 'Data/X_train_B.csv'
filename_y_train_B = 'Data/Y_train_B.csv'
filename_X_test_B = 'Data/X_test_B.csv'
filename_y_test_B = 'Data/Y_test_B.csv'

#################################################################################################################
### Main Code

#Dataset A
X_train_A,y_train_A = PrepareTrainingData(filename_X_train_A,filename_y_train_A)
soft_margin_support_vector_machines_model_A,w_vector_SM_SVM_A,b_SM_SVM_A,y_pred_SM_SVM_A = ComputeSoftMarginSupportVectorMachinesTrain(X_train_A,y_train_A)
scaled_inner_product_SM_SVM_A,num_values_less_or_equal_1_SM_SVM_A = ReplaceZerosWithMinusOnes(X_train_A,y_train_A,w_vector_SM_SVM_A,b_SM_SVM_A)
support_vectors_SM_SVM_A,num_support_vectors_SM_SVM_A,support_vectors_SM_SVM_A_indices,support_vectors_SM_SVM_A_outputs,duel_coefficients_SM_SVM_A,parameter_vector_SM_SVM_A = ComputeParameterVectorFromSupportVectors(soft_margin_support_vector_machines_model_A,X_train_A,y_train_A)

#Dataset B
X_train_B,y_train_B,X_test_B,y_test_B = PrepareTrainingAndTestingData(filename_X_train_B,filename_y_train_B,filename_X_test_B,filename_y_test_B)
logistic_regression_model_B,parameter_vector_LogR,y_pred_LogR_train_B,y_pred_LogR_test_B,zero_one_loss_LogR_B = ComputeLogisticRegressionTrainAndTest(X_train_B,y_train_B,X_test_B,y_test_B)
soft_margin_support_vector_machines_model_B,w_vector_SM_SVM_B,b_SM_SVM_B,y_pred_SM_SVM_train_B,y_pred_SM_SVM_test_B,zero_one_loss_SM_SVM_B = ComputeSoftMarginSupportVectorMachinesTrainAndTest(X_train_B,y_train_B,X_test_B,y_test_B)
scaled_inner_product_SM_SVM_B,num_values_less_or_equal_1_SM_SVM_B = ReplaceZerosWithMinusOnes(X_train_B,y_train_B,w_vector_SM_SVM_B,b_SM_SVM_B)
support_vectors_SVM_SM_B,num_support_vectors_SM_SVM_B,support_vectors_SM_SVM_B_indices,support_vectors_SM_SVM_B_outputs,duel_coefficients_SM_SVM_B,parameter_vector_SM_SVM_B = ComputeParameterVectorFromSupportVectors(soft_margin_support_vector_machines_model_B,X_train_B,y_train_B)

#Doesn't work for this case
#hard_margin_support_vector_machines_model_B,w_vector_HM_SVM_B,b_HM_SVM_B,y_pred_HM_SVM_train_B,y_pred_HM_SVM_test_B,zero_one_loss_HM_SVM_B = ComputeHardMarginSupportVectorMachinesTrainAndTest(X_train_B,y_train_B,X_test_B,y_test_B)
#scaled_inner_product_HM_SVM_B,num_values_less_or_equal_1_HM_SVM_B = ReplaceZerosWithMinusOnes(X_train_B,y_train_B,w_vector_HM_SVM_B,b_HM_SVM_B)
#support_vectors_SVM_HM_B,num_support_vectors_HM_SVM_B,support_vectors_HM_SVM_B_indices,support_vectors_HM_SVM_B_outputs,duel_coefficients_HM_SVM_B,parameter_vector_HM_SVM_B = ComputeParameterVectorFromSupportVectors(hard_margin_support_vector_machines_model_B,X_train_B,y_train_B)
PrintOutputs(num_values_less_or_equal_1_SM_SVM_A,num_support_vectors_SM_SVM_A,support_vectors_SM_SVM_A_indices,support_vectors_SM_SVM_A_outputs,parameter_vector_SM_SVM_A,parameter_vector_LogR,zero_one_loss_LogR_B,w_vector_SM_SVM_B,b_SM_SVM_B,zero_one_loss_SM_SVM_B,num_support_vectors_SM_SVM_B,support_vectors_SM_SVM_B_indices,support_vectors_SM_SVM_B_outputs)


