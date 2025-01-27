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

#StandardizeTrainingAndTestData Function
def StandardizeTrainingData(X_train):
	X_train_std = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
	return X_train_std

#PrepareTrainingAndTestData Function
def PrepareTrainingAndTestData(filename_X_train,filename_y_train):
	X_train,y_train = LoadTrainingData(filename_X_train,filename_y_train)
	y_train = y_train.reshape(-1) 
	return X_train,y_train

#ComputeLogisticRegression Function
def ComputeLogisticRegression(X_train,y_train):
    logistic_regression_model = sm.Logit(y_train,X_train).fit()
    y_pred_LogR = logistic_regression_model.predict(X_train).reshape(-1,1)
    return logistic_regression_model,y_pred_LogR

#ComputeLogisticRegressionRegularized Function
def ComputeLogisticRegressionRegularized(X_train,y_train):
    logistic_regression_model_regularized = sm.Logit(y_train,X_train).fit_regularized()
    y_pred_LogR_regularized = logistic_regression_model_regularized.predict(X_train).reshape(-1,1)
    return logistic_regression_model_regularized,y_pred_LogR_regularized

#ComputeSoftMarginSupportVectorMachines Function
def ComputeSoftMarginSupportVectorMachines(X_train,y_train):
    soft_margin_support_vector_machines_model = SVC(C=1,kernel='linear').fit(X_train,y_train)
    y_pred_SM_SVM = soft_margin_support_vector_machines_model.predict(X_train).reshape(-1,1)
    return soft_margin_support_vector_machines_model,y_pred_SM_SVM

#ComputeHardMarginSupportVectorMachines Function
def ComputeHardMarginSupportVectorMachines(X_train,y_train):
    hard_margin_support_vector_machines_model = SVC(C=sys.float_info.max,kernel='linear').fit(X_train,y_train)
    y_pred_HM_SVM = hard_margin_support_vector_machines_model.predict(X_train).reshape(-1,1)
    return hard_margin_support_vector_machines_model,y_pred_HM_SVM

#PrintOutputs Function
def PrintOutputs(y_pred_LogR,y_pred_SM_SVM,y_pred_HM_SVM):
    print("************************************************************")
    print("################ Logistic Regression Regularized ################")
    print("Predicted y values from training points")
    print(y_pred_LogR)
    print()
    print("************************************************************")
    print("################ Soft-Margin Support Vector Machines ################")
    print("Predicted y values from training points")
    print(y_pred_SM_SVM)
    print()
    print("************************************************************")
    print("################ Hard-Margin Support Vector Machines ################")
    print("Predicted y values from training points")
    print(y_pred_HM_SVM)
    print()
    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train = 'Data/X_train_A.csv'
filename_y_train = 'Data/Y_train_A.csv'

#################################################################################################################
### Main Code

#Obtain the data
X_train,y_train = PrepareTrainingAndTestData(filename_X_train,filename_y_train)
#Produce the models and compute the outputs
logistic_regression_model,y_pred_LogR = ComputeLogisticRegression(X_train,y_train)
logistic_regression_model_regularized,y_pred_LogR_regularized = ComputeLogisticRegressionRegularized(X_train,y_train)
soft_margin_support_vector_machines_model,y_pred_SM_SVM = ComputeSoftMarginSupportVectorMachines(X_train,y_train)
hard_margin_support_vector_machines_model,y_pred_HM_SVM = ComputeHardMarginSupportVectorMachines(X_train,y_train)
#Print the outputs
PrintOutputs(y_pred_LogR_regularized,y_pred_SM_SVM,y_pred_HM_SVM)