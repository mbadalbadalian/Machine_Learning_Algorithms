import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error

#################################################################################################################
### Other Functions

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
    
    #Reshape data to make it a column vector
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return X_train,X_test,y_train,y_test

#PrepareTrainingAndTestData Function
def PrepareTrainingAndTestData(filename_X_train_A,filename_X_test_A,filename_y_train_A,filename_y_test_A,filename_X_train_B,filename_X_test_B,filename_y_train_B,filename_y_test_B,filename_X_train_C,filename_X_test_C,filename_y_train_C,filename_y_test_C):
    X_train_A,X_test_A,y_train_A,y_test_A = LoadTrainingAndTestData(filename_X_train_A,filename_X_test_A,filename_y_train_A,filename_y_test_A)
    X_train_B,X_test_B,y_train_B,y_test_B = LoadTrainingAndTestData(filename_X_train_B,filename_X_test_B,filename_y_train_B,filename_y_test_B)
    X_train_C,X_test_C,y_train_C,y_test_C = LoadTrainingAndTestData(filename_X_train_C,filename_X_test_C,filename_y_train_C,filename_y_test_C)
    return X_train_A,X_test_A,y_train_A,y_test_A,X_train_B,X_test_B,y_train_B,y_test_B,X_train_C,X_test_C,y_train_C,y_test_C

#ComputeLinearRegression Function
def ComputeLinearRegression(X_train,X_test,y_train,y_test):
    #Create linear regression model
    linear_regression_model = LinearRegression().fit(X_train,y_train)
    
    #Get parameter values
    w_vector_LR = linear_regression_model.coef_
    b_LR = linear_regression_model.intercept_
    z_vector_LR = CreatezVector(w_vector_LR,b_LR)

    #Compute predictions
    y_pred = linear_regression_model.predict(X_test)

    #Calculate MSE
    mse_LR = mean_squared_error(y_test, y_pred)
    return linear_regression_model,z_vector_LR,mse_LR

#ComputeLinearRegressionAllModels Function
def ComputeLinearRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,X_train_B,X_test_B,y_train_B,y_test_B,X_train_C,X_test_C,y_train_C,y_test_C):
	linear_regression_model_A,z_vector_LinR_A,mse_LinR_A = ComputeLinearRegression(X_train_A,X_test_A,y_train_A,y_test_A)
	linear_regression_model_B,z_vector_LinR_B,mse_LinR_B = ComputeLinearRegression(X_train_B,X_test_B,y_train_B,y_test_B)
	linear_regression_model_C,z_vector_LinR_C,mse_LinR_C = ComputeLinearRegression(X_train_C,X_test_C,y_train_C,y_test_C)
	return linear_regression_model_A,z_vector_LinR_A,mse_LinR_A,linear_regression_model_B,z_vector_LinR_B,mse_LinR_B,linear_regression_model_C,z_vector_LinR_C,mse_LinR_C

#ComputeRidgeRegression Function
def ComputeRidgeRegression(X_train,X_test,y_train,y_test,lambda_val):
    #Create ridge regression model
    ridge_regression_model = Ridge(alpha=lambda_val).fit(X_train,y_train)
    
    #Get parameter values
    w_vector_RR = ridge_regression_model.coef_
    b_RR = ridge_regression_model.intercept_
    z_vector_RR = CreatezVector(w_vector_RR,b_RR)
    
    #Compute predictions
    y_pred = ridge_regression_model.predict(X_test)
    
    #Calculate MSE
    mse_RR = mean_squared_error(y_test, y_pred)
    return ridge_regression_model,z_vector_RR,mse_RR

#ComputeRidgeRegressionAllModels Function
def ComputeRidgeRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,lambda_val_A,X_train_B,X_test_B,y_train_B,y_test_B,lambda_val_B,X_train_C,X_test_C,y_train_C,y_test_C,lambda_val_C):
    ridge_regression_model_A,z_vector_RR_A,mse_RR_A = ComputeRidgeRegression(X_train_A,X_test_A,y_train_A,y_test_A,lambda_val_A)
    ridge_regression_model_B,z_vector_RR_B,mse_RR_B = ComputeRidgeRegression(X_train_B,X_test_B,y_train_B,y_test_B,lambda_val_B)
    ridge_regression_model_C,z_vector_RR_C,mse_RR_C = ComputeRidgeRegression(X_train_C,X_test_C,y_train_C,y_test_C,lambda_val_C)
    return ridge_regression_model_A,z_vector_RR_A,mse_RR_A,ridge_regression_model_B,z_vector_RR_B,mse_RR_B,ridge_regression_model_C,z_vector_RR_C,mse_RR_C

#ComputeLassoRegression Function
def ComputeLassoRegression(X_train,X_test,y_train,y_test,lambda_val):
    #Create lasso regression model
    lasso_regression_model = Lasso(alpha=lambda_val).fit(X_train,y_train)
    
    #Get parameter values
    w_vector_LR = lasso_regression_model.coef_
    b_LR = lasso_regression_model.intercept_
    z_vector_LR = CreatezVector(w_vector_LR,b_LR)
    
    #Compute predictions
    y_pred = lasso_regression_model.predict(X_test)
    
    #Calculate MSE
    mse_LR = mean_squared_error(y_test, y_pred)
    return lasso_regression_model,z_vector_LR,mse_LR

#ComputeLassoRegressionAllModels Function
def ComputeLassoRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,lambda_val_A,X_train_B,X_test_B,y_train_B,y_test_B,lambda_val_B,X_train_C,X_test_C,y_train_C,y_test_C,lambda_val_C):
    lasso_regression_model_A,z_vector_LR_A,mse_LR_A = ComputeLassoRegression(X_train_A,X_test_A,y_train_A,y_test_A,lambda_val_A)
    lasso_regression_model_B,z_vector_LR_B,mse_LR_B = ComputeLassoRegression(X_train_B,X_test_B,y_train_B,y_test_B,lambda_val_B)
    lasso_regression_model_C,z_vector_LR_C,mse_LR_C = ComputeLassoRegression(X_train_C,X_test_C,y_train_C,y_test_C,lambda_val_C)
    return lasso_regression_model_A,z_vector_LR_A,mse_LR_A,lasso_regression_model_B,z_vector_LR_B,mse_LR_B,lasso_regression_model_C,z_vector_LR_C,mse_LR_C

#KFoldCrossValidation Function
def KFoldCrossValidation(X_data,y_data,model_type,lambda_val_list,K=10):
    #Determine testing batch size
    test_size = X_data.shape[0]//K

    #Initiate list
    average_mse_list = np.zeros([lambda_val_list.shape[0],2])
    average_mse_list[:,0] = lambda_val_list
    
    #Perform K-Fold CV for each lambda
    for i in range(lambda_val_list.shape[0]):
        #Initialize values
        lambda_val = lambda_val_list[i]
        total_mse = 0
        if i == 0:
            best_lambda_val = lambda_val
            best_mse = np.inf
        
        #Perform K-Cross CV
        for k in range(K):
            #Set starting and ending indices for testing data
            test_start_index = k*test_size
            test_end_index = min((k+1)*test_size,X_data.shape[0]) 

            #Set training and testing data
            X_train = np.vstack((X_data[:test_start_index,:],X_data[test_end_index:,:]))
            y_train = np.vstack((y_data[:test_start_index,:],y_data[test_end_index:,:]))
            X_test = X_data[test_start_index:test_end_index,:]
            y_test = y_data[test_start_index:test_end_index,:]
            
            #Compute regression and calculate MSE
            if model_type == 'ridge':
                model,z_vector,mse = ComputeRidgeRegression(X_train,X_test,y_train,y_test,lambda_val)
            elif model_type == 'lasso':
                model,z_vector,mse = ComputeLassoRegression(X_train,X_test,y_train,y_test,lambda_val)
            else:
                break
            total_mse += mse
        
        #Compute and record average MSE
        average_mse = total_mse/K
        average_mse_list[i,1] = average_mse

        #Keep best MSE and lambda value
        if average_mse < best_mse:
            best_mse = average_mse
            best_lambda_val = lambda_val
    return average_mse_list,best_mse,best_lambda_val

#ComputeBestLambdaAllModels Function
def ComputeBestLambdaAllModels(X_train_A,y_train_A,X_train_B,y_train_B,X_train_C,y_train_C,lambda_val_list,K):
    average_mse_list_RR_A,best_mse_RR_A,best_lambda_val_RR_A = KFoldCrossValidation(X_train_A,y_train_A,'ridge',lambda_val_list)
    average_mse_list_RR_B,best_mse_RR_B,best_lambda_val_RR_B = KFoldCrossValidation(X_train_B,y_train_B,'ridge',lambda_val_list)
    average_mse_list_RR_C,best_mse_RR_C,best_lambda_val_RR_C = KFoldCrossValidation(X_train_C,y_train_C,'ridge',lambda_val_list)
    average_mse_list_LR_A,best_mse_LR_A,best_lambda_val_LR_A = KFoldCrossValidation(X_train_A,y_train_A,'lasso',lambda_val_list)
    average_mse_list_LR_B,best_mse_LR_B,best_lambda_val_LR_B = KFoldCrossValidation(X_train_B,y_train_B,'lasso',lambda_val_list)
    average_mse_list_LR_C,best_mse_LR_C,best_lambda_val_LR_C = KFoldCrossValidation(X_train_C,y_train_C,'lasso',lambda_val_list)
    return average_mse_list_RR_A,best_mse_RR_A,best_lambda_val_RR_A,average_mse_list_RR_B,best_mse_RR_B,best_lambda_val_RR_B,average_mse_list_RR_C,best_mse_RR_C,best_lambda_val_RR_C,average_mse_list_LR_A,best_mse_LR_A,best_lambda_val_LR_A,average_mse_list_LR_B,best_mse_LR_B,best_lambda_val_LR_B,average_mse_list_LR_C,best_mse_LR_C,best_lambda_val_LR_C

#PrintOutputs Function
def PrintOutputs(mse_LinR_A,mse_LinR_B,mse_LinR_C,average_mse_list_RR_A,best_mse_RR_A,best_lambda_val_RR_A,average_mse_list_RR_B,best_mse_RR_B,best_lambda_val_RR_B,average_mse_list_RR_C,best_mse_RR_C,best_lambda_val_RR_C,average_mse_list_LR_A,best_mse_LR_A,best_lambda_val_LR_A,average_mse_list_LR_B,best_mse_LR_B,best_lambda_val_LR_B,average_mse_list_LR_C,best_mse_LR_C,best_lambda_val_LR_C,K):    
    print("************************************************************")
    print("################ Linear Regression################")
    print("Average Mean Squared Error Dataset A:",mse_LinR_A)
    print()
    print("Average Mean Squared Error Dataset B:",mse_LinR_B)
    print()
    print("Average Mean Squared Error Dataset C:",mse_LinR_C)
    print()
    print("************************************************************")
    print('################ Ridge Regression ('+str(K)+'-Fold CV)################')
    print("[Lambda Value, Average Mean Squared Error Dataset A")
    print(average_mse_list_RR_A)
    print("Best Average Mean Squared Error Dataset A:",best_mse_RR_A)
    print("Best lambda value Dataset A:",best_lambda_val_RR_A)
    print()
    print("[Lambda Value, Average Mean Squared Error Dataset B")
    print(average_mse_list_RR_B)
    print("Best Average Mean Squared Error Dataset B:",best_mse_RR_B)
    print("Best lambda value Dataset B:",best_lambda_val_RR_B)
    print()
    print("[Lambda Value, Average Mean Squared Error Dataset C")
    print(average_mse_list_RR_C)
    print("Best Average Mean Squared Error Dataset C:",best_mse_RR_C)
    print("Best lambda value Dataset C:",best_lambda_val_RR_C)
    print()
    print("************************************************************")
    print('################ Lasso Regression ('+str(K)+'-Fold CV)################')
    print("[Lambda Value, Average Mean Squared Error Dataset A")
    print(average_mse_list_LR_A)
    print("Best Average Mean Squared Error Dataset A:",best_mse_LR_A)
    print("Best lambda value Dataset A:",best_lambda_val_LR_A)
    print()
    print("[Lambda Value, Average Mean Squared Error Dataset B")
    print(average_mse_list_LR_B)
    print("Best Average Mean Squared Error Dataset B:",best_mse_LR_B)
    print("Best lambda value Dataset B:",best_lambda_val_LR_B)
    print()
    print("[Lambda Value, Average Mean Squared Error Dataset C")
    print(average_mse_list_LR_C)
    print("Best Average Mean Squared Error Dataset C:",best_mse_LR_C)
    print("Best lambda value Dataset C:",best_lambda_val_LR_C)
    print()
    print("************************************************************")
    print()
    return

#PlotHistogram Function
def PlotHistogram(z_vector_LinR,z_vector_RR,z_vector_LR,dataset_letter,K):
    num_bins = 10

    #Add each regression model parameters on histogram
    plt.hist(z_vector_LinR, bins=num_bins, alpha=0.3, label='Unregularized')
    plt.hist(z_vector_RR, bins=num_bins, alpha=0.3, label='Ridge')
    plt.hist(z_vector_LR, bins=num_bins, alpha=0.3, label='Lasso')
    plt.title('Histogram of Parameters Values For Data '+str(dataset_letter)+' ('+str(K)+'-Fold CV Used)')
    plt.xlabel('Parameter Values')
    plt.ylabel('Number of Occurences')
    plt.legend()
    plt.show()
    return

#PlotAllHistograms Function
def PlotAllHistograms(z_vector_LinR_A,z_vector_RR_A,z_vector_LR_A,z_vector_LinR_B,z_vector_RR_B,z_vector_LR_B,z_vector_LinR_C,z_vector_RR_C,z_vector_LR_C,K):
    PlotHistogram(z_vector_LinR_A,z_vector_RR_A,z_vector_LR_A,'A',K)
    PlotHistogram(z_vector_LinR_B,z_vector_RR_B,z_vector_LR_B,'B',K)
    PlotHistogram(z_vector_LinR_C,z_vector_RR_C,z_vector_LR_C,'C',K)
    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train_A = 'Data/X_train_A.csv'
filename_X_test_A = 'Data/X_test_A.csv'
filename_y_train_A = 'Data/Y_train_A.csv'
filename_y_test_A = 'Data/Y_test_A.csv'
filename_X_train_B = 'Data/X_train_B.csv'
filename_X_test_B = 'Data/X_test_B.csv'
filename_y_train_B = 'Data/Y_train_B.csv'
filename_y_test_B = 'Data/Y_test_B.csv'
filename_X_train_C = 'Data/X_train_C.csv'
filename_X_test_C = 'Data/X_test_C.csv'
filename_y_train_C = 'Data/Y_train_C.csv'
filename_y_test_C = 'Data/Y_test_C.csv'
lambda_val_list = np.arange(1,11)
K = 10

#################################################################################################################
### Main Code

X_train_A,X_test_A,y_train_A,y_test_A,X_train_B,X_test_B,y_train_B,y_test_B,X_train_C,X_test_C,y_train_C,y_test_C = PrepareTrainingAndTestData(filename_X_train_A,filename_X_test_A,filename_y_train_A,filename_y_test_A,filename_X_train_B,filename_X_test_B,filename_y_train_B,filename_y_test_B,filename_X_train_C,filename_X_test_C,filename_y_train_C,filename_y_test_C)
linear_regression_model_A,z_vector_LinR_A,mse_LinR_A,linear_regression_model_B,z_vector_LinR_B,mse_LinR_B,linear_regression_model_C,z_vector_LinR_C,mse_LinR_C = ComputeLinearRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,X_train_B,X_test_B,y_train_B,y_test_B,X_train_C,X_test_C,y_train_C,y_test_C)
average_mse_list_RR_A,best_mse_RR_A,best_lambda_val_RR_A,average_mse_list_RR_B,best_mse_RR_B,best_lambda_val_RR_B,average_mse_list_RR_C,best_mse_RR_C,best_lambda_val_RR_C,average_mse_list_LR_A,best_mse_LR_A,best_lambda_val_LR_A,average_mse_list_LR_B,best_mse_LR_B,best_lambda_val_LR_B,average_mse_list_LR_C,best_mse_LR_C,best_lambda_val_LR_C = ComputeBestLambdaAllModels(X_train_A,y_train_A,X_train_B,y_train_B,X_train_C,y_train_C,lambda_val_list,K)
ridge_regression_model_A,z_vector_RR_A,mse_RR_A,ridge_regression_model_B,z_vector_RR_B,mse_RR_B,ridge_regression_model_C,z_vector_RR_C,mse_RR_C = ComputeRidgeRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,best_lambda_val_RR_A,X_train_B,X_test_B,y_train_B,y_test_B,best_lambda_val_RR_B,X_train_C,X_test_C,y_train_C,y_test_C,best_lambda_val_RR_B)
lasso_regression_model_A,z_vector_LR_A,mse_LR_A,lasso_regression_model_B,z_vector_LR_B,mse_LR_B,lasso_regression_model_C,z_vector_LR_C,mse_LR_C = ComputeLassoRegressionAllModels(X_train_A,X_test_A,y_train_A,y_test_A,best_lambda_val_LR_A,X_train_B,X_test_B,y_train_B,y_test_B,best_lambda_val_LR_B,X_train_C,X_test_C,y_train_C,y_test_C,best_lambda_val_LR_B)
PrintOutputs(mse_LinR_A,mse_LinR_B,mse_LinR_C,average_mse_list_RR_A,best_mse_RR_A,best_lambda_val_RR_A,average_mse_list_RR_B,best_mse_RR_B,best_lambda_val_RR_B,average_mse_list_RR_C,best_mse_RR_C,best_lambda_val_RR_C,average_mse_list_LR_A,best_mse_LR_A,best_lambda_val_LR_A,average_mse_list_LR_B,best_mse_LR_B,best_lambda_val_LR_B,average_mse_list_LR_C,best_mse_LR_C,best_lambda_val_LR_C,K)
PlotAllHistograms(z_vector_LinR_A,z_vector_RR_A,z_vector_LR_A,z_vector_LinR_B,z_vector_RR_B,z_vector_LR_B,z_vector_LinR_C,z_vector_RR_C,z_vector_LR_C,K)
