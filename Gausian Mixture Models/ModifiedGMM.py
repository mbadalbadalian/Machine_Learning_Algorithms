import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#################################################################################################################
### Other Functions

#LoadTrainingAndTestingData Function
def LoadTrainingAndTestingData():
    #Obtain training and testing data from MNIST dataset
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    
    #Flatten input images
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    
    #Normalize input data
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/255.0
    
    #Perform PCA on input data
    PCA_method = PCA(n_components=50)
    X_train = PCA_method.fit_transform(X_train)
    X_test = PCA_method.transform(X_test)
    return X_train,y_train,X_test,y_test

#CreateInitialModel Function
def CreateInitialModel(X,K):
    #Initialize model parameters
    EPSILON = 1e-6
    n,d = X.shape
    initial_model = {}
    initial_model['K'] = K
    initial_model['pi'] = np.ones(K)/K
    initial_model['mu'] = X[np.random.choice(n,K,replace=False)]
    initial_model['S'] = np.array([np.var(X,axis=0) for k in range(1,K+1)])
    return initial_model

#EMAlgorithmForDiagonalGMM Function
def EMAlgorithmForDiagonalGMM(X,K,initial_model):
    #Initialize values
    model = initial_model
    MAXITER = 500
    TOL = 1e-5
    n,d = X.shape
    EPSILON = 1e-9
    l = []
    for iter in range(1,MAXITER+1):
        r = np.zeros([n,K])
        #Expectation step
        for k in range(1,K+1):
            #Compute responsibility
            #Handle division by zero
            sqrt_data = 2*np.pi*model['S'][k-1]
            sqrt_data = np.where(sqrt_data <= 0, EPSILON, sqrt_data)
            model_s_kminus1 = model['S'][k-1]
            model_s_kminus1 = np.where(model_s_kminus1 == 0, EPSILON, model_s_kminus1)
            r[:,k-1] = model['pi'][k-1]*np.prod(1.0/np.sqrt(sqrt_data)*np.exp(-0.5*((X-model['mu'][k-1])**2)/(model_s_kminus1)),axis=1)
        
        #Normalize responsibility
        #Handle division by zero
        r_idot = np.sum(r,axis=1,keepdims=True)
        r = np.where(np.isnan(r), EPSILON, r)
        r_idot = np.where(r_idot == 0, EPSILON, r_idot)
        r = r/(r_idot)
        
        #Compute negative log-likelihood
        l.append(-1*np.sum(np.log(r_idot)))
        if (iter > 1) and (np.abs(l[iter-1]-l[iter-2]) <= TOL*np.abs(l[iter-1])):
            break
        
        #Maximization step
        for k in range(1,K+1):
            r_dotk = np.sum(r[:,k-1])+EPSILON
            
            #Compute model parameters
            model['pi'][k-1] = r_dotk/n
            model['mu'][k-1] = np.sum(r[:,k-1].reshape(-1,1)*X,axis=0)/r_dotk    
            model['S'][k-1] = np.sum(r[:,k-1].reshape(-1,1)*((X - model['mu'][k-1])**2),axis=0)/r_dotk
    l = np.array(l)
    return model,l

#GetBestGMMModelParameters Function
def GetBestGMMModelParameters(GMM_dataset,best_K,sorted_mixing_weights_filepath,sorted_mean_vectors_filepath,sorted_covariance_diagonal_vectors_filepath):
    #Initialize optimal GMM model
    best_initial_diagonal_GMM_model = CreateInitialModel(GMM_dataset,best_K)
    
    #Create GMM model
    best_diagonal_GMM_model,l = EMAlgorithmForDiagonalGMM(GMM_dataset,best_K,best_initial_diagonal_GMM_model)
    
    #Sort model parameters according to mixing weights increasing order
    sorted_mixing_weights = best_diagonal_GMM_model['pi'][np.argsort(best_diagonal_GMM_model['pi'])]
    sorted_mean_vectors = best_diagonal_GMM_model['mu'][np.argsort(best_diagonal_GMM_model['pi'])]
    sorted_covariance_diagonal_vectors = best_diagonal_GMM_model['S'][np.argsort(best_diagonal_GMM_model['pi'])]
    
    #Save model parameters
    np.save(sorted_mixing_weights_filepath,sorted_mixing_weights)
    np.save(sorted_mean_vectors_filepath,sorted_mean_vectors)
    np.save(sorted_covariance_diagonal_vectors_filepath,sorted_covariance_diagonal_vectors)
    return sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors

#CreateOrLoadBestModelParameters Function
def CreateOrLoadBestModelParameters(GMM_dataset,best_K,sorted_mixing_weights_filepath,sorted_mean_vectors_filepath,sorted_covariance_diagonal_vectors_filepath,create_or_load_string_best_model='load'):
    if create_or_load_string_best_model in ['Create','create']:
        #Create sorted model parameters
        sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors = GetBestGMMModelParameters(GMM_dataset,best_K,sorted_mixing_weights_filepath,sorted_mean_vectors_filepath,sorted_covariance_diagonal_vectors_filepath)
    else:
        #Load sorted model parameters
        sorted_mixing_weights = np.load(sorted_mixing_weights_filepath)
        sorted_mean_vectors = np.load(sorted_mean_vectors_filepath)
        sorted_covariance_diagonal_vectors = np.load(sorted_covariance_diagonal_vectors_filepath)
    return sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors

#CreateInitialBayesClassifierModel Function
def CreateInitialBayesClassifierModel(X_train,X_test,k):
    #Initialize model parameters
    bayes_classifier_models = {}
    bayes_classifier_models['Digit'] = np.arange(10)    
    bayes_classifier_models['K'] = k
    bayes_classifier_models['pi'] = np.zeros([10,k])
    bayes_classifier_models['mu'] = np.zeros([10,k,X_train.shape[1]])
    bayes_classifier_models['S'] = np.zeros([10,k,X_train.shape[1]])
    bayes_classifier_models['Weights'] = np.zeros(10)
    bayes_classifier_models['Probabilities'] = np.zeros([10,X_test.shape[0]])
    return bayes_classifier_models

#ObtainMNISTBayesModels Function
def ObtainMNISTBayesModels(X_train,y_train,X_test,y_test,k):
    #Initialize bayes classifier models
    bayes_classifier_models = CreateInitialBayesClassifierModel(X_train,X_test,k)
    for current_digit in range(10):
        X_digit = X_train[y_train==current_digit]
        initial_model = CreateInitialModel(X_digit,k)
        model,l = EMAlgorithmForDiagonalGMM(X_digit,k,initial_model)
        
        #Update model parameters
        bayes_classifier_models['pi'][current_digit] = model['pi']
        bayes_classifier_models['mu'][current_digit] = model['mu']
        bayes_classifier_models['S'][current_digit] = model['S']
        bayes_classifier_models['Weights'][current_digit] = X_digit.shape[0]/X_train.shape[0]
    return bayes_classifier_models

#BayesClassifierPredict Function
def BayesClassifierPredict(X_test,bayes_classifier_models):
    for current_digit in range(10):
        EPSILON = 1e-6
        
        #Perform computations
        reversed_variance = 1/bayes_classifier_models['S'][current_digit]
        quadratic_term = -0.5*((X_test**2)@reversed_variance.T - 2*X_test@(bayes_classifier_models['mu'][current_digit]*reversed_variance).T + np.sum((bayes_classifier_models['mu'][current_digit]**2)*reversed_variance,axis=1)) 
        pi_term = np.log(bayes_classifier_models['pi'][current_digit] + EPSILON) - 0.5*np.sum(np.log(bayes_classifier_models['S'][current_digit] + EPSILON),axis=1)
        log_S_term = -0.5*np.sum(np.log(bayes_classifier_models['S'][current_digit] + EPSILON),axis=1)
        
        #Reshape terms
        pi_term = pi_term.reshape((1, -1))
        log_S_term = log_S_term.reshape((1, -1))
        
        #Compute log-likelihood matrix
        log_r_matrix = quadratic_term + pi_term + log_S_term
        log_r_matrix = log_r_matrix.T
        
        #Update class probabilities
        max_value = np.max(log_r_matrix,axis=0,keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(log_r_matrix - max_value),axis=0)) + max_value.squeeze()
        bayes_classifier_models['Probabilities'][current_digit] = log_sum_exp*bayes_classifier_models['Weights'][current_digit]
    #Compute prediction
    y_pred = np.argmax(bayes_classifier_models['Probabilities'],axis=0)
    return y_pred

#CalculateErrorRate Function
def CalculateErrorRate(y_test,y_pred):
    error_rate = 1 - (np.sum(y_pred==y_test)/y_test.shape[0])
    return error_rate

#CalculateErrorRatesForBayesClassifier Function
def CalculateErrorRatesForBayesClassifier(X_train,y_train,X_test,y_test,K,error_rate_list_filepath):
    error_rate_list = []
    for k in range(1,K+1):
        bayes_classifier_models = ObtainMNISTBayesModels(X_train,y_train,X_test,y_test,k)
        y_pred = BayesClassifierPredict(X_test,bayes_classifier_models)
        error_rate = CalculateErrorRate(y_test,y_pred)
        error_rate_list.append(error_rate)
    error_rate_list = np.array(error_rate_list)
    np.save(error_rate_list_filepath,error_rate_list)
    return error_rate_list

#CreateOrLoadErrorRateList Function
def CreateOrLoadErrorRateList(X_train,y_train,X_test,y_test,K,error_rate_list_filepath,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        error_rate_list = CalculateErrorRatesForBayesClassifier(X_train,y_train,X_test,y_test,K,error_rate_list_filepath)
    else:
        error_rate_list = np.load(error_rate_list_filepath)
    return error_rate_list

#PrintOutputs Function
def PrintOutputs(error_rate_list):
    print("************************************************************")
    print('################ Part b) ################')
    print('Error Rate (K = 1 to 5):')
    print(error_rate_list)
    print("************************************************************")
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    K = 5
    error_rate_list_filepath = 'Created_Data/A3_E3_Q1_b/error_rate_list.npy'
    create_or_load_string = 'Load'
    
    #Main Code
    X_train,y_train,X_test,y_test = LoadTrainingAndTestingData()
    error_rate_list = CreateOrLoadErrorRateList(X_train,y_train,X_test,y_test,K,error_rate_list_filepath,create_or_load_string)
    PrintOutputs(error_rate_list)
    