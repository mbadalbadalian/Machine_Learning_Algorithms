import numpy as np
import matplotlib.pyplot as plt

#################################################################################################################
### Other Functions

#LoadTrainingAndTestingData Function
def LoadTrainingAndTestingData(gmm_dataset_filepath):
    #Load GMM data
    GMM_dataset = np.genfromtxt(gmm_dataset_filepath, delimiter=',')
    return GMM_dataset

#CreateInitialModel Function
def CreateInitialModel(X,K):
    #Initialize model parameters
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
    EPSILON = 1e-6
    l = []
    for iter in range(1,MAXITER+1):
        r = np.zeros([n,K])
        #Expectation step
        for k in range(1,K+1):
            #Compute responsibility
            #Handle division by zero
            if np.isnan(model['pi'][k-1]*np.prod(1.0/np.sqrt(2*np.pi*model['S'][k-1])*np.exp(-0.5*((X-model['mu'][k-1])**2)/model['S'][k-1]),axis=1)).any():
                r[:,k-1] = model['pi'][k-1]*np.prod(1.0/(np.sqrt(2*np.pi*model['S'][k-1])+EPSILON)*np.exp(-0.5*((X-model['mu'][k-1])**2)/model['S'][k-1]),axis=1)
                continue
            r[:,k-1] = model['pi'][k-1]*np.prod(1.0/np.sqrt(2*np.pi*model['S'][k-1])*np.exp(-0.5*((X-model['mu'][k-1])**2)/model['S'][k-1]),axis=1)
        
        #Normalize responsibility
        r_idot = np.sum(r,axis=1,keepdims=True)
        r = r/r_idot
        
        #Compute negative log-likelihood
        l.append(-1*np.sum(np.log(r_idot)))
        if (iter > 1) and (np.abs(l[iter-1]-l[iter-2]) <= TOL*np.abs(l[iter-1])):
            break
        
        #Maximization step
        for k in range(1,K+1):
            r_dotk = np.sum(r[:,k-1])
            
            #Compute model parameters
            model['pi'][k-1] = r_dotk/n
            model['mu'][k-1] = np.sum(r[:,k-1].reshape(-1,1)*X,axis=0)/r_dotk    
            model['S'][k-1] = np.sum(r[:,k-1].reshape(-1,1)*((X - model['mu'][k-1])**2),axis=0)/r_dotk
    l = np.array(l)
    return model,l

#ComputeAllDiagonalGMMModels Function
def ComputeAllDiagonalGMMModels(GMM_dataset,K,negative_log_likelihood_list_filepath):
    negative_log_likelihood_list = np.zeros(K)
    for k in range(1,K+1):
        initial_diagonal_GMM_model = CreateInitialModel(GMM_dataset,k)
        diagonal_GMM_model,l = EMAlgorithmForDiagonalGMM(GMM_dataset,k,initial_diagonal_GMM_model)
        negative_log_likelihood_list[k-1] = l[-1]
    np.save(negative_log_likelihood_list_filepath,negative_log_likelihood_list)
    return negative_log_likelihood_list

#CreateOrLoadNegativeLogLikelihoodList Function
def CreateOrLoadNegativeLogLikelihoodList(GMM_dataset,K,negative_log_likelihood_list_filepath,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        negative_log_likelihood_list = ComputeAllDiagonalGMMModels(GMM_dataset,K,negative_log_likelihood_list_filepath)
    else:
        negative_log_likelihood_list = np.load(negative_log_likelihood_list_filepath)
    return negative_log_likelihood_list

#CreatePlots Function
def CreatePlots(K,negative_log_likelihood_list):
    K_list = np.arange(1,K+1)
    plt.figure()
    plt.plot(K_list,negative_log_likelihood_list)
    plt.xlabel('K values')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood For A Range Of K Values')
    plt.grid(True)
    plt.show()
    return

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

#PrintOutputs Function
def PrintOutputs(best_k,sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors):
    print("************************************************************")
    print('################ Part a) Parameters For Diagonal GMM Using K = '+str(best_k)+' ################')
    print('Parameters For Diagonal')
    print()
    print('Sorted Mixing Weights:')
    print(sorted_mixing_weights)
    print()
    print('Sorted Mean Vector:')
    print(sorted_mean_vectors)
    print()
    print('Sorted Diagonals Of The Covariance Matrix:')
    print(sorted_covariance_diagonal_vectors)
    print("************************************************************")
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    GMM_dataset_filepath = 'Data/gmm_dataset.csv'
    K = 10
    negative_log_likelihood_list_filepath = 'Created_Data/A3_E3_Q1_a/negative_log_likelihood_list.npy'
    create_or_load_string = 'Load'
    best_K = 6
    sorted_mixing_weights_filepath = 'Created_Data/A3_E3_Q1_a/sorted_mixing_weights.npy'
    sorted_mean_vectors_filepath = 'Created_Data/A3_E3_Q1_a/sorted_mean_vectors.npy'
    sorted_covariance_diagonal_vectors_filepath = 'Created_Data/A3_E3_Q1_a/sorted_covariance_diagonal_vectors.npy' 
    create_or_load_string_best_model = 'Load'
    
    #Main Code
    GMM_dataset = LoadTrainingAndTestingData(GMM_dataset_filepath)
    negative_log_likelihood_list = CreateOrLoadNegativeLogLikelihoodList(GMM_dataset,K,negative_log_likelihood_list_filepath,create_or_load_string)
    CreatePlots(K,negative_log_likelihood_list)
    sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors = CreateOrLoadBestModelParameters(GMM_dataset,best_K,sorted_mixing_weights_filepath,sorted_mean_vectors_filepath,sorted_covariance_diagonal_vectors_filepath,create_or_load_string_best_model)
    PrintOutputs(best_K,sorted_mixing_weights,sorted_mean_vectors,sorted_covariance_diagonal_vectors)
    