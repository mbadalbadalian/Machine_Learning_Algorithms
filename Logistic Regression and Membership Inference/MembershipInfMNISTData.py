import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#################################################################################################################
### Other Functions

def LoadMNISTData():
    mnist = fetch_openml('Fashion-MNIST',version=1)
    X_data,y_data = mnist.data.astype('float32')/255, mnist.target.astype('int')
    return X_data,y_data

def TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,regularization,sigma_squared=None):
    if regularization == 'l2':
        logistic_regression_model = LogisticRegression(C=0.01,penalty='l2',solver='lbfgs',max_iter=2500)
    else:
        logistic_regression_model = LogisticRegression(C=1e10,solver='lbfgs',max_iter=5000)

    logistic_regression_model.fit(X_train,y_train)
    if sigma_squared is not None:
        noise = np.random.normal(0, sigma_squared, size=logistic_regression_model.coef_.shape[0])
        noise = noise.reshape(-1, 1)
        logistic_regression_model.coef_ += noise
    IN_accuracy = accuracy_score(y_train,logistic_regression_model.predict(X_train))
    OUT_accuracy = accuracy_score(y_test,logistic_regression_model.predict(X_test))
    return logistic_regression_model,IN_accuracy,OUT_accuracy

def MembershipInferenceAttackOnModel(logistic_regression_model,X_train,y_train,X_test,y_test):
    training_predictions = logistic_regression_model.predict(X_train)
    testing_predictions = logistic_regression_model.predict(X_test)
    attack_accuracy = (accuracy_score(y_train,training_predictions) + (1-accuracy_score(y_test,testing_predictions)))/2
    return attack_accuracy

def MeasureAttackWithVaryingNoise(X_train,y_train,X_test,y_test,regularization,sigma_squared_list=None):
    attack_accuracy_list = []
    for sigma_squared in sigma_squared_list:
        logistic_regression_model, _, _ = TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,regularization,sigma_squared)
        attack_accuracy = MembershipInferenceAttackOnModel(logistic_regression_model,X_train,y_train,X_test,y_test)
        attack_accuracy_list.append(attack_accuracy)
    return attack_accuracy_list

def TrainAndEvaluateAllModels(number_of_data_points_list,sigma_squared_list,number_of_data_points_for_attack_with_varying_noise=400):
    X_data,y_data = LoadMNISTData()
    unregularized_results = {'number_of_data_points': [], 'training_IN_accuracy': [], 'testing_OUT_accuracy': [], 'attack_accuracy': [], 'sigma_squared_list': [], 'number_of_points_used_for_attack_with_varying_noise_list_accuracy': [], 'training_IN_accuracy_with_varying_noise': [], 'testing_OUT_accuracy_with_varying_noise': [], 'attack_with_varying_noise_list_accuracy': []}
    regularized_results = {'number_of_data_points': [], 'training_IN_accuracy': [], 'testing_OUT_accuracy': [], 'attack_accuracy': [], 'sigma_squared_list': [], 'number_of_points_used_for_attack_with_varying_noise_list_accuracy': [], 'training_IN_accuracy_with_varying_noise': [], 'testing_OUT_accuracy_with_varying_noise': [], 'attack_with_varying_noise_list_accuracy': []}
    
    for number_of_data_points in number_of_data_points_list:
        X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,train_size=number_of_data_points,stratify=y_data)

        logistic_regression_model_unregularized,logistic_regression_model_unregularized_IN_accuracy,logistic_regression_model_unregularized_OUT_accuracy = TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,'None')
        logistic_regression_model_regularized,logistic_regression_model_regularized_IN_accuracy,logistic_regression_model_regularized_OUT_accuracy = TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,'l2')

        attack_accuracy_unregularized = MembershipInferenceAttackOnModel(logistic_regression_model_unregularized,X_train,y_train,X_test,y_test)
        attack_accuracy_regularized = MembershipInferenceAttackOnModel(logistic_regression_model_regularized,X_train,y_train,X_test,y_test)

        unregularized_results['number_of_data_points'].append(number_of_data_points)
        unregularized_results['training_IN_accuracy'].append(logistic_regression_model_unregularized_IN_accuracy)
        unregularized_results['testing_OUT_accuracy'].append(logistic_regression_model_unregularized_OUT_accuracy)
        unregularized_results['attack_accuracy'].append(attack_accuracy_unregularized)

        regularized_results['number_of_data_points'].append(number_of_data_points)
        regularized_results['training_IN_accuracy'].append(logistic_regression_model_regularized_IN_accuracy)
        regularized_results['testing_OUT_accuracy'].append(logistic_regression_model_regularized_OUT_accuracy)
        regularized_results['attack_accuracy'].append(attack_accuracy_regularized)
    
    X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,train_size=number_of_data_points_for_attack_with_varying_noise,stratify=y_data)
    
    logistic_regression_model_unregularized_IN_accuracy_with_varying_noise_list = []
    logistic_regression_model_unregularized_OUT_accuracy_with_varying_noise_list = []
    logistic_regression_model_regularized_IN_accuracy_with_varying_noise_list = []
    logistic_regression_model_regularized_OUT_accuracy_with_varying_noise_list = []
    for sigma_squared in sigma_squared_list:
        logistic_regression_model_unregularized,logistic_regression_model_unregularized_IN_accuracy_with_varying_noise,logistic_regression_model_unregularized_OUT_accuracy_with_varying_noise = TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,'None',sigma_squared)
        logistic_regression_model_regularized,logistic_regression_model_regularized_IN_accuracy_with_varying_noise,logistic_regression_model_regularized_OUT_accuracy_with_varying_noise = TrainModelUsingLogisticRegression(X_train,y_train,X_test,y_test,'l2',sigma_squared)
        logistic_regression_model_unregularized_IN_accuracy_with_varying_noise_list.append(logistic_regression_model_unregularized_IN_accuracy_with_varying_noise)
        logistic_regression_model_unregularized_OUT_accuracy_with_varying_noise_list.append(logistic_regression_model_unregularized_OUT_accuracy_with_varying_noise)
        logistic_regression_model_regularized_IN_accuracy_with_varying_noise_list.append(logistic_regression_model_regularized_IN_accuracy_with_varying_noise)
        logistic_regression_model_regularized_OUT_accuracy_with_varying_noise_list.append(logistic_regression_model_regularized_OUT_accuracy_with_varying_noise)

    attack_with_varying_noise_list_accuracy_unregularized = MeasureAttackWithVaryingNoise(X_train,y_train,X_test,y_test,'None',sigma_squared_list)
    attack_with_varying_noise_list_accuracy_regularized = MeasureAttackWithVaryingNoise(X_train,y_train,X_test,y_test,'l2',sigma_squared_list) 

    unregularized_results['sigma_squared_list'] = sigma_squared_list
    regularized_results['sigma_squared_list'] = sigma_squared_list
    
    unregularized_results['number_of_points_used_for_attack_with_varying_noise_list_accuracy'].append(number_of_data_points)
    regularized_results['number_of_points_used_for_attack_with_varying_noise_list_accuracy'].append(number_of_data_points)
    
    unregularized_results['training_IN_accuracy_with_varying_noise'].append(logistic_regression_model_unregularized_IN_accuracy_with_varying_noise_list)
    regularized_results['training_IN_accuracy_with_varying_noise'].append(logistic_regression_model_regularized_IN_accuracy_with_varying_noise_list)
    
    unregularized_results['testing_OUT_accuracy_with_varying_noise'].append(logistic_regression_model_unregularized_OUT_accuracy_with_varying_noise_list)
    regularized_results['testing_OUT_accuracy_with_varying_noise'].append(logistic_regression_model_regularized_OUT_accuracy_with_varying_noise_list)
    
    unregularized_results['attack_with_varying_noise_list_accuracy'].append(attack_with_varying_noise_list_accuracy_unregularized)
    regularized_results['attack_with_varying_noise_list_accuracy'].append(attack_with_varying_noise_list_accuracy_regularized)
    return unregularized_results,regularized_results

def CreateAllPlots(unregularized_results,regularized_results):
    plt.figure()
    plt.plot(unregularized_results['number_of_data_points'],unregularized_results['training_IN_accuracy'],label='Training Accuracy')
    plt.plot(unregularized_results['number_of_data_points'],unregularized_results['testing_OUT_accuracy'],label='Testing Accuracy')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.title('Unregularized Logistic Regression Model Error vs Number of Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(regularized_results['number_of_data_points'],regularized_results['training_IN_accuracy'],label='Training Accuracy')
    plt.plot(regularized_results['number_of_data_points'],regularized_results['testing_OUT_accuracy'],label='Testing Accuracy')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.title('Regularized Logistic Regression Model Error vs Number of Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(unregularized_results['number_of_data_points'],unregularized_results['attack_accuracy'],label='Unregularized Model')
    plt.plot(regularized_results['number_of_data_points'],regularized_results['attack_accuracy'],label='Regularized Model')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Attack Accuracy')
    plt.title('Unregularized and Regularized Model Attack Accuracy vs Number of Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(unregularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,unregularized_results['training_IN_accuracy_with_varying_noise']),label='Training Accuracy With Noise')
    plt.plot(unregularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,unregularized_results['testing_OUT_accuracy_with_varying_noise']),label='Testing Accuracy With Noise')
    plt.xlabel('Sigma Squared')
    plt.ylabel('Accuracy')
    plt.title('Unregularized Logistic Regression Model (With Noise) Error vs Sigma Squared')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(regularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,regularized_results['training_IN_accuracy_with_varying_noise']),label='Training Accuracy With Noise')
    plt.plot(regularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,regularized_results['testing_OUT_accuracy_with_varying_noise']),label='Testing Accuracy With Noise')
    plt.xlabel('Sigma Squared')
    plt.ylabel('Accuracy')
    plt.title('Regularized Logistic Regression Model (With Noise) Error vs Sigma Squared')
    plt.legend()
    plt.grid(True)
    plt.show() 
    
    plt.figure()
    plt.plot(unregularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,unregularized_results['attack_with_varying_noise_list_accuracy']),label='Unregularized Model')
    plt.plot(regularized_results['sigma_squared_list'],np.full_like(sigma_squared_list,regularized_results['attack_with_varying_noise_list_accuracy']),label='Regularized Model')
    plt.xlabel('Sigma Squared')
    plt.ylabel('Attack Accuracy')
    plt.title('Unregularized and Regularized Model (With Noise) Attack Accuracy vs Sigma Squared')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":    
    #Variables
    number_of_data_points_list = [100,200,400,800,1600,2500,5000,10000]
    sigma_squared_list = np.linspace(0,5,10)
    number_of_data_points_for_attack_with_varying_noise = 400
    
    #Main Code
    unregularized_results,regularized_results = TrainAndEvaluateAllModels(number_of_data_points_list,sigma_squared_list,number_of_data_points_for_attack_with_varying_noise)
    CreateAllPlots(unregularized_results,regularized_results)