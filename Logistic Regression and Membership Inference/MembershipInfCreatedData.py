import numpy as np
import matplotlib.pyplot as plt

#################################################################################################################
### Other Functions

def GenerateDataset(d_value, n_value=50):
    X_data = np.random.normal(0, 1, size=(n_value, d_value))
    y_data = np.random.normal(0, 1, size=(n_value, d_value))
    empirical_mean = np.mean(X_data, axis=0)
    return X_data, y_data, empirical_mean

def CalcBestAccuracyWithThresholds(X_data, y_data, empirical_mean):
    X_thresholds = np.dot(X_data, empirical_mean)
    y_thresholds = np.dot(y_data, empirical_mean)
    thresholds = np.concatenate([X_thresholds, y_thresholds])

    best_accuracy = -1
    best_threshold = -1

    for threshold in thresholds:
        X_predictions = np.dot(X_data, empirical_mean.T)
        y_predictions = np.dot(y_data, empirical_mean.T)

        correct_predictions_X = np.sum(X_predictions >= threshold)
        correct_predictions_Y = np.sum(y_predictions < threshold)

        correct_predictions = correct_predictions_X + correct_predictions_Y
        accuracy = correct_predictions / (2*len(X_data))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def CalculateBestAccuracyPerDimension(d_value_list, n_value=50):
    best_thresholds_list = []
    best_accuracy_list = []

    for d_value in d_value_list:
        X_data, y_data, empirical_mean = GenerateDataset(d_value, n_value)
        best_threshold, best_accuracy = CalcBestAccuracyWithThresholds(X_data, y_data, empirical_mean)
        best_thresholds_list.append(best_threshold)
        best_accuracy_list.append(best_accuracy)

    return best_thresholds_list, best_accuracy_list

def CalcAccuracy(X_data, y_data, empirical_mean, threshold):
    X_thresholds = np.dot(X_data, empirical_mean)
    y_thresholds = np.dot(y_data, empirical_mean)

    correct_predictions_X = np.sum(X_thresholds >= threshold)
    correct_predictions_Y = np.sum(y_thresholds < threshold)

    correct_predictions = correct_predictions_X + correct_predictions_Y
    accuracy = correct_predictions/(2*len(X_data))

    return accuracy

def CalculateAccuracyUsingKnownThresholds(best_thresholds_list,d_value_list,n_value=50):
    accuracy_list = []
    i = 0
    for d_value in d_value_list:
        X_data,y_data,empirical_mean = GenerateDataset(d_value,n_value)
        accuracy = CalcAccuracy(X_data,y_data,empirical_mean,best_thresholds_list[i])
        accuracy_list.append(accuracy)
        i = i + 1
    return accuracy_list

def GenerateDatasetWithDifferentialPrivacy(d_value,n_value,sigma_squared):
    X_data, Y_data, empirical_mean = GenerateDataset(d_value, n_value)
    differential_privacy = np.random.normal(0, np.sqrt(sigma_squared), size=(d_value,))
    private_mean = empirical_mean + differential_privacy
    return X_data, Y_data, private_mean

def CalculateBestAccuracyDifferentialPrivacyThreshold(X_data, Y_data, private_mean):
    X_thresholds = np.dot(X_data, private_mean)
    Y_thresholds = np.dot(Y_data, private_mean)
    thresholds = np.concatenate([X_thresholds, Y_thresholds])
    accuracy_list = []
    best_accuracy = -1
    
    for threshold in thresholds:
        accuracy = CalcAccuracy(X_thresholds, Y_thresholds, private_mean, threshold)
        accuracy_list.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_accuracy, best_threshold

def PerformExperiment(d_value, n_value, num_trials=1000):
    sigma_squared_values_list = np.linspace(0, 1, d_value + 1)
    accuracy_estimate_list = []
    accuracy_attack_list = []
    for sigma_squared in sigma_squared_values_list:    
        accuracy_estimate = 0
        accuracy_attack = 0
        for trial in range(num_trials):
            X_data, Y_data, private_mean = GenerateDatasetWithDifferentialPrivacy(d_value, n_value, sigma_squared)
            accuracy, _ = CalculateBestAccuracyDifferentialPrivacyThreshold(X_data, Y_data, private_mean)
            accuracy_estimate += np.linalg.norm(private_mean)**2
            accuracy_attack += accuracy
        accuracy_estimate = accuracy_estimate/num_trials
        accuracy_attack = accuracy_attack/num_trials
        accuracy_estimate_list.append(accuracy_estimate)
        accuracy_attack_list.append(accuracy_attack)
    return sigma_squared_values_list,accuracy_estimate_list,accuracy_attack_list

def CreateAllPlots(d_value_list,best_accuracy_list,accuracy_list_using_known_thresholds,sigma_squared_values_list,accuracy_estimate_list,accuracy_attack_list):
    plt.figure()
    plt.plot(d_value_list,best_accuracy_list)
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.title('Membership Inference Attack Accuracy vs Data Dimension (With Unknown Threshold)')
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()
    plt.plot(d_value_list,accuracy_list_using_known_thresholds)
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.title('Membership Inference Attack Accuracy vs Data Dimension (With Known Thresholds)')
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(sigma_squared_values_list,accuracy_estimate_list)
    plt.xlabel('σ^2')
    plt.ylabel('Estimate Squared')
    plt.title('Estimate Accuracy vs Variance (Sigma Squared)')
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(sigma_squared_values_list,accuracy_attack_list)
    plt.xlabel('Variance (Sigma Squared)')
    plt.ylabel('Membership Inference Attack Accuracy')
    plt.title('Membership Inference Attack Accuracy vs Variance (Sigma Squared)')
    plt.grid(True)
    plt.show()
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":    
    #Variables
    d_value_list = range(10,501)
    n_value = 50
    d_fixed = 50
    num_trials = 1000
    
    #Main Code
    best_thresholds_list,best_accuracy_list = CalculateBestAccuracyPerDimension(d_value_list,n_value)
    accuracy_list_using_known_thresholds = CalculateAccuracyUsingKnownThresholds(best_thresholds_list,d_value_list,n_value)
    sigma_squared_values_list,accuracy_estimate_list,accuracy_attack_list = PerformExperiment(d_fixed,n_value,num_trials)
    
    CreateAllPlots(d_value_list,best_accuracy_list,accuracy_list_using_known_thresholds,sigma_squared_values_list,accuracy_estimate_list,accuracy_attack_list)