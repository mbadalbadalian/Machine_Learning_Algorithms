import numpy as np
import matplotlib.pyplot as plt
import random

#################################################################################################################
### Other Functions

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

#CreateBinaryDecisionTreeModel Function
def CreateBinaryDecisionTreeModel(X_train,y_train,X_test,y_test,error_method_name,max_depth):
    #Create binary_decision_tree_model
    binary_decision_tree_model = BinaryDecisionTree(error_method_name,max_depth)
    
    #Fit data
    binary_decision_tree_model.FitData(X_train,y_train)

    #Compute training and testing predictions
    y_pred_train = binary_decision_tree_model.PredictAllLabels(X_train)
    y_pred_test = binary_decision_tree_model.PredictAllLabels(X_test)

    #Compute training and testing accuracies
    training_accuracy = CalcAccuracy(y_train,y_pred_train)
    testing_accuracy = CalcAccuracy(y_test,y_pred_test)
    return binary_decision_tree_model,y_pred_test,training_accuracy,testing_accuracy

#CalcAccuracy Function
def CalcAccuracy(y_true,y_pred):
    accuracy = 1 - np.mean(np.abs(y_true - y_pred))
    return accuracy

#PrintOutputs Function
def PrintOutputs():
    return

#################################################################################################################
### Class

#Create a class node
class Node():
    #Define constructor method
    def __init__(self,best_variable_index=None,threshold=None,left_branch=None,right_branch=None,error=None,leaf_value=None):        
        #Decision Node
        self.best_variable_index = best_variable_index
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.error = error

        #Leaf Node
        self.leaf_value = leaf_value
        return

#Create a class for binary decision trees
class BinaryDecisionTree():
    #Define constructor method
    def __init__(self,error_method_name='Entropy Loss',max_depth=None):
        #Initalize object parameters
        self.root = None
        self.error_method_name = error_method_name
        self.max_depth = max_depth
        self.min_samples_split = 2
        return

    #BuildBinaryDecisionTree Function
    def BuildBinaryDecisionTree(self,X_train,y_train,curr_depth=0):
        #Ensure tree has not reached max depth and has remmaining data values to split
        if curr_depth<=self.max_depth and X_train.shape[0] > self.min_samples_split:
            #Get values corresponding to the best split
            best_variable_index,threshold,X_train_left,y_train_left,X_train_right,y_train_right,error = self.GetBestSplit(X_train,y_train)
            #Checking to make sure data is not pure
            if error != 0:
                left_bin_decision_tree = self.BuildBinaryDecisionTree(X_train_left,y_train_left,curr_depth+1)
                right_bin_decision_tree = self.BuildBinaryDecisionTree(X_train_right,y_train_right,curr_depth+1)
                return Node(best_variable_index,threshold,left_bin_decision_tree,right_bin_decision_tree,error)

        #Calculate leaf value
        calculated_leaf_value = self.CalculateLeafValue(y_train)
        return Node(leaf_value=calculated_leaf_value)
    
    #GetBestSplit Function
    def GetBestSplit(self,X_train,y_train):
        #Initialize values
        max_error = np.inf
        best_variable_index = 0
        threshold = X_train[0,0]

        #Split data accordingly
        X_train_left,y_train_left,X_train_right,y_train_right = self.SplitData(0,threshold,X_train,y_train)
        
        #Find best way to split data
        for curr_variable_index in range(X_train.shape[1]):
            for curr_threshold in np.unique(X_train[:,curr_variable_index]):
                curr_X_train_left,curr_y_train_left,curr_X_train_right,curr_y_train_right = self.SplitData(curr_variable_index,curr_threshold,X_train,y_train)
                if (curr_y_train_left.shape[0]>0) and (curr_y_train_right.shape[0]>0):

                    #Calculate error
                    curr_error = (curr_y_train_left.shape[0]/y_train.shape[0])*self.CalcError(curr_y_train_left) + (curr_y_train_right.shape[0]/y_train.shape[0])*self.CalcError(curr_y_train_right)
                    if curr_error < max_error:
                        best_variable_index = curr_variable_index
                        threshold = curr_threshold
                        X_train_left = curr_X_train_left
                        y_train_left = curr_y_train_left
                        X_train_right = curr_X_train_right
                        y_train_right = curr_y_train_right
                        max_error = curr_error
        return best_variable_index,threshold,X_train_left,y_train_left,X_train_right,y_train_right,max_error
    
    #SplitData Function
    def SplitData(self,best_variable_index,curr_threshold,X_train,y_train):
        X_train_left = X_train[X_train[:,best_variable_index]<=curr_threshold]
        X_train_right = X_train[X_train[:,best_variable_index]>curr_threshold]
        y_train_left = y_train[(X_train[:,best_variable_index]<=curr_threshold).reshape(-1,1)].reshape(-1,1)
        y_train_right = y_train[(X_train[:,best_variable_index]>curr_threshold).reshape(-1,1)].reshape(-1,1)
        return X_train_left,y_train_left,X_train_right,y_train_right
    
    #CalcMisclassificationError Function
    def CalcMisclassificationError(self,y_array):
        p = np.mean(y_array)
        error = min(p,1-p)
        return error
    
    #CalcEntropyLoss Function
    def CalcEntropyLoss(self,y_array):
        p = np.mean(y_array)
        
        #Account for case of undefined errors
        if (p<=0) or (p>=1):
            error = 0
        else:
            error = -1*p*np.log2(p) - (1-p)*np.log2(1-p)
        return error
    
    #CalcGiniIndex Function
    def CalcGiniIndex(self,y_array):
        p = np.mean(y_array)
        error = p*(1-p)
        return error
    
    #CalcError Function
    def CalcError(self,y_train):
        #Choose error loss calculating method according to set value
        if self.error_method_name in ['Misclassification Error','Misclassification error','misclassification error','Misclassification','misclassification']:
            error = self.CalcMisclassificationError(y_train)
        elif self.error_method_name in ['Entropy Loss','Entropy loss','entropy loss','Entropy','entropy']:
            error = self.CalcEntropyLoss(y_train)
        else:
            error = self.CalcGiniIndex(y_train)
        return error
    
    #CalculateLeafValue Function
    def CalculateLeafValue(self,y_train):
        leaf_value = np.round(np.mean(y_train))
        return leaf_value
    
    #FitData Function
    def FitData(self,X_train,y_train):
        self.root = self.BuildBinaryDecisionTree(X_train,y_train)
        return
    
    #PredictLabel Function
    def PredictLabel(self,binary_decision_tree,X_data):
        #If it is a leaf node, immediately take the value
        if binary_decision_tree.leaf_value != None: 
            y_pred = binary_decision_tree.leaf_value
            return y_pred
        
        #Get splitting variable value 
        best_variable_value = X_data[0,binary_decision_tree.best_variable_index]
        
        #Pivot left or right accordingly
        if best_variable_value<=binary_decision_tree.threshold:
            return self.PredictLabel(binary_decision_tree.left_branch,X_data)
        else:
            return self.PredictLabel(binary_decision_tree.right_branch,X_data)

    #PredictAllLabels Function
    def PredictAllLabels(self,X_test):
        y_pred = np.array([self.PredictLabel(self.root,X_test[i,:].reshape(1,-1)) for i in range(X_test.shape[0])])
        y_pred = y_pred.reshape(-1,1)
        return y_pred

#Define Bagging Class
class Bagging:
    #Define constructor method
    def __init__(self,number_of_trees):
        self.number_of_trees = number_of_trees
        self.trees = np.array([])
        return

    #FitData Function
    def FitData(self,X_train,y_train,error_method_name,max_depth):
        for i in range(self.number_of_trees):
            X_bootstrapped,y_bootstrapped = self.GetBoostrappedData(X_train,y_train)
            binary_decision_tree_model = BinaryDecisionTree(error_method_name,max_depth)
            binary_decision_tree_model.FitData(X_bootstrapped,y_bootstrapped)
            self.trees.np.append(binary_decision_tree_model)
        return

    #GetBoostrappedData Function
    def GetBoostrappedData(self,X_train,y_train):
        X_bootstrapped = random(X_train)
        y_bootstrapped = random(y_train)
        return X_bootstrapped,y_bootstrapped
    
    #PredictLabels Function
    def PredictLabels(self,X_test):
        return

#Define RandomForests Class
class RandomForests:
    def __init__(self,number_of_trees):
        return

#################################################################################################################
### Main Functions

#CreateAllBinaryDecisionTreeModels Function
def CreateAllBinaryDecisionTreeModels(X_train,y_train,X_test,y_test):
    error_method_name1 = 'Misclassification Error'
    error_method_name2 = 'Entropy Loss'
    error_method_name3 = 'Gini Loss'

    #Initialize lists to store training and testing accuracies
    max_depth_list = np.arange(11)
    training_accuracy_misclassification_error_list = np.zeros(max_depth_list.shape[0])
    testing_accuracy_misclassification_error_list = np.zeros(max_depth_list.shape[0])
    training_accuracy_entropy_loss_list = np.zeros(max_depth_list.shape[0])
    testing_accuracy_entropy_loss_list = np.zeros(max_depth_list.shape[0])
    training_accuracy_gini_loss_list = np.zeros(max_depth_list.shape[0])
    testing_accuracy_gini_loss_list = np.zeros(max_depth_list.shape[0])

    #Perform for a range of tree depth values
    for max_depth in range(11):
        #Perform for each method
        binary_decision_tree_model_misclassification_error,y_pred_test_misclassification_error,training_accuracy_misclassification_error,testing_accuracy_misclassification_error = CreateBinaryDecisionTreeModel(X_train,y_train,X_test,y_test,error_method_name1,max_depth)
        binary_decision_tree_model_entropy_loss,y_pred_test_entropy_loss,training_accuracy_entropy_loss,testing_accuracy_entropy_loss = CreateBinaryDecisionTreeModel(X_train,y_train,X_test,y_test,error_method_name2,max_depth)
        binary_decision_tree_model_gini_loss,y_pred_test_gini_loss,training_accuracy_gini_loss,testing_accuracy_gini_loss = CreateBinaryDecisionTreeModel(X_train,y_train,X_test,y_test,error_method_name3,max_depth)
        
        #Store training and testing accuracies in lists
        training_accuracy_misclassification_error_list[max_depth] = training_accuracy_misclassification_error
        testing_accuracy_misclassification_error_list[max_depth] = testing_accuracy_misclassification_error
        training_accuracy_entropy_loss_list[max_depth] = training_accuracy_entropy_loss
        testing_accuracy_entropy_loss_list[max_depth] = testing_accuracy_entropy_loss
        training_accuracy_gini_loss_list[max_depth] = training_accuracy_gini_loss
        testing_accuracy_gini_loss_list[max_depth] = testing_accuracy_gini_loss
    return max_depth_list,training_accuracy_misclassification_error_list,testing_accuracy_misclassification_error_list,training_accuracy_entropy_loss_list,testing_accuracy_entropy_loss_list,training_accuracy_gini_loss_list,testing_accuracy_gini_loss_list

#CreatePlot Function
def CreatePlot(max_depth_list,training_accuracy_list,testing_accuracy_list,error_method_name):
    plt.plot(max_depth_list,training_accuracy_list,color='green',label='Training Accuracy')
    plt.plot(max_depth_list,testing_accuracy_list,color='blue',label='Testing Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Using '+str(error_method_name))
    plt.title('Accuracy vs Max Depth Using Binary Decision Trees With '+str(error_method_name))
    plt.legend()
    plt.grid(True)
    plt.show()
    return

#################################################################################################################
### Variables

filename_X_train_D = 'Data/X_train_D.csv'
filename_y_train_D = 'Data/Y_train_D.csv'
filename_X_test_D = 'Data/X_test_D.csv'
filename_y_test_D = 'Data/Y_test_D.csv'

error_method_name1 = 'Misclassification Error'
error_method_name2 = 'Entropy Loss'
error_method_name3 = 'Gini Loss'

#################################################################################################################
### Main Code

X_train_D,y_train_D,X_test_D,y_test_D = LoadTrainingAndTestingData(filename_X_train_D,filename_y_train_D,filename_X_test_D,filename_y_test_D)
max_depth_list,training_accuracy_misclassification_error_list,testing_accuracy_misclassification_error_list,training_accuracy_entropy_loss_list,testing_accuracy_entropy_loss_list,training_accuracy_gini_loss_list,testing_accuracy_gini_loss_list = CreateAllBinaryDecisionTreeModels(X_train_D,y_train_D,X_test_D,y_test_D)

#Create training and testing accuracy plots for each loss method
CreatePlot(max_depth_list,training_accuracy_misclassification_error_list,testing_accuracy_misclassification_error_list,error_method_name1)
CreatePlot(max_depth_list,training_accuracy_entropy_loss_list,testing_accuracy_entropy_loss_list,error_method_name2)
CreatePlot(max_depth_list,training_accuracy_gini_loss_list,testing_accuracy_gini_loss_list,error_method_name3)


