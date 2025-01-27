import numpy as np

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

#CalcGradWAndGradB Function
def CalcGradWAndGradB(C,x_i,y_i,curr_w,curr_b,epsilon):
    diff_y = (y_i - (np.dot(x_i,curr_w) + curr_b))[0,0]
    #Seperate gradient into three different cases
    if diff_y > epsilon:
        grad_w = -1*x_i.T
        grad_b = -1
    elif (diff_y <= epsilon) and (diff_y >= -1*epsilon):
        grad_w = np.zeros(curr_w.shape)
        grad_b = 0
    else:
        grad_w = x_i.T
        grad_b = 1
    return grad_w,grad_b

#GradientDescentSVR Function
def GradientDescentSVR(X_train,y_train,w_vector,b,max_pass,step_size):
    #Initialize values
    C = 1
    epsilon = 0.5
    curr_w = w_vector
    curr_b = b
    curr_step_size = step_size
    for t in range(int(max_pass)):
        curr_step_size = curr_step_size/5
        for i in range(y_train.shape[0]):
            x_i = X_train[i,:].reshape(1,-1)
            y_i = y_train[i,:].reshape(-1,1)
            grad_w,grad_b = CalcGradWAndGradB(C,x_i,y_i,curr_w,curr_b,epsilon)
            curr_w = curr_w - curr_step_size*grad_w
            curr_b = curr_b - curr_step_size*grad_b
            curr_w = curr_w/(curr_step_size + 1)
    return curr_w,curr_b

#CalcAllErrorsAndLoss Function
def CalcAllErrorsAndLoss(X_train,y_train,X_test,y_test,w_vector,b):
    C = 1
    epsilon = 0.5
    training_error = C*np.sum(np.maximum(np.abs(y_train-np.dot(X_train,w_vector)+b)-epsilon,0))
    test_error = C*np.sum(np.maximum(np.abs(y_test-np.dot(X_test,w_vector)+b)-epsilon,0))
    training_loss =  0.5*np.dot(w_vector.T,w_vector)[0,0] + training_error
    return training_error,training_loss,test_error

#PrintOutputs Function
def PrintOutputs(training_error,training_loss,test_error,max_pass,step_size):
    print("************************************************************")
    print("################ Gradient Descent With Support Vector Machines ################")
    print("The following is using max_pass="+str(max_pass)+", step_size="+str(step_size))
    print()
    print("Training Error")
    print(training_error)
    print("Training Loss:")
    print(training_loss)
    print("Test Error:")
    print(test_error)
    print("************************************************************")
    return

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

filename_X_train_C = 'Data/X_train_C.csv'
filename_y_train_C = 'Data/Y_train_C.csv'
filename_X_test_C = 'Data/X_test_C.csv'
filename_y_test_C = 'Data/Y_test_C.csv'

max_pass = 1e3
step_size = 0.1
 
#################################################################################################################
### Main Code

X_train_C,y_train_C,X_test_C,y_test_C = LoadTrainingAndTestingData(filename_X_train_C,filename_y_train_C,filename_X_test_C,filename_y_test_C)
initial_w_vector = np.zeros([X_train_C.shape[1],1])
initial_b = 0
w_vector,b = GradientDescentSVR(X_train_C,y_train_C,initial_w_vector,initial_b,max_pass,step_size)
training_error,training_loss,test_error = CalcAllErrorsAndLoss(X_train_C,y_train_C,X_test_C,y_test_C,w_vector,b)
PrintOutputs(training_error,training_loss,test_error,max_pass,step_size)



