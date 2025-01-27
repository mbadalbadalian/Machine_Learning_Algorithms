import A4_E3_Q1_a
import A4_E3_Q1_b
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#################################################################################################################
### Other Functions

def CreateFittedAdversarialFastGradientSignModel(CNN_adversarial_fast_gradient_sign_fitted_model,train_loader,CNN_adversarial_fast_gradient_sign_fitted_model_filepath,training_epsilon,num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_adversarial_fast_gradient_sign_fitted_model.parameters(),lr=0.001)

    for epoch in range(num_epochs):
        for i,data in enumerate(train_loader,0):
            X_train,y_train = data
            X_train.requires_grad = True  
            optimizer.zero_grad()

            y_pred = CNN_adversarial_fast_gradient_sign_fitted_model(X_train)
            loss = criterion(y_pred,y_train)
            loss.backward()

            X_train_adversarial = X_train + training_epsilon*X_train.grad.sign()
            X_train_adversarial = torch.clamp(X_train_adversarial,0,1)

            optimizer.zero_grad()
            y_pred_adversarial = CNN_adversarial_fast_gradient_sign_fitted_model(X_train_adversarial)
            loss_adversarial = criterion(y_pred_adversarial,y_train)
            loss_adversarial.backward()
            optimizer.step()
            
    torch.save(CNN_adversarial_fast_gradient_sign_fitted_model.state_dict(),CNN_adversarial_fast_gradient_sign_fitted_model_filepath)
    return CNN_adversarial_fast_gradient_sign_fitted_model

def LoadFittedAdversarialFastGradientSignModel(CNN_fitted_model_filepath):
    CNN_fitted_model = A4_E3_Q1_a.LoadCNNModel(CNN_fitted_model_filepath)
    return CNN_fitted_model

def CreateOrLoadFittedAdversarialFastGradientSignModel(CNN_adversarial_fast_gradient_sign_fitted_model,train_loader,CNN_adversarial_fast_gradient_sign_fitted_model_filepath,training_epsilon,num_epochs,create_or_load_string_fitted='load'):
    if create_or_load_string_fitted in ['Create','create']:                                                           
        CNN_adversarial_fast_gradient_sign_fitted_model = CreateFittedAdversarialFastGradientSignModel(CNN_adversarial_fast_gradient_sign_fitted_model,train_loader,CNN_adversarial_fast_gradient_sign_fitted_model_filepath,training_epsilon,num_epochs)
    else:
        CNN_adversarial_fast_gradient_sign_fitted_model = LoadFittedAdversarialFastGradientSignModel(CNN_adversarial_fast_gradient_sign_fitted_model_filepath)
    return CNN_adversarial_fast_gradient_sign_fitted_model

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":    
    #Variables
    num_epochs = 5
    train_filepath = 'Created_Data/A4_E3_Q1/train_MNIST.pth'
    test_filepath = 'Created_Data/A4_E3_Q1/test_MNIST.pth'
    CNN_initial_model_filepath = 'Models/A4_E3_Q1/CNN_initial_model.pth'
    CNN_fitted_model_filepath = 'Models/A4_E3_Q1/CNN_fitted_model_'+str(num_epochs)+'epochs.pth'
    train_accuracy_filepath = 'Created_Data/A4_E3_Q1/train_accuracy_'+str(num_epochs)+'epochs.pth'
    test_accuracy_filepath = 'Created_Data/A4_E3_Q1/test_accuracy_'+str(num_epochs)+'epochs.pth'
    train_loss_filepath = 'Created_Data/A4_E3_Q1/train_loss_'+str(num_epochs)+'epochs.pth'
    test_loss_filepath = 'Created_Data/A4_E3_Q1/test_loss_'+str(num_epochs)+'epochs.pth'
    CNN_adversarial_fast_gradient_sign_fitted_model_filepath = 'Models/A4_E3_Q1/CNN_adversarial_fast_gradient_sign_fitted_model.pth'
    fast_gradient_sign_accuracy_adversarial_fast_gradient_sign_list_filepath = 'Created_Data/A4_E3_Q1/CNN_adversarial_fast_gradient_sign_fitted_model_'+str(num_epochs)+'epochs_adversarial_fast_gradient_sign_accuracy.pth'
    num_images = 5
    training_epsilon = 0.2
    testing_epsilons = [0.2,0.1,0.5]
    
    create_or_load_string_data = 'Load'
    create_or_load_string_intial = 'Load'
    create_or_load_string_adversarial_fitted = 'Create'
    
    PrintIntro()
    train_loader,test_loader = A4_E3_Q1_a.CreateOrLoadData(train_filepath,test_filepath,create_or_load_string_data)
    CNN_initial_model = A4_E3_Q1_a.CreateOrLoadInitialCNNModel(CNN_initial_model_filepath,create_or_load_string_intial)
    
    CNN_adversarial_fast_gradient_sign_fitted_model = CreateOrLoadFittedAdversarialFastGradientSignModel(CNN_initial_model,train_loader,CNN_adversarial_fast_gradient_sign_fitted_model_filepath,training_epsilon,num_epochs,create_or_load_string_adversarial_fitted)
    fast_gradient_sign_fitted_model_accuracy_adversarial_fast_gradient_sign_list = A4_E3_Q1_b.DisplayAdversarialFastGradientSignImages(CNN_adversarial_fast_gradient_sign_fitted_model,test_loader,testing_epsilons,fast_gradient_sign_accuracy_adversarial_fast_gradient_sign_list_filepath,num_images)