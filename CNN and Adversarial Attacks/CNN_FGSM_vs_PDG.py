import A4_E3_Q1_a
import A4_E3_Q1_b
import A4_E3_Q1_c
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tabulate import tabulate

#################################################################################################################
### Other Functions

def CreateAdversarialDataUsingProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,X_train,y_train,criterion,optimizer,epsilon):
    alpha = 10e-3
    num_iterations = 10
    X_train_clone = X_train.clone().detach().requires_grad_(True)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        y_pred = CNN_adversarial_projected_gradient_descent_fitted_model(X_train_clone)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor([y_train])
        loss = criterion(y_pred,y_train)
        loss.backward()

        X_train_adversarial = X_train_clone + alpha * X_train_clone.grad.sign()
        X_train_adversarial = X_train + torch.clamp(X_train_adversarial, X_train-epsilon, X_train+epsilon)
        X_train_adversarial = torch.clamp(X_train_adversarial, 0, 1)
    return X_train_adversarial

def EvaluateModelWithAdversarialProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,epsilon):
    CNN_adversarial_projected_gradient_descent_fitted_model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_adversarial_projected_gradient_descent_fitted_model.parameters(), lr=0.001)
    for X_test, y_train in test_loader:
        X_test_adversarial = CreateAdversarialDataUsingProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,X_test,y_train,criterion,optimizer,epsilon)
        y_pred = CNN_adversarial_projected_gradient_descent_fitted_model(X_test_adversarial)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_train.size(0)
        correct += (predicted == y_train).sum().item()

    accuracy = correct/total
    return accuracy

def DisplayAdversarialProjectedGradientDescentImages(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,epsilons,num_images=5):
    for epsilon in epsilons:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(CNN_adversarial_projected_gradient_descent_fitted_model.parameters(), lr=0.001)
        accuracy_adversarial_projected_gradient_descent = EvaluateModelWithAdversarialProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,epsilon)
        
        print(f'Model accuracy on projected gradient descent adversarial set with epsilon = {epsilon}: {accuracy_adversarial_projected_gradient_descent}')

        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Sample Images for epsilon = {epsilon} using Projected Gradient Descent Method', fontsize=16)

        total_images = len(test_loader.dataset)
        random_indices = torch.randperm(total_images)[:num_images]
        
        for i, idx in enumerate(random_indices,1):
            X_test, y_test = test_loader.dataset[idx]
            if i > num_images:
                break

            X_test_adversarial = CreateAdversarialDataUsingProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,X_test.unsqueeze(0),y_test,criterion,optimizer,epsilon)
            X_test_adversarial = X_test_adversarial.squeeze(0)

            label_scalar = y_test.item() if torch.is_tensor(y_test) else y_test

            plt.subplot(2, num_images, i)
            plt.imshow(X_test[0].detach().numpy(), cmap='gray')
            plt.title(f'Original\nLabel: {label_scalar}')
            plt.axis('off')

            plt.subplot(2, num_images, i + num_images)
            plt.imshow(X_test_adversarial[0].detach().numpy(), cmap='gray')
            plt.title(f'PGD Adversarial\nPrediction: {CNN_adversarial_projected_gradient_descent_fitted_model(X_test_adversarial.unsqueeze(0)).argmax(dim=1)[0].item()}')
            plt.axis('off')
        plt.show()
    return

def CreateFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model,train_loader,CNN_adversarial_projected_gradient_descent_fitted_model_filepath,training_epsilon,num_epochs=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_adversarial_projected_gradient_descent_fitted_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):            
            X_train, y_train = data
            X_train_adversarial = CreateAdversarialDataUsingProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,X_train,y_train,criterion,optimizer,training_epsilon)
            optimizer.zero_grad()
            y_pred_adversarial = CNN_adversarial_projected_gradient_descent_fitted_model(X_train_adversarial)
            adversarial_loss = criterion(y_pred_adversarial,y_train)
            adversarial_loss.backward(retain_graph=True)
            optimizer.step()
    torch.save(CNN_adversarial_projected_gradient_descent_fitted_model.state_dict(), CNN_adversarial_projected_gradient_descent_fitted_model_filepath)
    return CNN_adversarial_projected_gradient_descent_fitted_model

def CreateFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model,train_loader,CNN_adversarial_projected_gradient_descent_fitted_model_filepath,training_epsilon,num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_adversarial_projected_gradient_descent_fitted_model.parameters(),lr=0.001)    
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader,0):
            X_train,y_train = data
            X_train_adversarial = CreateAdversarialDataUsingProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,X_train,y_train,criterion,optimizer,training_epsilon)
            optimizer.zero_grad()
            y_pred_adversarial = CNN_adversarial_projected_gradient_descent_fitted_model(X_train_adversarial)
            adversarial_loss = criterion(y_pred_adversarial,y_train)
            CNN_adversarial_projected_gradient_descent_fitted_model.zero_grad()
            adversarial_loss.backward(retain_graph=True)
            optimizer.step()
    torch.save(CNN_adversarial_projected_gradient_descent_fitted_model.state_dict(),CNN_adversarial_projected_gradient_descent_fitted_model_filepath)
    return CNN_adversarial_projected_gradient_descent_fitted_model

def LoadFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model_filepath):
    CNN_adversarial_projected_gradient_descent_fitted_model = A4_E3_Q1_a.LoadCNNModel(CNN_adversarial_projected_gradient_descent_fitted_model_filepath)
    return CNN_adversarial_projected_gradient_descent_fitted_model

def CreateOrLoadFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model,train_loader,CNN_adversarial_projected_gradient_descent_fitted_model_filepath,training_epsilon,num_epochs,create_or_load_string_fitted='load'):
    if create_or_load_string_fitted in ['Create','create']:                                                       
        CNN_adversarial_projected_gradient_descent_fitted_model = CreateFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model,train_loader,CNN_adversarial_projected_gradient_descent_fitted_model_filepath,training_epsilon,num_epochs)
    else:
        CNN_adversarial_projected_gradient_descent_fitted_model = LoadFittedAdversarialProjectedGradientDescentModel(CNN_adversarial_projected_gradient_descent_fitted_model_filepath)
    return CNN_adversarial_projected_gradient_descent_fitted_model

def CreateAndPrintAccuracyTable(CNN_fitted_model,CNN_adversarial_fast_gradient_sign_fitted_model,CNN_adversarial_projected_gradient_descent_fitted_model,testing_epsilons):
    accuracy_standard = A4_E3_Q1_b.EvaluateModel(CNN_fitted_model,test_loader)
    fast_gradient_sign_accuracy_fitted_model_standard_filepath = A4_E3_Q1_b.EvaluateModel(CNN_adversarial_fast_gradient_sign_fitted_model,test_loader)
    projected_gradient_descent_fitted_model_accuracy_standard_filepath = A4_E3_Q1_b.EvaluateModel(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader)
    for epsilon in testing_epsilons:        
        accuracy_adversarial_fast_gradient_sign = A4_E3_Q1_b.EvaluateModelWithAdversarialFastGradientSignMethod(CNN_fitted_model,test_loader,epsilon)
        fast_gradient_sign_accuracy_fitted_model_adversarial_fast_gradient_sign = A4_E3_Q1_b.EvaluateModelWithAdversarialFastGradientSignMethod(CNN_adversarial_fast_gradient_sign_fitted_model,test_loader,epsilon)
        projected_gradient_descent_fitted_model_accuracy_adversarial_fast_gradient_sign = A4_E3_Q1_b.EvaluateModelWithAdversarialFastGradientSignMethod(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,epsilon)

        
        accuracy_adversarial_projected_gradient_descent = EvaluateModelWithAdversarialProjectedGradientDescentMethod(CNN_fitted_model,test_loader,epsilon)
        fast_gradient_sign_accuracy_fitted_model_adversarial__projected_gradient_descent = EvaluateModelWithAdversarialProjectedGradientDescentMethod(CNN_adversarial_fast_gradient_sign_fitted_model,test_loader,epsilon)
        projected_gradient_descent_fitted_model_accuracy_adversarial_projected_gradient_descent = EvaluateModelWithAdversarialProjectedGradientDescentMethod(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,epsilon)

        print("************************************************************")
        print(f'################ Epsilon = {epsilon} ################')
        data = [
            ["", "Standard Training", "FGSM Adversarial Training", "PGD Adversarial Training"],
            ["Standard Test Accuracy", accuracy_standard, fast_gradient_sign_accuracy_fitted_model_standard_filepath, projected_gradient_descent_fitted_model_accuracy_standard_filepath],
            ["Robust Test Accuracy (FGSM)", accuracy_adversarial_fast_gradient_sign, fast_gradient_sign_accuracy_fitted_model_adversarial_fast_gradient_sign, projected_gradient_descent_fitted_model_accuracy_adversarial_fast_gradient_sign],
            ["Robust Test Accuracy (PGD)", accuracy_adversarial_projected_gradient_descent, fast_gradient_sign_accuracy_fitted_model_adversarial__projected_gradient_descent, projected_gradient_descent_fitted_model_accuracy_adversarial_projected_gradient_descent],
        ]

        # Print the table
        print(tabulate(data,headers="firstrow",tablefmt="grid"))
    return

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
    CNN_adversarial_projected_gradient_descent_fitted_model_filepath = 'Models/A4_E3_Q1/CNN_adversarial_projected_gradient_descent_fitted_model.pth'
    
    num_images = 5
    training_epsilon = 0.2
    testing_epsilons = [0.2,0.1,0.5]
    
    create_or_load_string_data = 'Load'
    create_or_load_string_intial = 'Load'
    create_or_load_string_fitted = 'Load'
    create_or_load_string_adversarial_fitted_fast_gradient_sign = 'Load'
    create_or_load_string_adversarial_fitted_projected_gradient_descent = 'Load'
    
    PrintIntro()
    train_loader,test_loader = A4_E3_Q1_a.CreateOrLoadData(train_filepath,test_filepath,create_or_load_string_data)
    CNN_initial_model = A4_E3_Q1_a.CreateOrLoadInitialCNNModel(CNN_initial_model_filepath,create_or_load_string_intial)
    CNN_adversarial_projected_gradient_descent_fitted_model = CreateOrLoadFittedAdversarialProjectedGradientDescentModel(CNN_initial_model,train_loader,CNN_adversarial_projected_gradient_descent_fitted_model_filepath,training_epsilon,num_epochs,create_or_load_string_adversarial_fitted_projected_gradient_descent)

    DisplayAdversarialProjectedGradientDescentImages(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,testing_epsilons,num_images)
    A4_E3_Q1_b.DisplayAdversarialFastGradientSignImages(CNN_adversarial_projected_gradient_descent_fitted_model,test_loader,testing_epsilons,num_images)
    
    CNN_fitted_model,_,_,_,_ = A4_E3_Q1_a.CreateOrLoadFittedModel(CNN_initial_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs,create_or_load_string_fitted)
    CNN_adversarial_fast_gradient_sign_fitted_model = A4_E3_Q1_c.CreateOrLoadFittedAdversarialFastGradientSignModel(CNN_initial_model,train_loader,CNN_adversarial_fast_gradient_sign_fitted_model_filepath,training_epsilon,num_epochs,create_or_load_string_adversarial_fitted_fast_gradient_sign)

