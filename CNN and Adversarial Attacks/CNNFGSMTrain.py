import A4_E3_Q1_a
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#################################################################################################################
### Other Functions

def CreateAdversarialDataUsingFastGradientSignMethod(CNN_fitted_model,X_test,y_test,epsilon):
    X_test.requires_grad = True
    y_pred = CNN_fitted_model(X_test)
    if isinstance(y_test,int):
        y_test = torch.tensor([y_test])
    loss = nn.CrossEntropyLoss()(y_pred,y_test)
    loss.backward()
    X_test_adversarial = X_test + epsilon*X_test.grad.sign()
    X_test_adversarial = torch.clamp(X_test_adversarial,0,1)
    return X_test_adversarial

def EvaluateModel(CNN_fitted_model,test_loader):
    CNN_fitted_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
            outputs = CNN_fitted_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    accuracy = correct/total
    return accuracy

def EvaluateModelWithAdversarialFastGradientSignMethod(CNN_fitted_model,test_loader,epsilon):
    correct = 0
    total = 0
    for X_test,y_test in test_loader:
        X_test_adversarial = CreateAdversarialDataUsingFastGradientSignMethod(CNN_fitted_model, X_test, y_test, epsilon)
        y_pred = CNN_fitted_model(X_test_adversarial)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()
    accuracy_adversarial = correct/total
    return accuracy_adversarial

def DisplayAdversarialFastGradientSignImages(CNN_fitted_model,test_loader,epsilons,num_images=5):
    for epsilon in epsilons:
        accuracy_adversarial_fast_gradient_sign = EvaluateModelWithAdversarialFastGradientSignMethod(CNN_fitted_model,test_loader,epsilon)
                
        print(f'Model accuracy on on fast gradient sign adversarial set with epsilon = {epsilon}: {accuracy_adversarial_fast_gradient_sign}')

        plt.figure(figsize=(15,5))
        plt.suptitle(f'Sample Images for epsilon = {epsilon} using Fast Gradient Sign Method', fontsize=16)
        
        total_images = len(test_loader.dataset)
        random_indices = torch.randperm(total_images)[:num_images]
        for i, idx in enumerate(random_indices, 1):
            X_test, y_test = test_loader.dataset[idx]
            if i > num_images:
                break

            X_test_adversarial = CreateAdversarialDataUsingFastGradientSignMethod(CNN_fitted_model, X_test, y_test, epsilon)

            label_scalar = y_test[0].item() if torch.is_tensor(y_test) else y_test

            plt.subplot(2, num_images, i)
            plt.imshow(X_test[0].detach().numpy(), cmap='gray')
            plt.title(f'Original\nLabel: {label_scalar}')
            plt.axis('off')

            plt.subplot(2, num_images, i + num_images)
            plt.imshow(X_test_adversarial[0].detach().numpy(), cmap='gray')
            plt.title(f'Perturbed\nPrediction: {CNN_fitted_model(X_test_adversarial).argmax(dim=1)[0].item()}')
            plt.axis('off')
        plt.show()
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
    num_images = 5
    epsilons = [0.2,0.1,0.5]
    
    create_or_load_string_data = 'Load'
    create_or_load_string_intial = 'Load'
    create_or_load_string_fitted = 'Load'
    
    PrintIntro()
    train_loader,test_loader = A4_E3_Q1_a.CreateOrLoadData(train_filepath,test_filepath,create_or_load_string_data)
    CNN_initial_model = A4_E3_Q1_a.CreateOrLoadInitialCNNModel(CNN_initial_model_filepath,create_or_load_string_intial)
    CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list = A4_E3_Q1_a.CreateOrLoadFittedModel(CNN_initial_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs,create_or_load_string_fitted)
    DisplayAdversarialFastGradientSignImages(CNN_fitted_model,test_loader,epsilons,num_images)