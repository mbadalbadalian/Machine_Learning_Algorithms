import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#################################################################################################################
### Other Functions

#Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolutional_layer_1 = nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.convolutional_layer_2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fully_connected_layer_1 = nn.Linear(3136,128)
        self.fully_connected_layer_2 = nn.Linear(128,10)

    #Forward Pass
    def forward(self, x):
        x = self.pool(torch.relu(self.convolutional_layer_1(x)))
        x = self.pool(torch.relu(self.convolutional_layer_2(x)))
        x = x.view(-1,3136)
        x = torch.relu(self.fully_connected_layer_1(x))
        x = self.fully_connected_layer_2(x)
        return x
    
def PrepareTrainingAndTestingData(train_filepath,test_filepath):
    torch.manual_seed(100)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    torch.save(train_loader, train_filepath)
    torch.save(test_loader, test_filepath)
    return train_loader, test_loader

#LoadTrainingAndTestingData Function
def LoadTrainingAndTestingData(train_filepath,test_filepath):
    #Load data
    train_loader = torch.load(train_filepath)
    test_loader = torch.load(test_filepath)
    return train_loader,test_loader

#CreateOrLoadData Function
def CreateOrLoadData(train_filepath,test_filepath,create_or_load_string_data='load'):
    if create_or_load_string_data in ['Create','create']:
        train_loader,test_loader = PrepareTrainingAndTestingData(train_filepath,test_filepath)
    else:
        train_loader,test_loader = LoadTrainingAndTestingData(train_filepath,test_filepath)
    return train_loader,test_loader

def CreateInitialCNNModel(CNN_initial_model_filepath):
    torch.manual_seed(100)
    CNN_initial_model = CNN()
    torch.save(CNN_initial_model.state_dict(), CNN_initial_model_filepath)
    return CNN_initial_model

def LoadCNNModel(CNN_model_filepath):
    CNN_model = CNN()
    CNN_model.load_state_dict(torch.load(CNN_model_filepath))
    CNN_model.eval()
    return CNN_model

#CNN Function
def CreateOrLoadInitialCNNModel(CNN_initial_model_filepath,create_or_load_string_intial='load'):
    if create_or_load_string_intial in ['Create','create']:
        #Create the CNN model
        CNN_initial_model = CreateInitialCNNModel(CNN_initial_model_filepath)
    else:
        #Load the CNN model
        CNN_initial_model = LoadCNNModel(CNN_initial_model_filepath)
    return CNN_initial_model

def CreateFittedCNNModel(CNN_fitted_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_fitted_model.parameters(),lr=0.001)
    
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = CNN_fitted_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        training_loss = running_loss / len(train_loader)
        training_accuracy = correct_train / total_train
        train_loss_list.append(training_loss)
        train_accuracy_list.append(training_accuracy)

        print(f'Epoch {epoch + 1}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy}')

        # Testing
        correct_test = 0
        total_test = 0
        testing_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = CNN_fitted_model(inputs)
                loss = criterion(outputs, labels)
                testing_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        testing_loss /= len(test_loader)
        testing_accuracy = correct_test / total_test
        test_loss_list.append(testing_loss)
        test_accuracy_list.append(testing_accuracy)

        print(f'Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy}')

    torch.save(train_loss_list, train_loss_filepath)
    torch.save(test_loss_list, test_loss_filepath)
    torch.save(train_accuracy_list, train_accuracy_filepath)
    torch.save(test_accuracy_list, test_accuracy_filepath)
    torch.save(CNN_fitted_model.state_dict(), CNN_fitted_model_filepath)
    return CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list

def LoadFittedCNNModel(CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath):
    CNN_fitted_model = LoadCNNModel(CNN_fitted_model_filepath)
    train_loss_list = torch.load(train_loss_filepath)
    test_loss_list = torch.load(test_loss_filepath)
    train_accuracy_list = torch.load(train_accuracy_filepath)
    test_accuracy_list = torch.load(test_accuracy_filepath)
    return CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list

def CreateOrLoadFittedModel(CNN_fitted_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs,create_or_load_string_fitted='load'):
    if create_or_load_string_fitted in ['Create','create']:                                                           
        CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list = CreateFittedCNNModel(CNN_fitted_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs)
    else:
        CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list = LoadFittedCNNModel(CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath)
    return CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list

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
    
    create_or_load_string_data = 'Create'
    create_or_load_string_intial = 'Create'
    create_or_load_string_fitted = 'Create'
    
    PrintIntro()
    train_loader,test_loader = CreateOrLoadData(train_filepath,test_filepath,create_or_load_string_data)
    CNN_initial_model = CreateOrLoadInitialCNNModel(CNN_initial_model_filepath,create_or_load_string_intial)
    CNN_fitted_model,train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list = CreateOrLoadFittedModel(CNN_initial_model,train_loader,test_loader,CNN_fitted_model_filepath,train_loss_filepath,test_loss_filepath,train_accuracy_filepath,test_accuracy_filepath,num_epochs,create_or_load_string_fitted)
