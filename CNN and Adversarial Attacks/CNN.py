import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#################################################################################################################
### Other Functions

#LoadTrainingAndTestData Function
def PrepareTrainingAndTestingData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath):
    #Extract training and testing data
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    
    #Resize images to 32 x 32
    X_train = tf.image.resize(X_train[...,tf.newaxis],[32,32]).numpy()
    X_test = tf.image.resize(X_test[...,tf.newaxis],[32,32]).numpy()

    #Normalize input data
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/255.0

    #Reshape data to pass into CNN
    X_train = X_train.reshape((-1, 32, 32, 1))
    X_test = X_test.reshape((-1, 32, 32, 1))
    y_train = y_train.astype('int32').reshape(-1,1)
    y_test = y_test.astype('int32').reshape(-1,1)

    #Save variables for future reference
    np.save(X_train_filepath,X_train)
    np.save(y_train_filepath,y_train)
    np.save(X_test_filepath,X_test)
    np.save(y_test_filepath,y_test)
    return X_train,y_train,X_test,y_test

#LoadTrainingAndTestingData Function
def LoadTrainingAndTestingData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath):
    #Load data
    X_train = np.load(X_train_filepath)
    y_train = np.load(y_train_filepath)
    X_test = np.load(X_test_filepath)
    y_test = np.load(y_test_filepath)
    return X_train,y_train,X_test,y_test

#CreateOrLoadData Function
def CreateOrLoadData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath,create_or_load_string_data='load'):
    if create_or_load_string_data in ['Create','create']:
        #If we want to create and prepare the data from scratch
        X_train,y_train,X_test,y_test = PrepareTrainingAndTestingData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath)
    else:
        #If we want to load the prepared data
        X_train,y_train,X_test,y_test = LoadTrainingAndTestingData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath)
    return X_train,y_train,X_test,y_test

#CreateInitialVGG11Model Function
def CreateInitialVGG11Model(VGG11_initial_model_filepath):
    #Create a sequential model
    VGG11_initial_model = tf.keras.models.Sequential()

    #Convolutional block 1
    VGG11_initial_model.add(tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),padding='same',input_shape=(32,32,1)))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))
    VGG11_initial_model.add(tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)))

    #Convolutional block 2
    VGG11_initial_model.add(tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))
    VGG11_initial_model.add(tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)))

    #Convolutional block 3
    VGG11_initial_model.add(tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))

    #Convolutional block 4
    VGG11_initial_model.add(tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))
    VGG11_initial_model.add(tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)))

    #Convolutional block 5
    VGG11_initial_model.add(tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))

    #Convolutional block 6
    VGG11_initial_model.add(tf.keras.layers.Conv2D(512,(3,3),strides=(1,1), padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))
    VGG11_initial_model.add(tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)))

    #Convolutional block 7
    VGG11_initial_model.add(tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))

    #Convolutional block 8
    VGG11_initial_model.add(tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same'))
    VGG11_initial_model.add(tf.keras.layers.BatchNormalization())
    VGG11_initial_model.add(tf.keras.layers.Activation('relu'))
    VGG11_initial_model.add(tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)))

    #Flatten
    VGG11_initial_model.add(tf.keras.layers.Flatten())
    
    #Fully connected layers
    VGG11_initial_model.add(tf.keras.layers.Dense(4096,activation='relu'))
    VGG11_initial_model.add(tf.keras.layers.Dropout(0.5))
    VGG11_initial_model.add(tf.keras.layers.Dense(4096,activation='relu'))
    VGG11_initial_model.add(tf.keras.layers.Dropout(0.5))
    VGG11_initial_model.add(tf.keras.layers.Dense(10))
    
    #Compute softmax cross entropy loss
    VGG11_initial_model.compile(loss=tf.compat.v1.losses.sparse_softmax_cross_entropy,metrics=['accuracy'])
    
    #Save model
    VGG11_initial_model.save(VGG11_initial_model_filepath)
    return VGG11_initial_model

#CreateOrLoadInitialModel Function
def CreateOrLoadInitialModel(VGG11_initial_model_filepath,create_or_load_string_intial='load'):
    if create_or_load_string_intial in ['Create','create']:
        #Create the VGG11 model
        VGG11_initial_model = CreateInitialVGG11Model(VGG11_initial_model_filepath)
    else:
        #Load the VGG11 model
        VGG11_initial_model = tf.keras.models.load_model(VGG11_initial_model_filepath)
    return VGG11_initial_model

#CreateFittedVGG11Model Function
def CreateFittedVGG11Model(X_train,y_train,X_test,y_test,VGG11_fitted_model,VGG11_fitted_model_filepath,train_accuracy_filepath,test_accuracy_filepath,train_loss_filepath,test_loss_filepath,num_epochs=10):
    #Initialize accuracies and losses lists
    train_accuracy_list = np.zeros([num_epochs])
    test_accuracy_list = np.zeros([num_epochs])
    train_loss_list = np.zeros([num_epochs])
    test_loss_list = np.zeros([num_epochs])
    
    #Perform several epochs of training
    for epoch in range(num_epochs):
        VGG11_fitted_model.fit(X_train,y_train,epochs=1,validation_data=(X_test,y_test))
        
        #Extract accuracy and loss values
        train_accuracy_list[epoch] = VGG11_fitted_model.history.history['accuracy'][0]
        test_accuracy_list[epoch] = VGG11_fitted_model.history.history['val_accuracy'][0]
        train_loss_list[epoch] = VGG11_fitted_model.history.history['loss'][0]
        test_loss_list[epoch] = VGG11_fitted_model.history.history['val_loss'][0]
    
    #Save the fitted VGG11 model
    VGG11_fitted_model.save(VGG11_fitted_model_filepath)

    #Save accuracies and losses
    np.save(train_accuracy_filepath,train_accuracy_list)
    np.save(test_accuracy_filepath,test_accuracy_list)
    np.save(train_loss_filepath,train_loss_list)
    np.save(test_loss_filepath,test_loss_list)
    return VGG11_fitted_model,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list

#CreateOrLoadFittedModel Function
def CreateOrLoadFittedModel(X_train,y_train,X_test,y_test,VGG11_fitted_model,VGG11_fitted_model_filepath,train_accuracy_filepath,test_accuracy_filepath,train_loss_filepath,test_loss_filepath,create_or_load_string_fitted='load'):
    if create_or_load_string_fitted in ['Create','create']:
        #Create VGG11 fitted model
        VGG11_fitted_model,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list = CreateFittedVGG11Model(X_train,y_train,X_test,y_test,VGG11_fitted_model,train_accuracy_filepath,test_accuracy_filepath,train_loss_filepath,test_loss_filepath,VGG11_fitted_model_filepath)
    else:
        #Load VGG11 fitted model
        VGG11_fitted_model= tf.keras.models.load_model(VGG11_initial_model_filepath)
        
        #Load accuracies and losses
        train_accuracy_list = np.load(train_accuracy_filepath)
        test_accuracy_list = np.load(test_accuracy_filepath)
        train_loss_list = np.load(train_loss_filepath)
        test_loss_list = np.load(test_loss_filepath)
    return VGG11_fitted_model,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list

#CreatePlots Function
def CreatePlots(num_epochs,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list):
    #Create list of epoch values used
    epochs_list = np.arange(num_epochs)   
    
    #Plot training accuracy vs number of epochs
    plt.figure()
    plt.plot(epochs_list,train_accuracy_list)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Number of Epochs (Original MNIST Data)')
    plt.grid(True)
    plt.show()
    
    #Plot testing accuracy vs number of epochs
    plt.figure()
    plt.plot(epochs_list,test_accuracy_list)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Number of Epochs (Original MNIST Data)')
    plt.grid(True)
    plt.show()
    
    #Plot training loss vs number of epochs
    plt.figure()
    plt.plot(epochs_list,train_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Number of Epochs (Original MNIST Data)')
    plt.grid(True)
    plt.show()
    
    #Plot training loss vs number of epochs
    plt.figure()
    plt.plot(epochs_list,test_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Number of Epochs (Original MNIST Data)')
    plt.grid(True)
    plt.show()
    return

#ComputeFlippedImagesPerformance Function
def ComputeFlippedImagesPerformance(VGG11_fitted_model,X_test,y_test):
    #Choose horizontal flippling and vertical flipping augmentations
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
    
    #Flip testing images horizontally
    X_test_horizontally_flipped = np.array([image_data_generator.apply_transform(image_data,{'flip_horizontal': True}) for image_data in X_test])
    
    #Flip testing images vertically
    X_test_vertically_flipped = np.array([image_data_generator.apply_transform(image_data,{'flip_vertical': True}) for image_data in X_test])
    
    #Compute accuracies
    original_accuracy = VGG11_fitted_model.evaluate(X_test, y_test)[1]
    X_test_horizontally_flipped_accuracy = VGG11_fitted_model.evaluate(X_test_horizontally_flipped, y_test)[1]
    X_test_vertically_flipped_accuracy = VGG11_fitted_model.evaluate(X_test_vertically_flipped, y_test)[1]
    return original_accuracy,X_test_horizontally_flipped_accuracy,X_test_vertically_flipped_accuracy

#ComputeGaussianImagesPerformance Function
def ComputeGaussianImagesPerformance(VGG11_fitted_model,gaussian_noise_variance_list,X_test,y_test):
    #Create list of gaussian noice variance values
    gaussian_noise_variance_accuracy_list = np.zeros(gaussian_noise_variance_list.shape)
    
    #Test out each variance value
    for i in range(len(gaussian_noise_variance_list)):
        #Add noise to testing images
        X_test_noisy = np.clip(X_test + np.random.normal(loc=0,scale=np.sqrt(gaussian_noise_variance_list[i]),size=X_test.shape),0,1)
        
        #Compute accuracy for each variance value used
        gaussian_noise_variance_accuracy_list[i] = VGG11_fitted_model.evaluate(X_test_noisy,y_test)[1]
    return gaussian_noise_variance_accuracy_list

#CreateAugmentedTrainingData Function
def CreateAugmentedTrainingData(X_train,y_train,X_train_augmented_filepath,y_train_augmented_filepath):
    #Create augmentations which becomes randomly applied to batches of data
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2)
    training_data_augmented = image_data_generator.flow(X_train,y_train)
    
    #Initialize augmented training data lists
    X_train_augmented = []
    y_train_augmented = []
    
    #Randomly apply augmentations to training data
    for i in range(len(training_data_augmented)):
        X_train_augmented_batch,y_train_augmented_batch = training_data_augmented.next()
        X_train_augmented.append(X_train_augmented_batch)
        y_train_augmented.append(y_train_augmented_batch)
    X_train_augmented = np.concatenate(X_train_augmented,axis=0)
    y_train_augmented = np.concatenate(y_train_augmented,axis=0)
    
    #Save augmented training data
    np.save(X_train_augmented_filepath,X_train_augmented)
    np.save(y_train_augmented_filepath,y_train_augmented)
    return X_train_augmented,y_train_augmented

#CreateOrLoadAugmentedTrainingData Function
def CreateOrLoadAugmentedTrainingData(X_train,y_train,X_train_augmented_filepath,y_train_augmented_filepath,create_or_load_string_augmented_training_data='load'):
    if create_or_load_string_augmented_training_data in ['Create','create']:
        #Create augmented data
        X_train_augmented,y_train_augmented = CreateAugmentedTrainingData(X_train,y_train,X_train_augmented_filepath,y_train_augmented_filepath)
    else:
        #Load augmented data
        X_train_augmented = np.load(X_train_augmented_filepath)
        y_train_augmented = np.load(y_train_augmented_filepath)
    return X_train_augmented,y_train_augmented

#PrintOutputs Function
def PrintOutputs(original_accuracy,X_test_horizontally_flipped_accuracy,X_test_vertically_flipped_accuracy,gaussian_noise_variance_list,gaussian_noise_variance_accuracy_list,original_accuracy_augmented,X_test_horizontally_flipped_accuracy_augmented,X_test_vertically_flipped_accuracy_augmented,gaussian_noise_variance_accuracy_list_augmented):
    print("************************************************************")
    print("################ Part c) i) ################")
    print('Original accuracy:',original_accuracy)
    print('Horizontally flipped accuracy:',X_test_horizontally_flipped_accuracy)
    print('Vertically flipped accuracy:',X_test_vertically_flipped_accuracy)
    print("************************************************************")
    print("################ Part c) ii) ################")
    for i in range(len(gaussian_noise_variance_list)):
        print('Accuracy for variance '+str(gaussian_noise_variance_list[i])+': '+str(gaussian_noise_variance_accuracy_list[i]))
    print("************************************************************")
    print("################ Part d) ################")
    print('Original accuracy using augmented training data:',original_accuracy_augmented)
    print('Horizontally flipped accuracy using augmented training data:',X_test_horizontally_flipped_accuracy_augmented)
    print('Vertically flipped accuracy using augmented training data:',X_test_vertically_flipped_accuracy_augmented)
    for i in range(len(gaussian_noise_variance_list)):
        print('Accuracy for variance '+str(gaussian_noise_variance_list[i])+' using augmented training data: '+str(gaussian_noise_variance_accuracy_list_augmented[i]))
    print("************************************************************")
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    num_epochs = 5
    X_train_filepath = 'Created_Data/A3_E2_Q1/X_train_MNIST.npy'
    y_train_filepath = 'Created_Data/A3_E2_Q1/y_train_MNIST.npy'
    X_test_filepath = 'Created_Data/A3_E2_Q1/X_test_MNIST.npy'
    y_test_filepath = 'Created_Data/A3_E2_Q1/y_test_MNIST.npy'
    VGG11_initial_model_filepath = 'Models/A3_E2_Q1/VGG11_initial_model.h5'
    VGG11_fitted_model_filepath = 'Models/A3_E2_Q1/VGG11_fitted_model_'+str(num_epochs)+'epochs.h5'
    train_accuracy_filepath = 'Created_Data/A3_E2_Q1/train_accuracy_'+str(num_epochs)+'epochs.npy'
    test_accuracy_filepath = 'Created_Data/A3_E2_Q1/test_accuracy_'+str(num_epochs)+'epochs.npy'
    train_loss_filepath = 'Created_Data/A3_E2_Q1/train_loss_'+str(num_epochs)+'epochs.npy'
    test_loss_filepath = 'Created_Data/A3_E2_Q1/test_loss_'+str(num_epochs)+'epochs.npy'
    gaussian_noise_variance_list = np.array([0.01,0.1,1])
    X_train_augmented_filepath = 'Created_Data/A3_E2_Q1/X_train_augmented_MNIST.npy'
    y_train_augmented_filepath = 'Created_Data/A3_E2_Q1/y_train_augmented_MNIST.npy'
    train_accuracy_augmented_filepath = 'Created_Data/A3_E2_Q1/train_accuracy_augmented_'+str(num_epochs)+'epochs.npy'
    test_accuracy_augmented_filepath = 'Created_Data/A3_E2_Q1/test_accuracy_augmented_'+str(num_epochs)+'epochs.npy'
    train_loss_augmented_filepath = 'Created_Data/A3_E2_Q1/train_loss_augmented_'+str(num_epochs)+'epochs.npy'
    test_loss_augmented_filepath = 'Created_Data/A3_E2_Q1/test_loss_augmented_'+str(num_epochs)+'epochs.npy'
    VGG11_fitted_model_augmented_filepath = 'Models/A3_E2_Q1/VGG11_fitted_model_augmented_'+str(num_epochs)+'epochs.h5'
    
    #Create or load
    create_or_load_string_data = 'Create'
    create_or_load_string_intial = 'Create'
    create_or_load_string_fitted = 'Create'
    create_or_load_string_augmented_training_data = 'Create'
    create_or_load_string_fitted_augmented = 'Create'
    
    #Main Code
    X_train,y_train,X_test,y_test = CreateOrLoadData(X_train_filepath,y_train_filepath,X_test_filepath,y_test_filepath,create_or_load_string_data)
    VGG11_initial_model = CreateOrLoadInitialModel(VGG11_initial_model_filepath,create_or_load_string_intial)
    VGG11_fitted_model,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list = CreateOrLoadFittedModel(X_train,y_train,X_test,y_test,VGG11_initial_model,train_accuracy_filepath,test_accuracy_filepath,train_loss_filepath,test_loss_filepath,VGG11_fitted_model_filepath,num_epochs,create_or_load_string_fitted)
    original_accuracy,X_test_horizontally_flipped_accuracy,X_test_vertically_flipped_accuracy = ComputeFlippedImagesPerformance(VGG11_fitted_model,X_test,y_test)
    gaussian_noise_variance_accuracy_list = ComputeGaussianImagesPerformance(VGG11_fitted_model,gaussian_noise_variance_list,X_test,y_test)
    CreatePlots(num_epochs,train_accuracy_list,test_accuracy_list,train_loss_list,test_loss_list)
    X_train_augmented,y_train_augmented = CreateOrLoadAugmentedTrainingData(X_train,y_train,X_train_augmented_filepath,y_train_augmented_filepath,create_or_load_string_augmented_training_data)
    VGG11_fitted_model_augmented,train_accuracy_list_augmented,test_accuracy_list_augmented,train_loss_list_augmented,test_loss_list_augmented = CreateOrLoadFittedModel(X_train_augmented,y_train_augmented,X_test,y_test,VGG11_initial_model,train_accuracy_augmented_filepath,test_accuracy_augmented_filepath,train_loss_augmented_filepath,test_loss_augmented_filepath,VGG11_fitted_model_augmented_filepath,num_epochs,create_or_load_string_fitted_augmented)
    original_accuracy_augmented,X_test_horizontally_flipped_accuracy_augmented,X_test_vertically_flipped_accuracy_augmented = ComputeFlippedImagesPerformance(VGG11_fitted_model_augmented,X_test,y_test)
    gaussian_noise_variance_accuracy_list_augmented = ComputeGaussianImagesPerformance(VGG11_fitted_model_augmented,gaussian_noise_variance_list,X_test,y_test)
    PrintOutputs(original_accuracy,X_test_horizontally_flipped_accuracy,X_test_vertically_flipped_accuracy,gaussian_noise_variance_list,gaussian_noise_variance_accuracy_list,original_accuracy_augmented,X_test_horizontally_flipped_accuracy_augmented,X_test_vertically_flipped_accuracy_augmented,gaussian_noise_variance_accuracy_list_augmented)  