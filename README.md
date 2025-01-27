# Machine-Learning-Algorithms

## Interesting Topics Explored

### 1. Spam Classification with Ensemble Methods
Classified spam and non-spam emails using decision trees, bagging, boosting, and random forests methods in Python. The implementation leverages the Numpy library for efficient computation, ensuring effective classification across diverse datasets.

### 2. GMM Model for PCA Transformed MNIST Datasets
Developed a Gaussian Mixture Model (GMM) to classify Principal Component Analysis (PCA) transformed MNIST datasets. This project explores the application of unsupervised learning techniques for effective data representation and classification.

### 3. VAE and GAN for MNIST Image Generation
Programmed a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) to generate MNIST image samples. The implementation is in Python using the PyTorch library, showcasing the use of deep learning models for image synthesis.

### 4. Robust CNN with Adversarial Training
Trained a Convolutional Neural Network (CNN) model on perturbed MNIST images. The training process involved the use of Fast Gradient Sign Method (FGSM) and Projected Gradient Descent to enhance the model's resilience against adversarial attacks.

### 5. Logistic Regression for Membership Inference Protection
Produced regularized and unregularized logistic regression models to protect against membership inference attacks using the MNIST dataset. This project addresses the security concerns associated with privacy-preserving machine learning models.

## Algorithms Used

### Regression Models

#### RidgeRegressionUsingClosedForm.py: 
Implementing Ridge Regression

#### RidgeRegressionWithGradientDescent.py: 
Implementing Ridge Regression With Gradient Descent

#### CompareClosedFormAndGradientDescentRidgeRegression.py: 
Testing Two Ridge Regression Models on Boston Housing Dataset

#### CompareLinRegRidgeRegLassoRegKFold.py: 
Implementing Linear Regression, Ridge Regression With K-Fold Cross Validation, and Lasso Regression With K-Fold Cross Validation

### K-Nearest Neighbours

#### KNNClassifier.py: 
Implementing K-Nearest Neighbour Regression

#### CompareKNNClassifierLinReg.py: 
Further Testing of K-Nearest Neighbour Regression vs Linear Regression on Various Mystery Datasets

#### CompareKNNClassifierLeastSquares.py: 
Further Testing of K-Nearest Neighbour Regression vs Least Squares Regression on Various Mystery Datasets

### Logistic Regression and SVMs

#### LogReg_HMSVM_SMSVM.py: 
Implementing Logistic Regression, Hard-Margin SVM, Soft-Margin SVM on Mystery Dataset

#### LogReg_HMSVM_SMSVM_2.py: 
Further Use of Logistic Regression, Hard-Margin SVM, Soft-Margin SVM on Mystery Dataset

#### SMSVMGradDesc.py: 
Implementing Soft-Margin SVM With Gradient Descent

### Decision Trees

#### DecisionTree.py: 
Implementing Decision Trees

#### DecisionTreeBagging.py: 
Implementing Decision Trees With Bagging

### Gausian Mixture Models

#### GMM.py: 
Implementing GMM

#### ModifiedGMM.py: 
Implementing Modified GMM 

### Generative Models

#### VAE.py: 
Implementing VAE

#### GAN.py: 
Implementing GAN

### CNN and Adversarial Attacks

#### CNN.py: 
Implementing CNN

#### CNN2.py: 
Additiona CNN implementation

#### CNNFGSMTrain.py: 
Testing CNN on FGSM Adversarially Generated MNIST Images 

#### CNNFGSMTest.py: 
Training CNN on FGSM Adversarially Generated MNIST Images

#### CNN_FGSM_vs_PDG.py: 
Comparing Previous CNN Models With PGD Adversarially Generated MNIST Images

### Logistic Regression and Membership Inference

#### MembershipInfCreatedData.py: 
Membership Inference on Created Data

#### MembershipInfMNISTData.py: 
Membership Inference on MNIST Images

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mbadalbadalian/Machine_Learning_Algorithms.git
