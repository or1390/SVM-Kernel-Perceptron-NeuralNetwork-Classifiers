# Mini-Project 2 - Neulral Network, Support Vector Machine and Kernel Perception algorithms
#### This project consists on implementation of three algorithms: Neural Network, Support Vector Machine and Kernel Perception.
#### The dataset used to train and test the models is MNIST. 

#### 1. In the main.py file, we do:
* Data reading and preprocessing
    * #### The dataset was downloaded from the link:  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist
    * #### From the list download: mnist.scale.bz2
    * #### Read the dataset and get a subset to run it faster
    * #### Split the training and test data using a ratio of 0.3 test size.

* Neural Network-algorithm
    * #### We have implemented a two-layer neural network for binary classification with a hidden layer of H=100 units and K=2 output units, one for each class
    * #### We do train and test our neural network and calculate the training and testing accuracies. 
    * #### In addition, we report the resut of the confusion matrix

* Support Vectort Machine- Pegasos Paper
    * #### This algorithm is originally designed for binary classification, but we have used Error-correcting output code(ECOC) to adapt it for multiclass classification
    * #### We do train and test our neural network and calculate the training and testing accuracies. 
    * #### In addition, we report the resut of the confusion matrix


* Kernel Perception
    * #### We have extended the perceptron training and prediction to use polynomial kernels
    * #### We do train and test our neural network and calculate the training and testing accuracies. 
    * #### In addition, we report the resut of the confusion matrix


#### 2. In the neural_network.py file, we do:
   * Initialization of the weights and biases for two layers by randomly generating them within a range (-1,1).The matrix dimensions of the weights in the first layer are obtained by: the number of features(inputs) and number of hidden units. Then, the matrix dimensions for the weights in the second layer are obtained by: number of hidden units and number of outputs.
   * train method – for each iteration defined as a parameter, we iterate over each instance of the training data. For each instance, we propagate to the hidden units of the next layer. The activation function applied is sigmoid function. Once the output is calculated, the next step is to update the weights by calling backpropagation function.The backpropagation is implemented by using the Chain Rule. 
   * predict method – for each instance in the testing dataset, we call the forward propagation and apply the sigmoid function for each layer. 
   * score function – calculate the accuracy of the training and testing dataset depending on the parameters when it is called.
  
      * #### Find below the results from the Neural Network
      ![image](https://github.com/or1390/mini-project2/blob/9a9c5db28eb9ce032442c3e733ca4bd274c05b48/neural_network.png) 
      ![image](https://github.com/or1390/mini-project2/blob/8f0bf31ba2e2435fc9252d8930d8d6f5c145a9f5/neural_network_confusion_matrix.png)



#### 3. In the SVM_multiclass.py  and SVM_binary.py file:
#### Our dataset is a multiclass classification problem ( labels 0-9). Therefore, we have extended from binary to multiclass classification. We do import SVM_binary.py file in the SVM_multiclass.py file
   * #### Find below the results from the SVM      
   ![image](https://github.com/or1390/mini-project2/blob/f80c5ad786a7f2b11d020f3422d42adcd06e0a13/SVM.png)
   ![image](https://github.com/or1390/mini-project2/blob/d8e814fc043c1e31d1b145a8825d596344c4a3c6/SVM_confusin_matrix.png)

#### 4. In the kernel_perceptron_multiclass.py  and kernel_perceptron_binary.py file:
   ![image](https://github.com/or1390/mini-project2/blob/9a9c5db28eb9ce032442c3e733ca4bd274c05b48/kernel_perceptron.png)
   ![image](https://github.com/or1390/mini-project2/blob/9a9c5db28eb9ce032442c3e733ca4bd274c05b48/kernel_perceptron_matrix.png)






