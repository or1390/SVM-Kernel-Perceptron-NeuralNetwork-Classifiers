import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from joblib import  Memory
from kernel_perceptron_binary import kernel_perceptron_binary
from kernel_perceptron_multiclass import kernel_perceptron_multiclass
from neural_network import neural_network
from SVM_binary import support_vector_machine
from SVM_multiclass import  SVM_multiclass

mem = Memory("./mycache")
@mem.cache
def get_data():
    x, label = load_svmlight_file('mnist.scale')
    x = x.toarray()
    label = label.astype(int)
    return  x, label


x, label = get_data()
subset_size = 1000
# x = x[:subset_size,]
# label = label[:subset_size]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(x, label, test_size=0.3)

X_train = X_train_new[0:int(subset_size * 0.7),]
y_train = y_train_new[0: int(subset_size * 0.7)]
X_test = X_test_new[0:int(subset_size * 0.3),]
y_test = y_test_new[0: int(subset_size * 0.3)]




#Neural Network
H = 100
K = 10
learning_rate = 0.1
num_of_iterations = 20
n_samples, n_features = X_train.shape
NN = neural_network(n_features,H, K,learning_rate,num_of_iterations )
NN.train(X_train, y_train)
score_training = NN.score(X_train,y_train)
print("Training accuracy for NN is:")
print(score_training)

y_predict_prc =NN.predict(X_test)
score_testing = NN.score(X_test,y_test)
print("Testing accuracy for NN is:")
print(score_testing)

NN_conf_matrix =  confusion_matrix(y_test, y_predict_prc)
print("Confusion matrix for Neural Network is:")
print(NN_conf_matrix)
'''

# Kernel Perceptron
kernel_prc = kernel_perceptron_multiclass(0.01, 3, 50, 1, 10)
kernel_prc.train(X_train, y_train)
training_score = kernel_prc.score(X_train, y_train)
print("Training accuracy for Kernel Perceptron is:")
print(training_score)
#y_predict_prc= kernel_prc.predict([X_test[0]])
y_predict_prc = kernel_prc.predict(X_test)
testing_score = kernel_prc.score(X_test, y_test)
print("Testing accuracy for Kernel Perceptron is:")
print(testing_score)
kernel_prc_conf_matrix =  confusion_matrix(y_test, y_predict_prc)
print("Confusion matrix for Kernel Perceptron is:")
print(kernel_prc_conf_matrix)

# SVM multiclass
num_of_iterations = 200000
lambda_rate = 1
num_of_classes = 10


SVM = SVM_multiclass(lambda_rate, num_of_iterations,num_of_classes)
SVM.train(X_train, y_train)
score_training = SVM.score(X_train, y_train)
print("Training accuracy for SVM is:")
print(score_training)
y_predict_SVM =SVM.predict(X_test)
score_testing = SVM.score(X_test, y_test)
print("Testing accuracy for SVM is")
print(score_testing)

svm_multiclass_conf_matrix =  confusion_matrix(y_test, y_predict_SVM)
print("Confusion matrix for SVM is:")
print(svm_multiclass_conf_matrix)
'''''