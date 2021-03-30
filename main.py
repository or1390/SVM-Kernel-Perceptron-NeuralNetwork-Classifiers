import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from joblib import  Memory
from kernel_perceptron import kernel_perceptron

mem = Memory("./mycache")
@mem.cache
def get_data():
    x, label = load_svmlight_file('mnist.scale')
    x = x.toarray()
    label = label.astype(int)
    return  x, label

x, label = get_data()
subset_size = 10
x_new = x[0:subset_size,]
y_new = label[0:subset_size]


X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.3, random_state=42)


kernel_prc = kernel_perceptron(0.01, 1, 10, 1, 10)
kernel_prc.train(X_train, y_train)
#y_prediction = kernel_prc.predict(X_test)
#accuracy = np.sum(y_prediction == y_test) / len(y_test)
#print("Accuracy for Kernel_Perception is:\n", accuracy)
