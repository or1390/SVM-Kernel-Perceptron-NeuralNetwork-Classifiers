import numpy as np
from math import log
from math import ceil
import re
from kernel_perceptron_binary import kernel_perceptron_binary


class kernel_perceptron_multiclass:
    # parameters initialization
    def __init__(self, b_learning_rate, d_degree, iterations, a, num_of_classes):
        self.b_learning_rate = b_learning_rate
        self.iterations = iterations
        self.d_degree = d_degree
        self.alpha = []
        self.a = a
        self.num_of_classes = num_of_classes
        self.num_of_classifiers = None
        self.kernel_binary1 = None
        self.kernel_binary2 = None
        self.kernel_binary3 = None
        self.kernel_binary4 = None
        self.X_train = None
        self.y_train = None


    def decimal_to_binary(self, num):
        return "{0:04b}".format(int(num))

    def binaryToDecimal(self,binary):
        decimal = int(binary,2)
        return (decimal)

    def min_num_bits_to_encode_number(self,dec_number):
        dec_number = dec_number + 1  # adjust by 1 for special cases
        # log of zero is undefined
        if 0 == dec_number:
            return 0
        dec_number = int(ceil(log(dec_number, 2)))  # logbase2 is available
        return (dec_number)


    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        count_correct = 0
        n_samples, n_features = X_train.shape
        self.num_of_classifiers = self.min_num_bits_to_encode_number(self.num_of_classes)

        current_binary_label = []
        # initialize alpha, all zeros
        self.alpha = np.zeros(n_samples)

        for indx, y_i in enumerate(y_train):
            current_binary_label.append((self.decimal_to_binary(y_train[indx])))

        output=[]
        for ele in current_binary_label:
            current = [int(char) for char in ele]
            output.append(current)

        output = np.array(output)
        array_classifier_col1 = output[:,0]
        array_classifier_col2 = output[:,1]
        array_classifier_col3 = output[:,2]
        array_classifier_col4 = output[:,3]

        kernel_binary1 = kernel_perceptron_binary(0.01, 1, 10, 1)
        kernel_binary1.train(X_train, np.where(array_classifier_col1==0, -1, array_classifier_col1))
        print("Finish 1")
        kernel_binary2 = kernel_perceptron_binary(0.01, 1, 10, 1)
        kernel_binary2.train(X_train, np.where(array_classifier_col2==0, -1, array_classifier_col2))
        print("Finish 2")
        kernel_binary3 = kernel_perceptron_binary(0.01, 1, 10, 1)
        kernel_binary3.train(X_train,np.where(array_classifier_col3==0, -1, array_classifier_col3))
        print("Finish 3")
        kernel_binary4 = kernel_perceptron_binary(0.01, 1, 10, 1)
        kernel_binary4.train(X_train,np.where(array_classifier_col4==0, -1, array_classifier_col4))

        for indx, x_i in enumerate(X_train):
            y_label_1 = kernel_binary1.predict(x_i)
         #   print(y_label_1)
            y_label_2 = kernel_binary2.predict(x_i)
         #   print(y_label_2)
            y_label_3 =kernel_binary3.predict(x_i)
          #  print(y_label_3)
            y_label_4 = kernel_binary4.predict(x_i)
           # print(y_label_4)
            y_label = []

            y_label.append(y_label_1)
            y_label.append(y_label_2)
            y_label.append(y_label_3)
            y_label.append(y_label_4)

            binary_list = ''.join([str(elem) for elem in y_label])


            y_label = str.replace(binary_list, "-1"," 0")
            no_space =re.sub(" +", " ", y_label)

            print(y_label)

            y_multiclass = self.binaryToDecimal(y_label)
            print(y_multiclass)

        if(y_train[indx] == y_multiclass):
           count_correct += count_correct
        print(count_correct)






