import numpy as np
from math import log
from math import ceil
from SVM_binary import  support_vector_machine


class SVM_multiclass:
    '''
    Parameters' Initialization
    '''
    def __init__(self, lambda_rate, num_of_iterations, num_of_classes):
        self.SVM_binary1 = None
        self.SVM_binary2 = None
        self.SVM_binary3 = None
        self.SVM_binary4 = None
        self.X_train = None
        self.y_train = None
        self.lambda_rate = lambda_rate
        self.num_of_iterations = num_of_iterations
        self.num_of_classes = num_of_classes

    '''
    decimal to binary conversion
    '''
    def decimal_to_binary(self, num):
        return "{0:04b}".format(int(num))
    '''
    binary to decimal conversion
    '''
    def binaryToDecimal(self, binary):
        decimal = int(binary, 2)
        return (decimal)

    '''
    this function calculates the number of necessary classifiers 
    to handle n-multiclass classification problem
    '''
    def min_num_bits_to_encode_number(self, dec_number):
        dec_number = dec_number + 1  # adjust by 1 for special cases
        # log of zero is undefined
        if 0 == dec_number:
            return 0
        dec_number = int(ceil(log(dec_number, 2)))  # logbase2 is available
        return (dec_number)
    '''
    this is a crucial function which is used to train the model
    '''

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        n_samples, n_features = X_train.shape
        self.num_of_classifiers = self.min_num_bits_to_encode_number(self.num_of_classes)

        current_binary_label = []
        for indx, y_i in enumerate(y_train):
            current_binary_label.append((self.decimal_to_binary(y_train[indx])))

        output = []
        for ele in current_binary_label:
            current = [int(char) for char in ele]
            output.append(current)

        output = np.array(output)
        array_classifier_col1 = output[:, 0] # classifier 1
        array_classifier_col2 = output[:, 1] # classifier 2
        array_classifier_col3 = output[:, 2] # classifier 3
        array_classifier_col4 = output[:, 3] # classifier 4
        '''
        for each classifier we call the SVM binary and flip 0 with -1
        '''
        self.SVM_binary1 = support_vector_machine(self.lambda_rate, self.num_of_iterations)
        self.SVM_binary1.train(X_train, np.where(array_classifier_col1 == 0, -1, array_classifier_col1))
        print("Finish 1")
        self.SVM_binary2 = support_vector_machine(self.lambda_rate, self.num_of_iterations)
        self.SVM_binary2.train(X_train, np.where(array_classifier_col2 == 0, -1, array_classifier_col2))
        print("Finish 2")
        self.SVM_binary3 = support_vector_machine(self.lambda_rate, self.num_of_iterations)
        self.SVM_binary3.train(X_train, np.where(array_classifier_col3 == 0, -1, array_classifier_col3))
        print("Finish 3")
        self.SVM_binary4 = support_vector_machine(self.lambda_rate, self.num_of_iterations)
        self.SVM_binary4.train(X_train, np.where(array_classifier_col4 == 0, -1, array_classifier_col4))
        print("Finish 4")

    def predict(self, X):
        self.X = X
        y_predicted = []
        for indx, x_i in enumerate(self.X):
            y_label_1 = self.SVM_binary1.predict(x_i)
            y_label_2 = self.SVM_binary2.predict(x_i)
            y_label_3 = self.SVM_binary3.predict(x_i)
            y_label_4 = self.SVM_binary4.predict(x_i)
            y_label = []

            y_label.append(y_label_1)
            y_label.append(y_label_2)
            y_label.append(y_label_3)
            y_label.append(y_label_4)

            binary_list = ''.join([str(elem) for elem in y_label])
            y_label = str.replace(binary_list, "-1", "0")
            binary_label = y_label.replace(" ", "")
            y_multiclass = self.binaryToDecimal(binary_label)
            #to manage the cases when the conversion from binary to decimal is greater than the number of
            #multiple classes , we consider the highest value
            if y_multiclass > 9:
                y_multiclass = 9
            y_predicted.append(y_multiclass)
        return y_predicted
    '''
    this function evaluates the number of correct predictions, the accuracy
    '''
    def score(self, X, y):
        predictions = self.predict(X)
        num_of_correct_prediction = 0
        for indx, y_i in enumerate(y):
            if predictions[indx] == y_i:
                num_of_correct_prediction += 1

        accuracy = float(num_of_correct_prediction) / len(y)
        return (accuracy)

