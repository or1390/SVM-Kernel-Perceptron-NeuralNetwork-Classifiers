import numpy as np
from tqdm import tqdm



class kernel_perceptron_binary:
    # parameters initialization
    def __init__(self, b_learning_rate, d_degree, iterations, a):
        self.b_learning_rate = b_learning_rate
        self.iterations = iterations
        self.d_degree = d_degree
        self.alpha = []
        self.a = a
        self.X_train = None


    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        ## initialize alpha, all zeros
        self.alpha = np.zeros(n_samples)
        self.X_train = X_train

        # loop through iterations
        for i in tqdm(range(self.iterations)):
            missclassification = 0
            # loop through instances in the training data
            for indx, x_i in enumerate(X_train):
                # for each instance of the training data calculate the predicted label
                y_predicted = self.predict(x_i)
                # check condition to update alpha
                if y_train[indx] * y_predicted <= 0:
                    missclassification += 1
                    self.alpha[indx] += self.alpha[indx] + y_train[indx]
            if missclassification < 7:
                break



    ''''
    The predict function calculates the kernel for each instance 
    in the training dataset with all other instances in the training dataset

    '''''
    def predict(self, sample):
        sum_value = 0
        for i, x_i in enumerate (self.X_train):
            kernel = (self.a + self.b_learning_rate * np.dot(x_i, sample)) ** self.d_degree
            sum_value += self.alpha[i] * kernel
            #print(sum)
        if sum_value > 1:
            return 1
        return -1


