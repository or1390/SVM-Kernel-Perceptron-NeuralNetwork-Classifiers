import  numpy as np

class kernel_perceptron:
    # parameters initialization
    def __init__(self, b_learning_rate, d_degree, iterations, a, classes):
        self.b_learning_rate = b_learning_rate
        self.iterations = iterations
        self.d_degree = d_degree
        self.alpha = []
        self.a = a
        self.classes = classes



    def decimal_to_binary(self, num):
        return "{0:04b}".format(int(num))

    def binary_classifer(bits):
        len_bits = len(bits)
        for indx , bit in enumerate (len_bits):
            pass


    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        num_bits = [4][n_features]
        ## initialize alpha, all zeros
        self.alpha = np.zeros(n_samples)


        # loop through iterations
        for i in range(self.iterations):
            # loop through instances in the training data
            for indx, x_i in enumerate(X_train):
                # for each instance of the training data calculate the predicted label by calling the function predict
           #     y_predicted = self.predict(x_i)
                bits = self.decimal_to_binary(y_train[indx])
                for i in range (4):
                    for j in range(n_samples):
                        num_bits.append(bits)
                        print(num_bits)

           # for i_bits in range(num_bits):
            #    y_predicted = self.binary_classifier(i_bits)

                # check condition to update alpha
              #  if y_train[indx] * y_predicted <= 0:
              #      self.alpha[indx] += self.alpha[indx] + y_train[indx]

    ''''
    The predict function calculates the kernel for each instance 
    in the training dataset with all other instances in the training dataset

    '''''

    def predict(self, sample):
        sum = 0.0
        for i, x_i in enumerate(self.X_train):
            kernel = (self.a + self.b_learning_rate * np.dot(x_i, sample)) ** self.d_degree
            sum += self.alpha[i] * kernel
        if sum > 1:
            return 1
        return -1

    ''''
    the test function is used to predict the label for the instances
    in the testing dataset.    
    '''''

    def test(self, X_test):
        output = []
        for i, i_test in enumerate(X_test):
            output.append(self.predict(i_test))
        return output