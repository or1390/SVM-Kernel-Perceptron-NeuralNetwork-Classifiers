import  numpy as np

class neural_network():
    #initialize parameters
    def __init__(self, num_input, num_hidden, num_output, learning_rate, num_of_iterations):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.bias = None
        self.X_train = None
        self.y_train= None

    '''
    activation function - Sigmoid
    '''
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    '''
    this function initialize the weights and bias 
    '''
    def init_network_parameters(self, n_features):

        self.weights_1 = np.random.uniform(size=(n_features, self.num_hidden), low=-1, high=1)
        self.weights_2 = np.random.uniform(size=(self.num_hidden, self.num_output), low=-1, high=1)

        self.bias_1 = np.random.rand(self.num_hidden, )
        self.bias_2 = np.random.rand(self.num_output, )

    '''
    this function evaluates the cost of forward propagation during the training. 
    It calculates the error of the predicted values using random weights for each layer. 
    The error is calculated for the instances in the training dataset.     
    '''
    def calculate_error_output(self, predicted):
        error = 0
        for index, item in enumerate(predicted):
            error = np.sum((self.y_train[index] - item)**2 / 2)
        return error

    '''
    This function is used to train the model using random weights and bias. 
    Then, it calculates the outputs and inputs of the layers by calling the forward propagation
    '''
    def train (self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        n_samples, n_features = X_train.shape

        self.init_network_parameters(n_features)
        for i in range(self.num_of_iterations):
            hidden_layer = self.forward_propagation(self.X_train,self.weights_1, self.bias_1)
            output_layer = self.forward_propagation(hidden_layer,self.weights_2, self.bias_2)

            output_max = self.find_max(output_layer) # get the max value for each unit in the final layer.
            print("Training label",output_max)
            total_error = self.calculate_error_output(output_max)
            print("Total error", total_error)
         #   self.back_propagation(self.X_train,hidden_layer,output_max)
    '''
    This function is used to for the test(unknown) dataset to predict the labels.
    '''
    def predict(self, X_test):
        self.X_test = X_test
        self.init_network_parameters(self.num_input)
        hidden_predicted = self.forward_propagation(self.X_test, self.weights_1, self.bias_1)
        output_predicted = self.forward_propagation(hidden_predicted,self.weights_2,self.bias_2)
        predicted = self.find_max(output_predicted)
        print("Predicted is:", predicted)

    '''
    This function calculates the accuracy (score) in the testing dataset.
    
    '''
    def score(self, X, y):
        predictions = self.predict(X)
        num_of_correct_prediction = 0
        for indx, y_i in enumerate(y):
            if predictions[indx] == y_i:
                num_of_correct_prediction += 1
        print("Correct pred", num_of_correct_prediction)
        accuracy = float(num_of_correct_prediction) / len(y)
        return (accuracy)

    '''
    This function is used to get the maximum value for each unit in the final layer
    '''
    def find_max(self, output_layer):
        label = []
        for i, element in enumerate (output_layer):
            output = np.argmax(output_layer[i])
            label.append(output)
        return  label

    '''
    In the forward propagation, we calculate the output of each layer, and
    then apply an activation function for each layer.
    '''
    def forward_propagation(self,X, weight, bias):
        net_h = np.dot(X,weight) + bias
        output_with_activation = self.sigmoid(net_h)
        return output_with_activation
    '''
    this function return a matrice with zeros and ones, which is used
    in the backpropagation function
    '''
    def one_hot(self, y):
        one_hot_output = np.zeros((self.y_train.size, self.y_train.max() + 1))
        one_hot_output[np.arange(self.y_train.size), self.y_train] = 1
        one_hot_output = one_hot_output.T
        return one_hot_output

    '''
    ReLu is a activation function to return a positive number
    if the input is a negative number
    '''

    def ReLU(self,Z):
        return Z > 0

    '''
    This function is used to update weights and bias after with the corresponding values
    which are obtained by the backpropragation algorithm.
    '''
    def update_parameters(self, dW1,db1,dW2,db2):
        self.weights_1 = self.weights_1 -self.learning_rate * dW1
        self.weights_2 = self.weights_2 - self.learning_rate * dW2
        self.b1 = self.bias_1 - self.learning_rate * db1
        self.b2 = self.bias_2 - self.learning_rate * db2

    '''
    Backpropagation starts from the final layer and uses the Chain Rule 
    to compute the gradients.
    
    '''

    def back_propagation(self, X, hidden_layer, output):
        m= self.y_train.size
        one_hot_Y = self.one_hot(self.y_train)
        dZ2 = output - one_hot_Y
        dW2 = 1 / m * dZ2.dot(hidden_layer)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = np.dot(self.weights_2,dZ2) * self.ReLU(X)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2
        self.update_parameters(dW1, db1, dW2, db2)










