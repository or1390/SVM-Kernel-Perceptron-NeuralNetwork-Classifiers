import  numpy as np
from tqdm import tqdm

class neural_network():
    '''
       Parameters' Initialization
    '''
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
    this function initializes the weights and biases 
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

    def expected_label(self, expected):
        e = []
        for i in range(self.num_output):
            if expected == i:
                e.append(1)
            else:
                e.append(0)
        return e


    '''
    This function is used to train the model using random weights and bias. 
    Then, it calculates the outputs and inputs of the layers by calling the forward propagation
    '''
    def train (self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        n_samples, n_features = X_train.shape

        self.init_network_parameters(n_features) #inizialize weights and biases

        for i in tqdm(range(self.num_of_iterations)):
            for x_i, item in enumerate (self.X_train):
                out_1 = self.forward_propagation(self.X_train[x_i], self.weights_1, self.bias_1)
                hidden_predicted = self.sigmoid(out_1)
                out_2 = self.forward_propagation(hidden_predicted, self.weights_2, self.bias_2)
                output_predicted = self.sigmoid(out_2)
                y_actual = self.expected_label(y_train[x_i])
                self.backpropagation(hidden_predicted, output_predicted, y_actual,item,n_features) # backpropagation


    ''' 
    This function is used to for the test(unknown) dataset to predict the labels.
    '''
    def predict(self, X_test):
        self.X_test = X_test
        predictions = []
        for x_i, item in enumerate(X_test):
            out_1 = self.forward_propagation(self.X_test[x_i], self.weights_1, self.bias_1)
            hidden_predicted = self.sigmoid(out_1)
            out_2 = self.forward_propagation(hidden_predicted,self.weights_2,self.bias_2)
            output_predicted = self.sigmoid(out_2)
            predicted = self.find_max(output_predicted)
            predictions.append(predicted)
        return  predictions

    '''
    This function calculates the accuracy (score) in the testing dataset.
    
    '''
    def score(self, X, y):
        predictions = self.predict(X)
        num_of_correct_prediction = 0
        for indx, y_i in enumerate(y):
            if predictions[indx] == y_i:
                num_of_correct_prediction += 1
        accuracy = float(num_of_correct_prediction) / len(y)
        return (accuracy)


    '''
    In this function is implemented the Chain RULE which is used to evaluate the error and 
    to backpropagate in order to update the weights correctly.
    '''
    def backpropagation (self,hidden_predicted, output_predicted,y_actual, item, n_features):

        output_layer_error = output_predicted - y_actual
        output_layer_delta = output_layer_error * output_predicted * (1 - output_predicted)

        output_layer_delta = output_layer_delta.reshape(self.num_output, 1)
        hidden_predicted = hidden_predicted.reshape(1, self.num_hidden)

        hidden_layer_error = np.dot(output_layer_delta, hidden_predicted)
        hidden_layer_delta = hidden_predicted * (1 - hidden_predicted)
        derivative = np.dot(self.weights_2, output_layer_delta)
        derivative = derivative.reshape(1, self.num_hidden)
        layer_1 = derivative * hidden_layer_delta
        input = item.reshape(1, n_features)
        dw1= np.dot(layer_1.T, input)

        self.weights_1 = self.weights_1 - self.learning_rate * dw1.T
        self.weights_2 = self.weights_2 - self.learning_rate * hidden_layer_error.T
    '''
    This function is used to get the maximum value for each unit in the final layer
    '''
    def find_max(self, output_layer):
        return np.argmax(output_layer, axis = 0)

    '''
    In the forward propagation, we calculate the output of each layer, and
    then apply an activation function for each layer.
    '''
    def forward_propagation(self,X, weight, bias):
        net_h = np.dot(X,weight) + bias
        return net_h

