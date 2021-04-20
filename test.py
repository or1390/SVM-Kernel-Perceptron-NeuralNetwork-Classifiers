import  numpy as np

class neural_network():

    def __init__(self, num_input, num_hidden, num_output, learning_rate, bias):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights= []
        self.net_h1 = []
        self.net_h2 = []
        self.output_neurons_of_layer =[]
        self.X_train = None
        self.y_train= None


    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def train (self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        n_samples, n_features = X_train.shape
        hidden_layer = []
        output_layer = []
        count = 0
        low = -1
        high = 1
        self.weights_1 = np.random.uniform(low, high, n_features)
        self.weights_2 = np.random.uniform(low, high, len(hidden_layer))
        self.bias = 1


        for index_1, item_1 in enumerate (X_train):
            hidden_layer.append(self.neurons_of_first_layer(X_train[index_1]))

        print("Output of the 1st layer", hidden_layer)
        print("Length of 1 st layer",len(hidden_layer))

        for index_2, item_2 in enumerate(output_layer):
            output_layer.append(self.neurons_of_second_layer(output_layer[index_2]))
      #  print("Output of the 2nd layer", output_layer)
      #  print("Length of 2 nd layer", len(output_layer))



    def neurons_of_first_layer(self, element):
        output = []
        out_01 = []

        self.net_h1 = np.dot(self.weights, element) + self.bias
        output.append(self.net_h1)

        for i , neuron in enumerate (output):
            net_with_activation_function_1 = self.sigmoid(output[i])

        return net_with_activation_function_1

    def neurons_of_second_layer(self, element):
        output_2 = []
        out_02 = []
        self.net_h2 = np.dot(self.weights, element) + self.bias
        output_2.append(self.net_h2)

        for i, neuron in enumerate(output_2):
            net_with_activation_function_2 = self.sigmoid(output_2[i])

        return net_with_activation_function_2




