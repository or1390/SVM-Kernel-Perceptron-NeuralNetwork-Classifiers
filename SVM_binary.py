import numpy as np
from tqdm import tqdm

class support_vector_machine():
    def __init__(self, lambda_rate, num_of_iterations, num_classes):
        self.lambda_rate = lambda_rate
        self.num_of_iterations= num_of_iterations
        self.X_train = None
        self.y_train = None
        self.weights = []
        self.classes = num_classes
      #  self.n_classes = len(num_classes)


    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features + 1)
        y_label = []
        for i in tqdm(range (self.num_of_iterations)):
            random_index = np.random.choice(n_samples, 1)[0]
            current_step_size = 1 / (self.lambda_rate * (i + 1))
            # print("Step size", current_step_size)
            current_sample , current_label = X_train[random_index] ,y_train[random_index]

            #print("Current instance", current_sample)
            current_label = self.y_train[random_index]
            # print("Actual label", current_label)
            y_predicted = self.predict(current_sample)
            current_sample = np.append(current_sample, 1)
            # print("Predicted label",y_predicted)
            # y_label.append(y_predicted)


            if current_label * y_predicted < 1:
                self.weights = (1-current_step_size *self.lambda_rate)*self.weights + current_step_size* current_label * current_sample
            else:
                self.weights= (1 - current_step_size * self.lambda_rate) * self.weights
            # print(self.weights)
        return (self.weights)

    def predict(self, current_sample):
        # sample = np.array(current_sample)
        # sample = np.append(sample, 1)
        current_sample = np.append(current_sample, 1)
        output = np.dot(self.weights, current_sample)
        if output > 0:
            return 1
        return -1
