import numpy as np
import math

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + math.exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # Calculate the weighted sum, and then compute the output.
            weighted_sum = np.sum(inputs * self.hidden_layer_weights[:4,i])
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.sum(hidden_layer_outputs * self.output_layer_weights[:3,i])
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)
            
        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        # Calculate output layer betas.
        output_layer_betas = (desired_outputs - output_layer_outputs)

        hidden_layer_betas = np.zeros(self.num_hidden)
        # Calculate hidden layer betas.
        output_vector =  np.array((np.ones(3) - output_layer_outputs) * output_layer_betas)
        hidden_layer_betas = np.dot(self.output_layer_weights, output_vector)
        
        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # Calculate output layer weight changes.
        hlo = np.array(hidden_layer_outputs).reshape(2,1)
        olo = np.array(output_layer_outputs).reshape(1,3)
        olb = np.array(output_layer_betas).reshape(1,3)
       
        delta_output_layer_weights = self.learning_rate * hlo * (olo * (1 - olo) * olb)
        
        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # Calculate hidden layer weight changes.
        inps = np.array(inputs).reshape(4,1)
        hlo = hlo.reshape(1,2)
        hlb = np.array(hidden_layer_betas).reshape(1,2)
        
        delta_hidden_layer_weights = self.learning_rate * inps * (hlo * (1 - hlo) * hlb)

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    # Update the weights of the output and hidden layers
    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        self.hidden_layer_weights += delta_hidden_layer_weights
        self.output_layer_weights += delta_output_layer_weights

    # Train the neural network for desired number of epochs
    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            correct_predictions = 0
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = output_layer_outputs.index(max(output_layer_outputs))
                predictions.append(predicted_class)
                outputs = desired_outputs[i].flatten().tolist()
                if predicted_class == outputs.index(max(desired_outputs[i])):
                    correct_predictions += 1

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print accuracy achieved over this epoch
            acc = correct_predictions / len(instances) * 100
            print('accuracy as % = ', acc)

    # Gather the predictions of the Neural Network on given instances
    def predict(self, instances):
        predictions = []
        correct_predictions = 0
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = output_layer_outputs.index(max(output_layer_outputs))
            predictions.append(predicted_class)
        return predictions