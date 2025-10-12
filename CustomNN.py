import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class ActivationFunction:
    @staticmethod
    def softmax(x):
        """
        Compute the Softmax activation function.

        Args:
            x (np.array): Input values

        Returns:
            np.array: Softmax activation values
        """
        # Soustraire le max pour stabilité numérique (éviter overflow)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
        # Normaliser
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        #print('Input : ', x,' Softmax :', softmax)
        return softmax


    @staticmethod
    def softmax_derivative(x):
        """
        Compute the derivative of the Softmax function.

        Args:
            x (np.array): Input values

        Returns:
            np.array: Derivative of Softmax function
        """
        s = ActivationFunction.softmax(x)
        return s * (1 - s) # This is the derivative for a single output softmax, not multiclass softmax.
        


    @staticmethod
    def relu(x):
        """
        Compute the ReLU activation function.

        Args:
            x (np.array or float): Input values

        Returns:
            np.array or float: ReLU activation values
        """
        # Use np.maximum which works for both scalars and arrays
        relu = np.maximum(0, x)
        return relu

    @staticmethod
    def relu_derivative(x):
        """
        Compute the derivative of the ReLU activation function.

        Args:
            x (np.array or float): Input values

        Returns:
            np.array or float: Derivative of ReLU function
        """
        # Ensure the output has the same shape as the input
        dev_relu = np.where(x > 0, 1, 0)
        return dev_relu
    
class Neuron:
    def __init__(self, input_size, activation_function):
        """
        Initialize a neuron with random weights and bias.

        Args:
            input_size (int): Number of input features
            activation_function (function): Activation function to use
        """
        ### Write your code here ###
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size) #He initialization
        self.bias = 0.0 # Initalize bias to 0 as a size 1 array

    def activate(self, x):
        """
        Compute the activation of the neuron.

        Args:
            x (np.array): Input features

        Returns:
            float: Activation value
        """
        val = np.dot(np.asarray(self.weights), np.asarray(x)) + self.bias
        return self.activation_function(val)
    
class Layer:
    def __init__(self, input_size, output_size, activation_function):
        """
        Initialize a layer of neurons.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of neurons in the layer
            activation_function (function): Activation function to use
        """
        self.neurons = [Neuron(input_size, activation_function) for _ in range(output_size)]
        self.activation_function = activation_function

    def forward(self, x):
        """
        Compute the forward pass through the layer.

        Args:
            x (np.array): Input features

        Returns:
            np.array: Output of the layer
        """
        if self.activation_function == ActivationFunction.softmax:
            pre_activations = np.array([np.dot(neuron.weights, x) + neuron.bias for neuron in self.neurons])
            return ActivationFunction.softmax(pre_activations)
        else:
            # Pour les autres fonctions d'activation, on utilise activate de chaque neurone
            return np.array([neuron.activate(x) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_dim, activation_function_hidden, activation_output):
        """
        Initialize a neural network with specified layer sizes.

        Args:
            layer_sizes (list): List of integers representing the size of each layer
            hiden_dim (int): Dimension of hidden layers
            activation_function_hidden (function): Activation function to use for hidden layers
            activation_output (function): Activation function to use for output layer
        """
        self.layers = []
        self.hidden_dim = hidden_dim
        self.activation_function_hidden = activation_function_hidden
        self.activation_output = activation_output
        # Input layer and hidden layers
        input_size = layer_sizes[0]
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(input_size, self.hidden_dim, self.activation_function_hidden))
            input_size = self.hidden_dim
        # Output layer
        self.layers.append(Layer(input_size, layer_sizes[-1], self.activation_output))
        self.loss_history = []


    def forward(self, x):
        """
        Compute the forward pass through the entire network.

        Args:
            x (np.array): Input features

        Returns:
            np.array: Output of the network
        """
        output_network = np.array(x)
        for layer in self.layers:
            output_network = layer.forward(output_network) # recursive call through the network
        return output_network

    def layer_backward(self, delta, layer_index, activations, zs):
        """
        Perform backpropagation for a single layer.

        Args:
            delta (np.array): Gradient of the loss with respect to the output of the current layer
            layer_index (int): Index of the current layer
            activations (list): List of activations from the forward pass
            zs (list): List of z values from the forward pass

        Returns:
            tuple: (delta for the previous layer, gradients for weights, gradients for biases)
        """
        layer = self.layers[layer_index]
        z = zs[layer_index]
        a_prev = activations[layer_index]

        # Calculate gradients for weights and biases for the current layer
        # Reshape delta and a_prev for matrix multiplication if needed
        if delta.ndim == 1:
            delta_reshaped = delta.reshape(-1, 1)
        else:
            delta_reshaped = delta

        if a_prev.ndim == 1:
            a_prev_reshaped = a_prev.reshape(1, -1)
        else:
            a_prev_reshaped = a_prev

        gradient_w = np.dot(delta_reshaped, a_prev_reshaped)
        gradient_b = delta # Bias gradient is the delta

        # Calculate delta for the previous layer
        delta_prev = None
        if layer_index > 0:
            # Form the weight matrix for the current layer
            layer_weights = np.array([neuron.weights for neuron in layer.neurons])
            delta_prev = np.dot(layer_weights.T, delta)

            # If it's a hidden layer, multiply delta by the derivative of the activation function of the previous layer's output (z)
            # This is done in the main backward loop after getting delta_prev

        return delta_prev, gradient_w, gradient_b


    def backward(self, x, y, learning_rate):
        """
        Perform backpropagation and update weights.

        Args:
            x (np.array): Input features
            y (np.array): True labels
            learning_rate (float): Learning rate for weight updates
        """
        # Forward pass - Store activations and z values for backward pass
        activations = [x]
        zs = []
        for layer in self.layers:
            # Calculate z for each neuron in the layer
            z_layer = []
            # The input to the layer is activations[-1]
            input_to_layer = activations[-1]
            for neuron in layer.neurons:
                z_layer.append(np.dot(np.array(neuron.weights), np.array(input_to_layer)) + neuron.bias)
            z = np.array(z_layer)
            zs.append(z)
            # Calculate activation for each neuron in the layer
            activation = layer.forward(input_to_layer) # layer.forward already applies the activation
            activations.append(activation)


        # Calculate initial delta for the output layer
        delta = self.loss_derivative(activations[-1], y) # dL/d(output)

        # Backward pass - Iterate backwards through the layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            z = zs[i]
            a_prev = activations[i] # Activation of the previous layer

            # If it's a hidden layer, we need to multiply the incoming delta by the derivative of the activation function
            if i < len(self.layers) - 1: # For hidden layers
                if self.activation_function_hidden == ActivationFunction.relu:
                    activation_derivative = ActivationFunction.relu_derivative(z)
                elif self.activation_function_hidden == ActivationFunction.softmax: # Softmax is typically not used in hidden layers
                    activation_derivative = ActivationFunction.softmax_derivative(z)
                else:
                    raise ValueError("Unsupported activation function for hidden layer")
                delta = delta * activation_derivative


            # Calculate gradients for weights and biases
            gradient_w = np.outer(delta, a_prev)
            gradient_b = delta


            # Calculate delta for the previous layer
            # delta_prev = dL/d(a_prev) = dL/dZ * dZ/d(a_prev)
            # dZ/d(a_prev) is the weight matrix of the current layer
            if i > 0:
                # Form the weight matrix for the current layer
                layer_weights = np.array([neuron.weights for neuron in layer.neurons])
                delta = np.dot(layer_weights.T, delta) # Propagate delta to the previous layer


        # Update weights and biases - Iterate forwards through the layers for updating
        # The gradients calculated in the backward pass are for each layer.
        # We need to apply these gradients to the corresponding layer's neurons.
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # We need to recalculate the gradients for each layer since delta has changed
            activations = [x]
            zs = []
            for current_layer in self.layers:
                input_to_layer = activations[-1]
                z_layer = []
                for neuron in current_layer.neurons:
                    z_layer.append(np.dot(np.array(neuron.weights), np.array(input_to_layer)) + neuron.bias)
                z = np.array(z_layer)
                zs.append(z)
                activation = current_layer.forward(input_to_layer)
                activations.append(activation)

            # Calculate initial delta for the output layer
            delta = activations[-1] - y

            # Lists to store gradients
            gradients_w = [None] * len(self.layers)
            gradients_b = [None] * len(self.layers)

            # Backward pass to calculate and store gradients
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                z = zs[i]
                a_prev = activations[i]

                # If it's a hidden layer, apply derivative of the activation function
                if i < len(self.layers) - 1:
                    if self.activation_function_hidden == ActivationFunction.relu:
                        activation_derivative = ActivationFunction.relu_derivative(z)
                    elif self.activation_function_hidden == ActivationFunction.softmax:
                        activation_derivative = ActivationFunction.softmax_derivative(z)
                    else:
                        raise ValueError("Unsupported activation function for hidden layer")
                    delta = delta * activation_derivative


                # Calculate gradients for weights and biases
                gradient_w = np.outer(delta, a_prev)
                gradient_b = delta

                # Store gradients
                gradients_w[i] = gradient_w
                gradients_b[i] = gradient_b


                # Calculate delta for the previous layer
                if i > 0:
                    layer_weights = np.array([neuron.weights for neuron in layer.neurons])
                    delta = np.dot(layer_weights.T, delta)


            # Update weights and biases - Iterate forwards to apply stored gradients
            for i in range(len(self.layers)):
                layer = self.layers[i]
                gradient_w = gradients_w[i]
                gradient_b = gradients_b[i]

                for j, neuron in enumerate(layer.neurons):
                    # Update weights using the corresponding row of the gradient_w matrix
                    neuron.weights -= learning_rate * gradient_w[j, :]
                    # Update bias using the corresponding element of the gradient_b array
                    neuron.bias -= learning_rate * gradient_b[j]


    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network.

        Args:
            X (np.array): Training features
            y (np.array): Training labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for weight updates
        """
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                # Calculate loss for the current sample
                output = np.array(self.forward(X[i]))
                output = self.forward(X[i])
                loss = self.compute_loss(output, y[i])
                total_loss += loss
                self.backward(X[i], y[i], learning_rate)

            average_loss = total_loss / len(X)
            self.loss_history.append(average_loss)

            if epoch % 1 == 0: # Print every epoch for a clearer view of training progress
                predictions = self.predict(X)
                accuracy = self.compute_accuracy(predictions, y) # Use compute_accuracy
                print(f"Epoch {epoch}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def compute_loss(self, predictions, y_true):
        """
        Compute cross-entropy loss.

        Args:
            predictions (np.array): Model predictions
            y_true (np.array): True labels (one-hot encoded)

        Returns:
            float: Cross-entropy loss
        """
        # Avoid log(0) by adding a small constant
        loss = -np.sum(y_true * np.log(predictions + 1e-9)) / y_true.shape[0]
        return loss
    
    def loss_derivative(self, predictions, y_true):
        """
        Compute the derivative of the cross-entropy loss.

        Args:
            predictions (np.array): Model predictions
            y_true (np.array): True labels (one-hot encoded)

        Returns:
            np.array: Derivative of the loss
        """
        return predictions - y_true


    def compute_accuracy(self, predictions, y_true):
        """
        Compute accuracy of the model.

        Args:
            predictions (np.array): Model predictions
            y_true (np.array): True labels (one-hot encoded)

        Returns:
            float: Accuracy
        """
        # Convert one-hot encoded true labels to class indices
        y_true_indices = np.argmax(y_true, axis=1)
        # Get predicted class indices
        predicted_indices = np.argmax(predictions, axis=1)
        # Compare and compute accuracy
        accuracy = np.mean(predicted_indices == y_true_indices)
        return accuracy


    def predict(self, X):
        """
        Make predictions for a set of input features.

        Args:
            X (np.array): Input features

        Returns:
            np.array: Predictions
        """
        return np.array([self.forward(x) for x in X])
    

np.random.seed(42) # Do not change
# For easy data loading and processing, we're using torch here. You're not allowed to use torch for Neural Net components though.
def get_train_test_dataset():
    import torch
    from torchvision import datasets, transforms

    # Define a transform to convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Take a subset of 1000 samples
    X = mnist_dataset.data[:1000].float() / 255.0  # Normalize pixel values to [0,1]
    y = mnist_dataset.targets[:1000]

    # Creat test / test split
    x_train, x_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # to be completed
    #print("Training set shape:", x_train.shape)
    #print("Test set shape :", x_test.shape)

    return x_train, y_train, x_test, y_test

def train_network(hidden_layer_size, hidden_dim, epochs, activation_hidden, activation_output, learning_rate):
    input_size, output_size = 28 * 28, 10
    layers = [input_size, hidden_layer_size, output_size]

    nn = NeuralNetwork(layers, hidden_dim, activation_hidden, activation_output)

    X_train, y_train, X_test, y_test = get_train_test_dataset()

    # --- Prepare training data ---
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_np = y_train.numpy()

    # One-hot encode labels
    y_train_onehot = np.eye(output_size)[y_train_np]

    # Train
    train_loss = nn.train(X_train_flat, y_train_onehot, epochs=epochs, learning_rate=learning_rate)

    # --- Evaluate on training data ---
    preds_train = nn.predict(X_train_flat)
    train_accuracy = np.mean(np.argmax(preds_train, axis=1) == y_train_np)

    # --- Evaluate on test data ---
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_test_np = y_test.numpy()
    preds_test = nn.predict(X_test_flat)
    test_accuracy = np.mean(np.argmax(preds_test, axis=1) == y_test_np)

    return nn, train_loss, train_accuracy, test_accuracy

hidden_layer_size = 128
hidden_dim = 128
epochs = 35
activation_hidden = ActivationFunction.relu
activation_output = ActivationFunction.softmax
learning_rate = 0.001
nn, train_loss, train_accuracy, test_accuracy = train_network(hidden_layer_size, hidden_dim,  epochs, activation_hidden, activation_output, learning_rate)


print("Train accuracy: ", train_accuracy)
print("Test accuracy: ", test_accuracy)

# Plot the training loss across 20 epochs

def plot_loss_curve(train_loss, val_loss=None, test_acc=None, title="Training Loss Across Epochs"):
    """Plot training loss and optional validation loss across epochs and annotate test accuracy.

    Args:
        train_loss (list or array): training loss per epoch
        val_loss (list or array, optional): validation loss per epoch (same length as train_loss)
        test_acc (float, optional): test accuracy in percent (0-100)
        title (str): plot title
    """
    epochs = len(train_loss)
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(9, 5))
    plt.plot(x, train_loss, marker='o', label='Train Loss')
    if val_loss is not None:
        # if val_loss shorter/longer, truncate/pad to match train length gracefully
        val_plot = val_loss[:epochs]
        plt.plot(x, val_plot, marker='o', label='Validation Loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # choose sensible xticks
    step = max(1, epochs // 10)
    plt.xticks(range(1, epochs + 1, step))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    if test_acc is not None:
        # annotate test accuracy on the plot
        plt.gca().text(0.98, 0.95, f'Test Acc: {test_acc:.2f}%', transform=plt.gca().transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Backward-compatible call: existing code called plot_loss_curve(nn.loss_history)
# plot_loss_curve(nn.loss_history)


# ===== Latent Space Visualization =====
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import inspect # Import inspect for type checking

def get_penultimate_activations(nn, X_batch):
    """Get activations from the last hidden layer"""
    # Check if nn is an instance of the custom NeuralNetwork and has layers
    if not hasattr(nn, 'layers') or not isinstance(nn.layers, list):
        print(f"Error: Expected nn to have a 'layers' attribute which is a list, but got type {type(nn)} with layers: {getattr(nn, 'layers', 'N/A')}")
        return np.array([]) # Return empty array or raise error

    # Process each sample individually
    activations_list = []
    for x in X_batch:
        activations = x
        # Iterate through layers, excluding the last one
        for i in range(len(nn.layers) - 1): # Iterate through all layers except the output layer
            layer = nn.layers[i]
            # The forward pass in the custom Layer class expects a single input sample
            # Ensure the layer is a custom Layer instance before calling forward
            if not isinstance(layer, Layer):
                print(f"Error: Expected layer at index {i} to be a custom Layer instance, but got type {type(layer)}")
                return np.array([]) # Return empty array or raise error

            activations = layer.forward(activations)
        activations_list.append(activations)
    return np.array(activations_list)

def visualize_activations(nn, X_test, y_test, digits=(3, 8), method='PCA'):
    """Visualize layer activations using PCA or t-SNE"""
    # Get activations from penultimate layer
    Z = get_penultimate_activations(nn, X_test)

    # If Z is empty, skip visualization
    if Z.size == 0:
        print("Skipping visualization due to error in getting activations.")
        return None

    # Apply dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)

    Z_2d = reducer.fit_transform(Z)

    # Plot visualization
    plt.figure(figsize=(12, 5))

    # Plot 2: Focus on specific digits
    plt.subplot(1, 2, 2)
    mask1 = y_test == digits[0]
    mask2 = y_test == digits[1]

    # write your code here
    plt.scatter(Z_2d[mask1, 0], Z_2d[mask1, 1], label=f'Digit {digits[0]}', alpha=0.5)
    plt.scatter(Z_2d[mask2, 0], Z_2d[mask2, 1], label=f'Digit {digits[1]}', alpha=0.5)
    plt.legend()

    plt.title(f'Latent Space: Digits {digits[0]} vs {digits[1]}')

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.show()

    return Z_2d

# Get test dataset to use for visualization
_, _, X_test, y_test = get_train_test_dataset()
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_test_np = y_test.numpy()

# Visualize activations for digits 3 and 8 using PCA
#visualize_activations(nn, X_test_flat, y_test_np, digits=(3, 8), method='PCA')

# Visualize activations for digits 3 and 8 using t-SNE
#visualize_activations(nn, X_test_flat, y_test_np, digits=(3, 8), method='TSNE')