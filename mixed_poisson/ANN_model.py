import tensorflow as tf
from tensorflow.keras.layers import Dense

class MixedPoissonANNModel(tf.keras.Model):
    def __init__(self, input_size, output_size, num_neurons):
        super().__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_size,))
        self.hidden_layers = tf.keras.Sequential([
            Dense(num_neurons, activation='relu'),
            Dense(num_neurons, activation='relu'),
            Dense(num_neurons, activation='relu')
        ])
        self.output_layer = Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layers(x)
        return self.output_layer(x)

# Example usage
input_size = 10  # User-defined input size
output_size = 5  # User-defined output size
num_neurons = 25 # User-defined number of neurons per layer


model = MixedPoissonANNModel(input_size, output_size, num_neurons)
model.build((2, input_size))
model.summary()
