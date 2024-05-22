import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from engine import train
from tensorflow.data import Dataset


def create_model(input_shape, output_shape, hidden_layers_neurons, activation='relu'):
    """
    Create a custom ANN model based on user-specified parameters.

    Args:
    - input_shape: tuple, shape of the input data
    - output_shape: tuple, shape of the output data
    - hidden_layers_neurons: list of integers, number of neurons in each hidden layer
    - activation: str, activation function to be used in hidden layers

    Returns:
    - model: tf.keras.Model object, the created ANN model
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    # hidden layers
    for neurons in hidden_layers_neurons:
        model.add(Dense(neurons, activation=activation))
        model.add(tf.keras.layers.BatchNormalization(synchronized=True))

    # output layer
    model.add(Dense(output_shape, activation=activation))  # Assuming only one output neuron for regression
    return model

class MixedPoissonANNModel(tf.keras.Model):
    def __init__(self, input_size, output_size, num_neurons):
        super().__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_size,))
        self.hidden_layers = tf.keras.Sequential([
            Dense(num_neurons, activation='relu'),
            Dense(num_neurons, activation='relu'),
            Dense(num_neurons, activation='relu')
        ])
        self.output_layer = Dense(output_size, activation='relu')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layers(x)
        return self.output_layer(x)

def create_Dataloader(input_dataset, output_dataset, batch_size = 32, 
                      train_percentage = 0.8, shuffling = 100):
    print("#"*80)
    print("Creating training and testing dataloaders")
    train_size = int(len(input_dataset) * train_percentage)
    indices = np.random.permutation(len(input_dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Split the datasets into training and testing datasets
    train_input = Dataset.from_tensor_slices(input_dataset[train_indices])
    train_output = Dataset.from_tensor_slices(output_dataset[train_indices])
    test_input = Dataset.from_tensor_slices(input_dataset[test_indices])
    test_output = Dataset.from_tensor_slices(output_dataset[test_indices])

    # Create TensorFlow datasets for training and testing
    train_dataset = Dataset.zip((train_input, train_output))
    test_dataset = Dataset.zip((test_input, test_output))
    train_dataloader = train_dataset.shuffle(shuffling).batch(batch_size)
    test_dataloader = test_dataset.shuffle(shuffling).batch(2)
    print("Train and test Dataloaders have been successfully created")
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    input_shape = 2
    output_shape = 2

    # Create sample input and output data
    num_samples = 50  # 2 batches of size 10

    input_data = np.linspace(0, 1, num_samples * input_shape).reshape((num_samples, input_shape))
    input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    output_data = np.sin(2 * input_data)

    output_dataset = tf.data.Dataset.from_tensor_slices(output_data)
    BATCH_SIZE = 16
    train_dataloader, test_dataloader = create_Dataloader(input_data, output_data, batch_size = BATCH_SIZE)
    # Iterate over the DataLoader to check the batches
    for batch_idx, (input_batch, output_batch) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Output batch shape: {output_batch.shape}")
        print()
    
    # model = MixedPoissonANNModel(input_shape, output_shape, num_neurons=25)
    hidden_layer_neurons = [25, 25, 25]
    model = create_model(input_shape, output_shape, hidden_layer_neurons, activation = "relu")
    # model.build((BATCH_SIZE, input_shape))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.summary()

    NUM_EPOCH = 10
    train(model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCH,
        device="cpu")