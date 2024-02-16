import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from engine import train

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

if __name__ == "__main__":
    input_shape = 5
    output_shape = 5

    # Create sample input and output data
    num_samples = 300  # 2 batches of size 10

    input_data = np.linspace(0, 1, num_samples * input_shape).reshape((num_samples, input_shape))
    print(input_data.shape)
    input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    output_data = np.sin(2 * np.pi * input_data)
    output_dataset = tf.data.Dataset.from_tensor_slices(output_data)
    combined_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    batch_size = 30
    train_dataloader = combined_dataset.batch(batch_size)
    test_dataloader = train_dataloader

    # Iterate over the DataLoader to check the batches
    for batch_idx, (input_batch, output_batch) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Output batch shape: {output_batch.shape}")
        print()

    # Example usage of the dataloader in a train_step
    model = MixedPoissonANNModel(input_shape, output_shape, num_neurons=25)
    model.build((batch_size, input_shape))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.summary()

    NUM_EPOCH = 5
    train(model,
        train_dataloader=train_dataloader,
        test_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCH,
        device="cpu")