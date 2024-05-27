import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
import itertools
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, reduced_problem, input_set, output_set,
                 input_scaling_range=None, output_scaling_range=None,
                 input_range=None, output_range=None, verbose=True):
        self.reduced_problem = reduced_problem
        self.verbose = verbose

        # Convert input_set and output_set to TensorFlow tensors
        if isinstance(input_set, np.ndarray):
            if self.verbose:
                print("Converting input set from numpy array to TensorFlow tensor")
            self.input_set = tf.convert_to_tensor(input_set, dtype=tf.float32)
        else:
            self.input_set = input_set

        if isinstance(output_set, np.ndarray):
            if self.verbose:
                print("Converting output set from numpy array to TensorFlow tensor")
            self.output_set = tf.convert_to_tensor(output_set, dtype=tf.float32)
        else:
            self.output_set = output_set

        # Convert scaling ranges to TensorFlow tensors
        self.input_scaling_range = self._convert_to_tensor(input_scaling_range)
        self.output_scaling_range = self._convert_to_tensor(output_scaling_range)
        self.input_range = self._convert_to_tensor(input_range)
        self.output_range = self._convert_to_tensor(output_range)

    def _convert_to_tensor(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            if self.verbose:
                print(f"Converting data {data} to TensorFlow tensor")
            return tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            return data

    def __len__(self):
        return self.input_set.shape[0]

    def __getitem__(self, idx):
        input_data = self.input_set[idx]
        label = self.output_set[idx]
        return self.transform(input_data), self.target_transform(label)

    def input_transform(self, input_data):
        input_data_scaled = (self.input_scaling_range[1] - self.input_scaling_range[0]) * \
                            (input_data - self.input_range[0]) / \
                            (self.input_range[1] - self.input_range[0]) + \
                            self.input_scaling_range[0]
        return input_data_scaled
    
    def output_transform(self, output_data):
        output_data_scaled = (self.output_scaling_range[1] - self.output_scaling_range[0]) * \
                            (output_data - self.output_range[0]) / \
                            (self.output_range[1] - self.output_range[0]) + \
                            self.output_scaling_range[0]
        return output_data_scaled

    def reverse_transform(self, input_data_scaled):
        input_data = (input_data_scaled - self.input_scaling_range[0]) * \
                     (self.input_range[1] - self.input_range[0]) / \
                     (self.input_scaling_range[1] - self.input_scaling_range[0]) + \
                     self.input_range[0]
        return input_data

    def reverse_target_transform(self, output_data_scaled):
        output_data = (output_data_scaled - self.output_scaling_range[0]) * \
                      (self.output_range[1] - self.output_range[0]) / \
                      (self.output_scaling_range[1] - self.output_scaling_range[0]) + \
                      self.output_range[0]
        return output_data
    
"""
def create_Dataloader(input_dataset, output_dataset, batch_size, shuffling = 100):
    input_dataset = Dataset.from_tensor_slices(input_dataset)
    output_dataset = Dataset.from_tensor_slices(output_dataset)

    combined_dataset = Dataset.zip((input_dataset, output_dataset))
    dataloader = combined_dataset.shuffle(shuffling).batch(batch_size)

    return dataloader
"""

def create_Dataloader(input_dataset, output_dataset, train_batch_size = 32, 
                      test_batch_size = 32, train_percentage = 0.8, shuffling = 100):
    print("#"*80)
    print("Creating training and testing dataloaders")
    train_size = int(len(input_dataset) * train_percentage)
    indices = np.random.permutation(len(input_dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Convert into numpy before slicing, tensors cannot be directly sliced
    ### TOASK: maybe there is a faster way for implementation?
    input_dataset_numpy = input_dataset.numpy().copy()
    output_dataset_numpy = output_dataset.numpy().copy()

    # Split the datasets into training and testing datasets
    train_input = Dataset.from_tensor_slices(input_dataset_numpy[train_indices])
    train_output = Dataset.from_tensor_slices(output_dataset_numpy[train_indices])
    test_input = Dataset.from_tensor_slices(input_dataset_numpy[test_indices])
    test_output = Dataset.from_tensor_slices(output_dataset_numpy[test_indices])

    # Create TensorFlow datasets for training and testing
    train_dataset = Dataset.zip((train_input, train_output))
    test_dataset = Dataset.zip((test_input, test_output))
    train_dataloader = train_dataset.shuffle(shuffling).batch(train_batch_size)
    test_dataloader = test_dataset.shuffle(shuffling).batch(test_batch_size)
    print("Train and test Dataloaders have been successfully created")
    return train_dataloader, test_dataloader

def generate_training_set(sample_size = [2, 2, 2, 2, 2]):
    set_1 = np.linspace(5, 20, sample_size[0])
    set_2 = np.linspace(10, 20, sample_size[1])
    set_3 = np.linspace(10, 50, sample_size[2])
    set_4 = np.linspace(0.3, 0.7, sample_size[3])
    set_5 = np.linspace(0.3, 0.7, sample_size[4])
    training_set = np.array(list(itertools.product(set_1,set_2, set_3,
                                                        set_4, set_5)))
    return training_set

if __name__ == "__main__":

    class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim-1]),
                                          np.zeros([1, para_dim-1])))
            self.input_scaling_range = [-1., 1.]
            self.output_range = [0., 0.]
            self.output_scaling_range = [-1., 1.]

        def update_input_range(self, input_data):
            min_values = np.min(input_data, axis=0)
            max_values = np.max(input_data, axis=0)
            result = np.stack([min_values, max_values])
            self.input_range = result
        
        ### IN real mixed-poisson problem, the output range can be directly updated without this function
        def update_output_range(self, output_data):
            min_values = np.min(output_data)
            max_values = np.max(output_data)
            result = np.stack([min_values, max_values])
            self.output_range = result

    input_data = generate_training_set()
    output_data = np.random.uniform(0., 1., [input_data.shape[0], 3])

    reduced_problem = ReducedProblem(input_data.shape[1])
    reduced_problem.update_input_range(input_data)
    ### In real mixed poisson problem, we diretly update the output range by choosing min and max values
    reduced_problem.update_output_range(output_data)
    print("OUTPUT RANGE BASED ON OUTPUT DATA:", reduced_problem.output_range)
    print("INPUT RANGE BASED ON INPUT DATA:", reduced_problem.input_range)
    customDataset = CustomDataset(reduced_problem, input_data, output_data,
                                  input_scaling_range = reduced_problem.input_scaling_range,
                                  output_scaling_range = reduced_problem.output_scaling_range,
                                  input_range = reduced_problem.input_range,
                                  output_range = reduced_problem.output_range,
                                  verbose = False
                                  )

    print("OUTPUT SCALING RANGE:", customDataset.output_scaling_range)
    print("INPUT SCALING RANGE:", customDataset.input_scaling_range)
    print("CURRENT OUTPUT DATASET:", customDataset.output_set)
    print("CURRENT INPUT DATASET:", customDataset.input_set)

    scaled_outputs = customDataset.output_transform(customDataset.output_set)
    scaled_inputs = customDataset.input_transform(customDataset.input_set)
    print(f"SCALED OUTPUT DATASET: {scaled_outputs}")
    print("SCALED INPUT DATASET:", scaled_inputs)

    ### TEST the reverse_transform functions, make sure the sum of error is zero
    original_inputs = customDataset.reverse_transform(scaled_inputs)
    print("RESCALED INPUTS differnence:", np.sum(np.abs(original_inputs - input_data)))
    original_outputs = customDataset.reverse_target_transform(scaled_outputs)
    print("RESCALED OUTPUS differnence:", np.sum(np.abs(original_outputs - output_data)))

    ### TEST dataloader function for tensorflow
    BATCH_SIZE = 16
    train_dataloader, test_dataloader = create_Dataloader(scaled_inputs, scaled_outputs, 
                                                          train_batch_size=BATCH_SIZE, test_batch_size=1)

    print(f"length of train and test dataloaders: {len(train_dataloader)}, {len(test_dataloader)}")
    for X, y in train_dataloader:
        print(f"Shape of training set: {X.shape}")
        print(f"X dtype: {X.dtype}")
        # print(f"X: {X}")
        print(f"Shape of training set: {y.shape}")
        print(f"y dtype: {y.dtype}")
        # print(f"y: {y}")

    for X, y in test_dataloader:
        print(f"Shape of test set: {X.shape}")
        print(f"X: {X}")
        print(f"Shape of test set: {y.shape}")
        print(f"y: {y}")
