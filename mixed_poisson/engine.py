from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import tensorflow as tf
import rbnicsx
from tensorflow.keras.callbacks import EarlyStopping

def train_step(model: tf.keras.Model,
               dataloader: tf.data.Dataset,
               loss_fn: tf.keras.losses.Loss,
               optimizer: tf.optimizers.Optimizer,
               device: str) -> Tuple[float, float]:
    """Trains a TensorFlow model for a single epoch.

    Args:
    model: A TensorFlow model to be trained.
    dataloader: A TensorFlow Dataset instance for the model to be trained on.
    loss_fn: A TensorFlow loss function to minimize. 
    optimizer: A TensorFlow optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    train_loss_metric = tf.keras.metrics.Mean()
    # train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for X, y in dataloader:
        X, y = tf.convert_to_tensor(X.numpy()), tf.convert_to_tensor(y.numpy())  # Convert to TensorFlow tensors
        with tf.GradientTape() as tape:
            prediction = model(X, training=True)
            loss = loss_fn(y, prediction)
        # print("BATCH LOSE:", loss)
        # Calculating the gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        # Updating the gradient
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_metric.update_state(loss)
        # train_acc_metric.update_state(y, prediction)

    train_loss = train_loss_metric.result().numpy()
    # train_acc = train_acc_metric.result().numpy()
    train_loss_metric.reset_states()
    # train_acc_metric.reset_states()

    #return train_loss, train_acc
    return train_loss

### TODO: Name it validate step
def test_step(model: tf.keras.Model,
              dataloader: tf.data.Dataset,
              loss_fn: tf.keras.losses.Loss,
              device: str) -> Tuple[float, float]:
    """Tests a TensorFlow model for a single epoch.

    Args:
    model: A TensorFlow model to be tested.
    dataloader: A TensorFlow Dataset instance for the model to be tested on.
    loss_fn: A TensorFlow loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    test_loss_metric = tf.keras.metrics.Mean()
    # test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for X, y in dataloader:
        X, y = tf.convert_to_tensor(X.numpy()), tf.convert_to_tensor(y.numpy())  # Convert to TensorFlow tensors
        prediction = model(X, training=False)
        loss = loss_fn(y, prediction)

        test_loss_metric.update_state(loss)
        # test_acc_metric.update_state(y, prediction)

    test_loss = test_loss_metric.result().numpy()
    # test_acc = test_acc_metric.result().numpy()
    test_loss_metric.reset_states()
    # test_acc_metric.reset_states()

    # return test_loss, test_acc
    return test_loss

def train(model: tf.keras.Model,
          train_dataloader: tf.data.Dataset,
          test_dataloader: tf.data.Dataset,
          optimizer: tf.optimizers.Optimizer,
          loss_fn: tf.keras.losses.Loss,
          epochs: int,
          device: str,
          early_stopping = 5) -> Dict[str, List[float]]:
    """Trains and tests a TensorFlow model.

    Args:
    model: A TensorFlow model to be trained and tested.
    train_dataloader: A TensorFlow Dataset instance for the model to be trained on.
    test_dataloader: A TensorFlow Dataset instance for the model to be tested on.
    optimizer: A TensorFlow optimizer to help minimize the loss function.
    loss_fn: A TensorFlow loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss for each epoch.
    In the form: {train_loss: [...],
                  test_loss: [...]}
    """

    result_dict = {"train_loss": [],
                   "test_loss": []}

    # Define Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping, restore_best_weights=True)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)

        test_loss = test_step(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device)

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | ")

        result_dict["train_loss"].append(train_loss)
        result_dict["test_loss"].append(test_loss)

        # Check for early stopping
        if len(result_dict["test_loss"]) > early_stopping.patience and \
           all(result_dict["test_loss"][-1 - i] > result_dict["test_loss"][-2 - i] for i in range(early_stopping.patience)):
            print("Early stopping triggered!")
            break

    return result_dict


def online_nn(reduced_problem, problem, online_mu, model, rb_size,
              input_scaling_range=None, output_scaling_range=None,
              input_range=None, output_range=None, verbose=False):
    '''
    Online phase
    Inputs:
        online_mu: np.ndarray [1,num_para]
            representing online parameter
        reduced_problem: reduced problem with attributes:
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the
                SCALED INPUT min_values and row 1 are the
                SCALED INPUT max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the
                SCALED OUTPUT min_values and row 1 are the
                SCALED OUTPUT max_values
            input_range: (2,num_para) np.ndarray, row 0 are the
                ACTUAL INPUT min_values and row 1 are the
                ACTUAL INPUT max_values
            output_range: (2,num_para) np.ndarray, row 0 are the
                ACTUAL OUTPUT min_values and row 1 are the
                ACTUAL OUTPUT max_values
    Output:
        solution_reduced: rbnicsx.online.create_vector, online solution
    '''
    if input_scaling_range is None:
        assert hasattr(reduced_problem, "input_scaling_range")
        input_scaling_range = reduced_problem.input_scaling_range
    if output_scaling_range is None:
        assert hasattr(reduced_problem, "output_scaling_range")
        output_scaling_range = reduced_problem.output_scaling_range
    if input_range is None:
        assert hasattr(reduced_problem, "input_range")
        input_range = reduced_problem.input_range
    if output_range is None:
        assert hasattr(reduced_problem, "output_range")
        output_range = reduced_problem.output_range

    online_mu_scaled = \
        (online_mu - input_range[0]) * \
        (input_scaling_range[1] - input_scaling_range[0]) / (input_range[1] -
                                           input_range[0]) + \
        input_scaling_range[0]
    
    online_mu_scaled_tensor = tf.convert_to_tensor(online_mu_scaled, dtype=tf.float32)
    pred_scaled = model(online_mu_scaled_tensor)
    pred_scaled_numpy = pred_scaled.numpy().copy()
    pred = (pred_scaled_numpy - output_scaling_range[0]) * (output_range[1] - output_range[0]) /   \
            (output_scaling_range[1] - output_scaling_range[0]) + output_range[0]

    solution_reduced = rbnicsx.online.create_vector(rb_size)
    solution_reduced.array = pred
    return solution_reduced

### TOASK: Why the fem solutiuon do not need collapse
def error_analysis(reduced_problem, problem, error_analysis_mu, model,
                   rb_size, online_nn, fem_solution, norm_error=None,
                   reconstruct_solution=None, input_scaling_range=None,
                   output_scaling_range=None, input_range=None,
                   output_range=None, index=None, verbose=False):
    '''
    Inputs:
        error_analysis_mu: np.ndarray of size [1,num_para] representing
            parameter set at which error analysis needs to be evaluated
        problem: full order model with method:
            norm_error(fem_solution,ann_reconstructed_solution)
            and methods required for online_nn
            (Used ONLY if norm_error is not specified)
            solve: method to compute full order model solution
        reduced_problem: reduced problem with attributes:
            reconstruct_solution: Reconstruct FEM solution from reduced
                basis solution
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the
                SCALED INPUT min_values and row 1 are the SCALED INPUT
                max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the
                SCALED OUTPUT min_values and row 1 are the SCALED OUTPUT
                max_values
            input_range: (2,num_para) np.ndarray, row 0 are the
                ACTUAL INPUT min_values and row 1 are the ACTUAL INPUT
                max_values
            output_range: (2,num_para) np.ndarray, row 0 are the
                ACTUAL OUTPUT min_values and row 1 are the ACTUAL OUTPUT
                max_values
    Outputs:
        error: float, Error computed with norm_error between FEM
            and RB solution
    '''
    ann_prediction = online_nn(reduced_problem, problem, [error_analysis_mu],
                               model, rb_size, input_scaling_range,
                               output_scaling_range, input_range,
                               output_range)
    if reconstruct_solution is None:
        ann_reconstructed_solution = \
            reduced_problem.reconstruct_solution(ann_prediction)
    else:
        ann_reconstructed_solution = reconstruct_solution(ann_prediction)

    """
    fem_solution = problem.solve(error_analysis_mu)
    if type(fem_solution) == tuple:
        assert index is not None
        fem_solution = fem_solution[index]
    
    """
    
    if norm_error is None:
        error = reduced_problem.norm_error(fem_solution,
                                           ann_reconstructed_solution)
    else:
        error = norm_error(fem_solution, ann_reconstructed_solution)
    return error, ann_reconstructed_solution