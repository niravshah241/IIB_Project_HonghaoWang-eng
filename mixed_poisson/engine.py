from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import tensorflow as tf

print("successful")
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
          device: str) -> Dict[str, List[float]]:
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
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
    """
    result_dict = {"train_loss": [],
                   "test_loss": []}

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

    return result_dict
