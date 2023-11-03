import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers


def train_model(model, training_data, x_val, y_val, loss, model_weights_name, epochs, batch_size, optimizer, metrics=['mse']):
    """
    Train a neural network model.

    Args:
        model (tf.keras.Model): The neural network model to train.
        training_data (tuple): Tuple of training data (x_train, y_train).
        x_val (ndarray): Validation data features.
        y_val (ndarray): Validation data labels.
        loss (str or callable): Loss function for training.
        model_weights_name (str): Name of the model weights file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
        metrics (list): List of metrics to monitor during training.

    Returns:
        tuple: A tuple containing the training history and the trained model.

    """
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_weights_name,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=0,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.00001
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )

    history = model.fit(training_data[0], training_data[1], validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                       callbacks=[reduce_lr, model_checkpoint_callback, early_stopping], shuffle=True)
    return history, model



def complex_conv(layer_filters=[16, 64, 128], bn=True, padding='same', kernel_size=5,
                pooling=True, dense_layers=[256], dense_activation='relu', input_shape=(141, 161, 7),
                dropout=0.65, activation='selu', output_shape=3123):
    """
    Create a complex convolutional neural network model.

    Args:
        layer_filters (list): List of layer filter sizes.
        bn (bool): Whether to use batch normalization.
        padding (str): Padding type.
        kernel_size (int): Convolutional kernel size.
        pooling (bool): Whether to use pooling layers.
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.
        activation (str): Activation function for convolutional layers.
        output_shape (int): Output shape.

    Returns:
        tf.keras.Model: The constructed convolutional neural network model.

    """
    inputs1 = tf.keras.layers.Input(shape=input_shape)
    x = contruct_base_conv(inputs1, layer_filters=layer_filters, bn=bn, padding=padding,
                         kernel_size=kernel_size, pooling=pooling, activation=activation, dropout=dropout)
    for neuron in dense_layers:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)

    output1 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
    output2 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
    output3 = tf.keras.layers.Dense(output_shape, activation='sigmoid', kernel_initializer='zeros')(x)

    output1 = reshape_output(output_shape, output1)
    output2 = reshape_output(output_shape, output2)
    output3 = reshape_output(output_shape, output3)

    concat = tf.keras.layers.Concatenate(axis=-2)([output1, output2, output3])
    model_transfer_train = tf.keras.models.Model(inputs1, concat)
    model_transfer_train.compile(loss=['mse'], optimizer='adam')
    return model_transfer_train

def contruct_base_conv(x, layer_filters=[50, 32, 16], bn=True, padding='same', kernel_size=3, pooling=True,
                        activation='relu', dropout=0.5, strides=1):
    """
    Construct base convolutional layers for the model.

    Args:
        x (tf.Tensor): Input tensor.
        layer_filters (list): List of layer filter sizes.
        bn (bool): Whether to use batch normalization.
        padding (str): Padding type.
        kernel_size (int): Convolutional kernel size.
        pooling (bool): Whether to use pooling layers.
        activation (str): Activation function for convolutional layers.
        dropout (float): Dropout rate.
        strides (int): Convolutional layer stride.

    Returns:
        tf.Tensor: Output tensor.

    """
    for layer_filt in layer_filters:
        x = conv_layer(x, n_filters=layer_filt, bn=bn, padding=padding, kernel_size=kernel_size, pooling=pooling,
                       activation=activation, strides=strides)
    flatten = tf.keras.layers.Flatten()(x)
    if dropout > 0.0:
        flatten = tf.keras.layers.Dropout(dropout)(flatten)
    return flatten



def conv_layer(x, n_filters=32, activation='relu', padding='same', kernel_size=(2, 3, 3),
              pooling=True, bn=True, strides=1):
    """
    Create a convolutional layer.

    Args:
        x (tf.Tensor): Input tensor.
        n_filters (int): Number of filters for the convolutional layer.
        activation (str): Activation function for the layer.
        padding (str): Padding type.
        kernel_size (int or tuple): Convolutional kernel size.
        pooling (bool): Whether to use pooling.
        bn (bool): Whether to use batch normalization.
        strides (int): Stride for the convolutional layer.

    Returns:
        tf.Tensor: Output tensor.

    """
    x = tf.keras.layers.Conv2D(filters=n_filters, activation=activation, padding=padding, kernel_size=kernel_size, strides=strides)(x)
    if pooling:
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    return x

def simple_conv(layer_filters=[50, 32, 16], bn=True, padding='same', kernel_size=3,
                pooling=True, dense_layers=[256, 11491], dense_activation='relu', input_shape=(60, 100, 10),
                dropout=0.6, activation='relu'):
    """
    Create a simple convolutional neural network model.

    Args:
        layer_filters (list): List of layer filter sizes.
        bn (bool): Whether to use batch normalization.
        padding (str): Padding type.
        kernel_size (int): Convolutional kernel size.
        pooling (bool): Whether to use pooling layers.
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.
        activation (str): Activation function for convolutional layers.

    Returns:
        tf.keras.Model: The constructed convolutional neural network model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    flatten = contruct_base_conv(inputs, layer_filters=layer_filters,
                                 bn=bn, padding=padding,
                                 kernel_size=kernel_size, pooling=pooling,
                                 activation=activation, dropout=dropout)
    x = tf.keras.layers.Dense(dense_layers[0], activation=dense_activation)(flatten)
    for neuron in dense_layers[1:]:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
    model_transfer_train = tf.keras.models.Model(inputs, x)
    return model_transfer_train

def predict(model, x_test, y_test, batch_size=32, key='Rain_bc', pred_name='simple_dense', loss='gamma', thres=0.95):
    """
    Make predictions using a trained model.

    Args:
        model (tf.keras.Model): The trained model for prediction.
        x_test (ndarray): Test data features.
        y_test (ndarray): Test data labels.
        batch_size (int): Batch size for prediction.
        key (str): Key for the output variable in the resulting dataset.
        pred_name (str): Name for the predicted variable in the dataset.
        loss (str): Type of loss function used in the model.
        thres (float): Threshold for rainfall occurrence.

    Returns:
        xarray.Dataset: A dataset containing the predicted variable.

    """
    data = y_test.to_dataset()
    preds = model.predict(x_test, verbose=1, batch_size=batch_size)
    if loss == "gamma":
        scale = np.exp(preds[:, 0])
        shape = np.exp(preds[:, 1])
        prob = preds[:, -1]
        rainfall = (prob > thres) * scale * shape
    else:
        rainfall = preds
    data[key].values = rainfall
    return data.rename({key: pred_name})

def input_dense(x, dropout=0.5):
    """
    Apply a dense layer to the input.

    Args:
        x (tf.Tensor): Input tensor.
        dropout (float): Dropout rate.

    Returns:
        tf.Tensor: Output tensor.

    """
    flatten = tf.keras.layers.Flatten()(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(flatten)
    else:
        x = flatten
    return x

def reshape_output(output_shape, x):
    """
    Reshape the output tensor.

    Args:
        output_shape (int): Desired output shape.
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Reshaped output tensor.

    """
    return tf.keras.layers.Reshape((1, output_shape))(x)

def simple_dense(dense_layers=[256, 11491], dense_activation='relu', input_shape=(60, 100, 10), dropout=0.6):
    """
    Create a simple dense neural network model.

    Args:
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.

    Returns:
        tf.keras.Model: The constructed dense neural network model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = input_dense(inputs, dropout=dropout)
    for neuron in dense_layers:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
    model_transfer_train = tf.keras.models.Model(inputs, x)
    return model_transfer_train

def linear_complex_model(dense_layers=[256], dense_activation='selu', input_shape=(60, 100, 10), dropout=0.6,
                         output_shape=11491):
    """
    Create a linear complex model with dense layers.

    Args:
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.
        output_shape (int): Output shape.

    Returns:
        tf.keras.Model: The constructed linear complex model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = input_dense(inputs, dropout=dropout)
    for neuron in dense_layers:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)

    output1 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
    output2 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
    output3 = tf.keras.layers.Dense(output_shape, activation='sigmoid', kernel_initializer='zeros')(x)

    output1 = reshape_output(output_shape, output1)
    output2 = reshape_output(output_shape, output2)
    output3 = reshape_output(output_shape, output3)

    concat = tf.keras.layers.Concatenate(axis=-2)([output1, output2, output3])
    model_transfer_train = tf.keras.models.Model(inputs, concat)
    model_transfer_train.compile(loss=['mse'], optimizer='adam')
    return model_transfer_train