import tensorflow as tf
import tensorflow.keras.backend as K

def gamma_loss(y_true, y_pred, eps=3e-2):
    """
    Custom loss function for a gamma distribution parameterization.

    Args:
        y_true (tensor): True target values.
        y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
        eps (float): Small constant to prevent numerical instability.

    Returns:
        tensor: The calculated loss value.

    """
    # Extract predicted values
    occurence = y_pred[:, -1]
    shape_param = K.exp(y_pred[:, 0])
    scale_param = K.exp(y_pred[:, 1])
    
    # Convert y_true to a binary indicator for rain (1 if > 0.0, 0 otherwise)
    bool_rain = tf.cast(y_true > 0.0, 'float32')
    eps = tf.cast(eps, 'float32')
    
    # Calculate the gamma loss
    loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) +
             bool_rain * (K.log(occurence + eps) +
                         (shape_param - 1) * K.log(y_true + eps) -
                         shape_param * tf.math.log(scale_param + eps) -
                         tf.math.lgamma(shape_param) -
                         y_true / (scale_param + eps)))
    
    # Calculate the absolute mean of the loss
    output_loss = tf.abs(K.mean(loss1))
    return output_loss

def gamma_mse_metric(y_true, y_pred, thres=0.5):
    """
    Custom metric for mean squared error of gamma distribution parameters.

    Args:
        y_true (tensor): True target values.
        y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
        thres (float): Threshold for rainfall occurrence.

    Returns:
        tensor: The calculated mean squared error.

    """
    # Extract predicted values
    occurence = y_pred[:, -1]
    shape_param = K.exp(y_pred[:, 0])
    scale_param = K.exp(y_pred[:, 1])
    
    # Convert y_true to a binary indicator for rainfall occurrence
    bool_rain = tf.cast(y_true > 0.0, 'float32')
    
    # Calculate the rainfall using the gamma distribution
    rainfall = shape_param * scale_param * tf.cast(occurence > thres, 'float32')
    
    # Calculate mean squared error between predicted and true rainfall
    return tf.keras.losses.mean_squared_error(rainfall, y_true)
