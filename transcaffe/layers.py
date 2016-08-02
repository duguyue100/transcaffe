"""Definition of the layers provided by different libraries.

All backends shares same interface. (hopefully)

Following layers not yet supported by Keras:
- LRN2D
- Scale

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
# Keras backend
from keras.layers import Input, merge
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D, Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

LIB_TYPE = "keras"


def activation(act_type, name):
    """Return an activation function.

    Parameters
    ----------
    act_type : string
        activation type, one of:
            "relu", "sigmoid", "softmax", "softmaxwithloss",
            "split", "tanh"
    name : string
        the name of the activation layer.

    Returns
    -------
    activation : keras.layers.core.Activation
        an Activation layer
    """
    if LIB_TYPE == "keras":
        if act_type in ["relu", "sigmoid"]:
            return Activation(activation=act_type, name=name)
        elif act_type in ["softmax", "softmaxwithloss"]:
            return Activation(activation="softmax", name=name)
        elif act_type in ["split", "tanh"]:
            return Activation(activation="tanh", name=name)
        else:
            raise Exception("The activation type is not supported!")


def dropout(prob, name):
    """Get a dropout layer.

    Parameters
    ----------
    prob : float
        the dropout probabily.
    name : string
        the name of the dropout layer

    Returns
    -------
    dropout : keras.layers.core.Dropout
    """
    if LIB_TYPE == "keras":
        return Dropout(p=prob, name=name)


def flatten(name):
    """Get a flatten layer.

    Parameters
    ----------
    name : string
        the name of the flatten layer

    Returns
    -------
    flatten : keras.layers.core.Flatten
    """
    if LIB_TYPE == "keras":
        return Flatten(name=name)


def dense(out_dim, name):
    """Get a dense layer.

    Parameters
    ----------
    out_dim : int
        the output dimension of such layer.
    name : string
        the name of the dense layer.

    Returns
    -------
    dense : keras.layers.core.Dense
    """
    if LIB_TYPE == "keras":
        return Dense(output_dim=out_dim, name=name)


def batch_norm(epsilon, axis, name):
    """Get a Batch Normalization layer.

    Parameters
    ----------
    epsilon : float
        >0, fuzzy parameters
    axis : integer
        identify which axis to normalize
    name : string
        the name of the batch normalization layer.
    """
    if LIB_TYPE == "keras":
        return BatchNormalization(epsilon=epsilon, axis=axis, name=name)


def zeropadding(padding, name):
    """Get a zero-padding layer.

    Parameters
    ----------
    padding : tuple
        (pad_h, pad_w)
    name : string
        the name of the zero-padding layer.

    Returns
    -------
    zeropadding : keras.layers.convolutional.ZeroPadding2D
    """
    if LIB_TYPE == "keras":
        return ZeroPadding2D(padding=padding, name=name+"_zeropadding")


def pooling(pool_size, strides, pool_type, name, border_mode="valid"):
    """Get a pooling layer by pooling type.

    Parameters
    ----------
    pool_size : tuple
        the pooling size (pool_h, pool_w)
    strides : tuple
        (stride_h, stride_w)
    border_mode : string
        border mode, currently just "valid"
    pool_type : int
        0 : max
        1 : avg
    name : string
        the name of the pooling layer

    Returns
    -------
    pooling : keras.layers.pooling.MaxPooling2D
              keras.layers.pooling.AveragePooling2D
    """
    if LIB_TYPE == "keras":
        if pool_type == 0:
            return MaxPooling2D(pool_size=pool_size,
                                strides=strides,
                                border_mode=border_mode, name=name)
        elif pool_type == 1:
            return AveragePooling2D(pool_size=pool_size,
                                    strides=strides,
                                    border_mode=border_mode, name=name)


def convolution(num_filter, num_row, num_col, bias, subsample, name):
    """Get a convolutional layer.

    Parameters
    ----------
    num_filter : int
        number of filters
    num_row : int
        kernel_h
    num_col : int
        kernel_w
    bias : bool
        if has bias
    subsample : tuple
        (stride_h, strid_w)
    name : string
        the name of the convolution layer

    Returns
    -------
    conolution : keras.layers.convolutional.Convolution2D
    """
    if LIB_TYPE == "keras":
        return Convolution2D(nb_filter=num_filter, nb_row=num_row,
                             nb_col=num_col, subsample=subsample, bias=bias,
                             name=name)


def input_layer(shape, name):
    """Get an input layer.

    Parameters
    ----------
    shape : tuple
        the input shape.
    name : string
        the name of the input layer
    """
    if LIB_TYPE == "keras":
        return Input(shape=shape, name=name)
