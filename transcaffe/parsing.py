"""Parse CaffeModel.

Helped by caffe2theano

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function

import numpy as np
from scipy.io import loadmat
from transcaffe import caffe_pb2, utils
from google.protobuf.text_format import Merge


def parse_caffemodel(filename):
    """Parse a given caffemodel.

    Parameters
    ----------
    filename : string
        absolute path of a given .caffemodel

    Returns
    -------
    """
    utils.file_checker(filename)

    net_param = caffe_pb2.NetParameter()

    f = open(filename, mode="rb")
    contents = f.read()
    f.close()

    net_param.ParseFromString(contents)
    layers = get_layers(net_param)

    layers_dict = {}
    for layer in layers:
        layers_dict[layer.name] = layer.blobs


def parse_mean_file(filename, mode="proto"):
    """Parse a mean file by given path.

    TODO: complete more options based on different Caffe Models

    Parameters
    ----------
    filename : string
        absolute path of the mean file
    mode : string
        "proto" for .binaryproto file
        "mat" for MAT binary file

    Returns
    -------
    mean_mat : numpy.ndarray
        an array that contains the mean values
    """
    utils.file_checker(filename)

    if mode == "proto":
        tp = caffe_pb2.TransformationParameter()
        f = open(filename, mode="rb")
        mean_contents = f.read()
        f.close()
        tp.ParseFromString(mean_contents)
        mean_mat = np.array(tp.mean_value).reshape((3,
                                                    tp.crop_size,
                                                    tp.crop_size))
        mean_mat = np.transpose(mean_mat, (1, 2, 0))
    elif mode == "mat":
        # based on VGG's Mat file.
        mean_contents = loadmat(filename)
        mean_mat = mean_contents["image_mean"]
        print(mean_mat.shape)

    return mean_mat


def parse_protobuf(filename):
    """Parse a given protobuf file.

    Parameters
    ----------
    filename : string
        absolute path of .prototxt file
    """
    utils.file_checker(filename)

    f = open(filename, mode="rb")

    net_param = caffe_pb2.NetParameter()

    Merge(f.read(), net_param)

    print(net_param.name)

    print(type(net_param))

    print(dir(net_param))

    f.close()


def get_layers(net_param):
    """Get layers information.

    Parameters
    ----------
    net_param : caffe_pb2.NetParameter

    Returns
    -------
    """
    if len(net_param.layers) > 0:
        return net_param.layers
    elif len(net_param.layer) > 0:
        return net_param.layer
    else:
        raise Exception("Couldn't find layers!")


def get_input_size(net_param):
    """Get input parameters, or guess one at least.

    Parameters
    ----------
    net_param : caffe_pb2.NetParameter
        structure that contains all the network parameters

    Returns
    -------
    in_size : tuple
        tuple that defines the input size
    """
    if len(net_param.input_dim) != 0:
        return net_param.input_dim
    elif len(net_param.input_shape) != 0:
        return net_param.input_shape
    else:
        print("[MESSAGE] Couldn't find Input shape in the Network Parameters."
              "The returned shape is inferenced from the network name")


def get_weights_raw(blobs):
    """Get raw weights data from given layer blobs.

    Parameters
    ----------
    blobs : list
        the blobs that defines weights and bias

    Returns
    -------
    raw_weights : numpy.ndarray
        a vector that contains weights of the layer
    """
    if len(blobs) > 0:
        return np.array(blobs[0].data)
    else:
        return None


def get_weights(blobs, n_filters=None, n_channels=None, height=None,
                width=None, mode="conv"):
    """Get weight matrix from given layer blobs.

    Parameters
    ----------
    blobs : list
        the blobs that defines weights and bias
        mode:
            "conv": convolution layer
                arrange in (num. filters, num. channels, height, width)

    Returns
    -------
    weights : numpy.ndarray
        a vector that contains weights of the layer
    """
    raw_weights = get_weights_raw(blobs)
    if raw_weights is not None:
        if mode == "conv":
            if n_filters is None or n_channels is None \
               or height is None or height is None:
                raise Exception("Lack parameters for defining convolutional "
                                "filters.")

        weights = raw_weights.reshape((n_filters, n_channels,
                                       height, width))

        return weights
    else:
        return raw_weights


def get_bias_raw(blobs):
    """Get raw bias data from given layer blobs.

    Parameters
    ----------
    blobs : list
        the blobs that defines weights and bias

    Returns
    -------
    raw_bias : numpy.ndarray
        a vector that contains bias of the layer
    """
    if len(blobs) > 1:
        return np.array(blobs[1].data)
    else:
        return None


def get_bias(blobs):
    """Get bias data from given layer blobs.

    Parameters
    ----------
    blobs : list
        the blobs that defines weights and bias

    Returns
    -------
    bias : numpy.ndarray
        a vector that contains bias of the layer
    """
    return get_bias_raw(blobs)
