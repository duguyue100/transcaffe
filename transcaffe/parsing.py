"""Parse CaffeModel.

Helped by caffe2theano, MarcBS's Caffe2Keras module.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat
from transcaffe import caffe_pb2, utils
from google.protobuf.text_format import Merge

v1_map = {0: 'NONE', 1: 'ACCURACY', 2: 'BNLL', 3: 'CONCAT', 4: 'CONVOLUTION',
          5: 'DATA', 6: 'DROPOUT', 7: 'EUCLIDEANLOSS', 8: 'FLATTEN',
          9: 'HDF5DATA', 10: 'HDF5OUTPUT', 11: 'IM2COL', 12: 'IMAGEDATA',
          13: 'INFOGAINLOSS', 14: 'INNERPRODUCT', 15: 'LRN',
          16: 'MULTINOMIALLOGISTICLOSS', 17: 'POOLING', 18: 'RELU',
          19: 'SIGMOID', 20: 'SOFTMAX', 21: 'SOFTMAXWITHLOSS', 22: 'SPLIT',
          23: 'TANH', 24: 'WINDOWDATA', 25: 'ELTWISE', 26: 'POWER',
          27: 'SIGMOIDCROSSENTROPYLOSS', 28: 'HINGELOSS', 29: 'MEMORYDATA',
          30: 'ARGMAX', 31: 'THRESHOLD', 32: 'DUMMY_DATA', 33: 'SLICE',
          34: 'MVN', 35: 'ABSVAL', 36: 'SILENCE', 37: 'CONTRASTIVELOSS',
          38: 'EXP', 39: 'DECONVOLUTION'}


def parse_caffemodel(filename):
    """Parse a given caffemodel.

    Parameters
    ----------
    filename : string
        absolute path of a given .caffemodel

    Returns
    -------
    layers : list
        The list representation of the network
    version : string
        pretrined network version
    """
    utils.file_checker(filename)

    net_param = caffe_pb2.NetParameter()

    f = open(filename, mode="rb")
    contents = f.read()
    f.close()

    net_param.ParseFromString(contents)

    return get_layers(net_param)


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

    Returns
    -------
    net_param : caffe_pb2.NetParameter
        The parsed .prototxt structure.
    """
    utils.file_checker(filename)

    f = open(filename, mode="rb")
    net_param = caffe_pb2.NetParameter()

    net_def = f.read()
    # append quotes around type information if needed.
    # it seems not working because has newer definititon?
    # net_def = f.read().split("\n")
    # for i, line in enumerate(net_def):
    #     l = line.strip().replace(" ", "").split('#')[0]
    #     if len(l) > 6 and l[:5] == 'type:' and l[5] != "\'" and l[5] != '\"':
    #         type_ = l[5:]
    #         net_def[i] = '  type: "' + type_ + '"'
    #
    # net_def = '\n'.join(net_def)
    # Check before Merge? For V1?
    Merge(net_def, net_param)
    f.close()

    return net_param


def get_layers(net_param):
    """Get layers information.

    Parameters
    ----------
    net_param : caffe_pb2.NetParameter
        A pretrined network description.
    Returns
    -------
    layers : list
        description of the layers.
    version : string
        version information of the pretrained model.
    """
    if len(net_param.layers) > 0:
        return net_param.layers[:], "V1"
    elif len(net_param.layer) > 0:
        return net_param.layer[:], "V2"
    else:
        raise Exception("Couldn't find layers!")


def get_layer_type(layer):
    """Get a given layer type.

    Parameters
    ----------
    layer : caffe_pb2.V1LayerParameter
        a given layer in the network

    Returns
    -------
    type : int or string
        type of the layer.
    """
    if type(layer.type) == int:
        return str(v1_map[layer.type]).lower()
    else:
        return str(layer.type).lower()


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
        return []


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


def check_phase(layer, phase):
    """Check if the layer matches with the target phase.

    Parameters
    ----------
    layer : caffe_pb2.V1LayerParameter
        A given layer.
    phase : int
        0 : train
        1 : test
    """
    try:
        return True if layer.include[0].phase == phase else False
    except IndexError:
        return True


def get_network(layers, phase):
    """Get structure of the network.

    Parameters
    ----------
    layers : list
        list of layers parsed from network parameters
    phase : int
        0 : train
        1 : test
    """
    num_layers = len(layers)
    network = OrderedDict()

    for i in xrange(num_layers):
        layer = layers[i]
        if check_phase(layer, phase):
            layer_id = "trans_layer_"+str(i)
            if layer_id not in network:
                network[layer_id] = []
            prev_blobs = map(str, layer.bottom)
            next_blobs = map(str, layer.top)

            for blob in prev_blobs+next_blobs:
                if blob not in network:
                    network[blob] = []

            for blob in prev_blobs:
                network[blob].append(layer_id)

            network[layer_id].extend(next_blobs)

    network = remove_loops(network)
    print (network)
    network = remove_blobs(network)

    return network


def remove_loops(network):
    """Remove potential loops from the network.

    Parameters
    ----------
    network : OrderedDict
        given network dictionary

    new_network : OrderedDict
        a loops free altered network.
    """
    for e in network:
        if e.startswith("trans_layer_"):
            continue

        idx = 0
        while idx < len(network[e]):
            next_e = network[e][idx]

            if e in network[next_e]:
                new_e = e+"_"+str(idx)
                network[e].remove(next_e)
                network[new_e] = network[e]
                network[e] = [next_e]
                network[next_e] = [new_e]

                for n in network[new_e]:
                    if network[n] == [e]:
                        network[n] = [new_e]

                e = new_e
                idx = 0
            else:
                idx += 1

    return network


def remove_blobs(network):
    """Remove blobs from network.

    Parameters
    ----------
    network : OrderedDict
        given network dictionary

    Returns
    -------
    new_network : OrderedDict
        blobs removed network dictionary
    """
    new_network = OrderedDict()

    def get_idx(x): return int(x[12:])
    for e in network:
        if e.startswith("trans_layer_"):
            idx = get_idx(e)
            if idx not in new_network:
                new_network[idx] = []

            for next_e in network[e]:
                next_es = map(get_idx, network[next_e])
                new_network[idx].extend(next_es)

    return new_network
