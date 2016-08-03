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

from keras.models import Model

from transcaffe import layers as L

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
        # try:
        #     scale = layer.transform_param.scale
        #     scale = 1 if scale <= 0 else scale
        # except AttributeError:
        #     pass
        return []


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


def reverse_net(network):
    """Reverse a network.

    Parameters
    ----------
    network : OrderedDict
        A parsed network

    Returns
    -------
    rev : OrderedDict
        reversed network
    """
    rev = OrderedDict()
    for node in network.keys():
        rev[node] = []
    for node in network.keys():
        for n in network[node]:
            rev[n].append(node)
    return rev


def get_input_layers(network):
    """Get input layers (layers with zero in-order).

    Parameters
    ----------
    network : OrderedDict
        A parsed network

    Returns
    -------
    in_layers : list
        a list of input layers
    """
    return get_output_layers(reverse_net(network))


def get_output_layers(network):
    """Get output layers (layers with zero out-order).

    Parameters
    ----------
    network : OrderedDict
        A parsed network

    Returns
    -------
    out_layers : list
        a list of out layers
    """
    out_layers = []
    for idx in network:
        if network[idx] == []:
            out_layers.append(idx)

    return out_layers


def get_model(layers, phase, input_dim, model_name, lib_type="keras"):
    """Get a model by given network parameters.

    Parameters
    ----------
    layers : list
        network structure by given parsed network.
    phase : int
        0 : train
        1 : test
    input_dim : list
        the input dimension
    model_name : string
        the name of the given model.
    lib_type : string
        currently only Keras is supported.
    """
    network = get_network(layers, phase)
    if len(network) == 0:
        raise Exception("No valid network is parsed!")

    in_layers = get_input_layers(network)
    out_layers = get_output_layers(network)
    rev_network = reverse_net(network)

    def data_layer(x): get_layer_type(x) in ['data', 'imagedata', 'memorydata',
                                             'hdf5data', 'windowdata']
    # remove the link from input to output.
    for in_idx in in_layers:
        for out_idx in out_layers:
            if out_idx in network[in_idx] and data_layer(layers[in_idx]):
                network[in_idx].remove[out_idx]
    net = [None]*(max(network)+1)
    for layer_id in network:
        layer = layers[layer_id]
        layer_name = layer.name
        layer_type = get_layer_type(layer)

        if layer_id in in_layers:
            net[layer_id] = L.input_layer(input_dim, layer_name)
        else:
            layer_in = [None]*(len(rev_network[layer_id]))
            for l in xrange(len(rev_network[layer_id])):
                layer_in[l] = net[rev_network[layer_id][l]]

            if layer_type in ["relu", "sigmoid", "softmax", "softmaxwithloss",
                              "split", "tanh"]:
                net[layer_id] = L.activation(act_type=layer_type,
                                             name=layer_name)(layer_in)
            elif layer_type == "batchnorm":
                epsilon = layer.batchnorm_param.eps
                axis = layer.scale_param.axis
                net[layer_id] = L.batch_norm(epsilon=epsilon, axis=axis,
                                             name=layer_name)(layer_in)
            elif layer_type == "lrn":
                alpha = layer.lrn_param.alpha
                k = layer.lrn_param.k
                beta = layer.lrn_param.beta
                n = layer.lrn_param.local_size

                net[layer_id] = L.lrn(alpha, k, beta, n, layer_name)(layer_in)
            elif layer_type == "scale":
                axis = layer.scale_param.axis

                net[layer_id] = L.scale(axis, layer_name)(layer_in)
            elif layer_type == "dropout":
                prob = layer.dropout_param.dropout_ratio
                net[layer_id] = L.dropout(prob, name=layer_name)(layer_in)
            elif layer_type == "flatten":
                net[layer_id] = L.flatten(name=layer_name)(layer_in)
            elif layer_type == "concat":
                axis = layer.concat_param.axis
                net[layer_id] = L.merge(layer_in, mode='concat',
                                        concat_axis=1, name=layer_name)
            elif layer_type == "eltwise":
                axis = layer.scale_param.axis
                op = layer.eltwise_param.operation

                if op == 0:
                    mode = "mul"
                elif op == 1:
                    mode = "sum"
                elif op == 2:
                    mode == "max"
                else:
                    raise NotImplementedError("Operation is not implemented!")

                net[layer_id] = L.merge(layer_in, mode=mode, concat_axis=axis,
                                        name=layer_name)
            elif layer_type == "innerproduct":
                output_dim = layer.inner_product_param.num_output

                if len(layer_in[0]._keras_shape[1:]) > 1:
                    layer_in = L.flatten(name=layer_name+"_flatten")(layer_in)

                net[layer_id] = L.dense(output_dim, name=layer_name)(layer_in)
            elif layer_type == "convolution":
                has_bias = layer.convolution_param.bias_term
                nb_filter = layer.convolution_param.num_output
                nb_col = (layer.convolution_param.kernel_size or
                          [layer.convolution_param.kernel_h])[0]
                nb_row = (layer.convolution_param.kernel_size or
                          [layer.convolution_param.kernel_w])[0]
                stride_h = (layer.convolution_param.stride or
                            [layer.convolution_param.stride_h])[0] or 1
                stride_w = (layer.convolution_param.stride or
                            [layer.convolution_param.stride_w])[0] or 1
                pad_h = (layer.convolution_param.pad or
                         [layer.convolution_param.pad_h])[0]
                pad_w = (layer.convolution_param.pad or
                         [layer.convolution_param.pad_w])[0]

                if pad_h + pad_w > 0:
                    layer_in = L.zeropadding(padding=(pad_h, pad_w),
                                             name=layer_name)(layer_in)

                net[layer_id] = L.convolution(nb_filter, nb_row, nb_col,
                                              bias=has_bias,
                                              subsample=(stride_h, stride_w),
                                              name=layer_name)(layer_in)
            elif layer_type == "pooling":
                kernel_h = layer.pooling_param.kernel_size or \
                    layer.pooling_param.kernel_h
                kernel_w = layer.pooling_param.kernel_size or \
                    layer.pooling_param.kernel_w

                stride_h = layer.pooling_param.stride or \
                    layer.pooling_param.stride_h or 1
                stride_w = layer.pooling_param.stride or \
                    layer.pooling_param.stride_w or 1

                pad_h = layer.pooling_param.pad or layer.pooling_param.pad_h
                pad_w = layer.pooling_param.pad or layer.pooling_param.pad_w

                if pad_h + pad_w > 0:
                    layer_in = L.zeropadding(padding=(pad_h, pad_w),
                                             name=layer_name)(layer_in)

                net[layer_id] = L.pooling(pool_size=(kernel_h, kernel_w),
                                          strides=(stride_h, stride_w),
                                          pool_type=layer.pooling_param.pool,
                                          name=layer_name)(layer_in)

    in_l = [None]*(len(in_layers))
    out_l = [None]*(len(out_layers))

    for i in xrange(len(in_layers)):
        in_l[i] = net[in_layers[i]]
    for i in xrange(len(out_layers)):
        out_l[i] = net[out_layers[i]]

    return Model(input=in_l, output=out_l, name=model_name)


def get_network_weights(layers, version):
    """Parse network weights.

    Parameters
    ----------
    layers : list
        List of parameter layers from caffemodel
    version : "string"
        "V1" or "V2"

    Return
    ------
    net_weights : OrderedDict
        network's weights
    """
    net_weights = OrderedDict()

    for layer in layers:
        layer_type = get_layer_type(layer)

        if layer_type == "innerproduct":
            blobs = layer.blobs

            if (version == "V1"):
                num_filters = blobs[0].num
                num_channels = blobs[0].channels
                num_col = blobs[0].height
                num_row = blobs[0].width
            elif (version == "V2"):
                if (len(blobs[0].shape.dim) == 4):
                    num_filters = int(blobs[0].shape.dim[0])
                    num_channels = int(blobs[0].shape.dim[1])
                    num_col = int(blobs[0].shape.dim[2])
                    num_row = int(blobs[0].shape.dim[3])
                else:
                    num_filters = 1
                    num_channels = 1
                    num_col = int(blobs[0].shape.dim[0])
                    num_row = int(blobs[0].shape.dim[1])
            else:
                raise Exception("Can't recognize the version %s" % (version))

            W = np.array(blobs[0].data).reshape(num_filters, num_channels,
                                                num_col, num_row)[0, 0, :, :]
            W = W.T
            b = np.array(blobs[1].data)
            layer_weights = [W.astype(dtype=np.float32),
                             b.astype(dtype=np.float32)]

            net_weights[layer.name] = layer_weights
        elif layer_type == "convolution":
            blobs = layer.blobs

            if (version == "V1"):
                num_filters = blobs[0].num
                num_channels = blobs[0].channels
                num_col = blobs[0].height
                num_row = blobs[0].width
            elif (version == "V2"):
                num_filters = int(blobs[0].shape.dim[0])
                num_channels = int(blobs[0].shape.dim[1])
                num_col = int(blobs[0].shape.dim[2])
                num_row = int(blobs[0].shape.dim[3])
            else:
                raise Exception("Can't recognize the version %s" % (version))

            num_group = layer.convolution_param.group
            num_channels *= num_group

            W = np.zeros((num_filters, num_channels, num_col, num_row))

            if layer.convolution_param.bias_term:
                b = np.array(blobs[1].data)
            else:
                b = None

            group_ds = len(blobs[0].data) // num_group
            ncs_group = num_channels // num_group
            nfs_group = num_filters // num_group

            for i in range(num_group):
                group_weights = W[i*nfs_group: (i+1)*nfs_group,
                                  i*ncs_group: (i+1)*ncs_group, :, :]
                group_weights[:] = np.array(
                    blobs[0].data[i*group_ds:
                                  (i+1)*group_ds]).reshape(group_weights.shape)

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] = np.rot90(W[i, j], 2)

            if b is not None:
                layer_weights = [W.astype(dtype=np.float32),
                                 b.astype(dtype=np.float32)]
            else:
                layer_weights = [W.astype(dtype=np.float32)]

            net_weights[layer.name] = layer_weights
        elif layer_type == "batchnorm":
            blobs = layer.blobs

            if (version == "V2"):
                num_kernels = int(blobs[0].shape.dim[0])
            else:
                raise NotImplementedError("Batchnorm is not "
                                          "implemented in %s" % (version))
            W_mean = np.array(blobs[0].data)
            W_std = np.array(blobs[1].data)

            net_weights[layer.name] = [np.ones(num_kernels),
                                       np.zeros(num_kernels),
                                       W_mean.astype(dtype=np.float32),
                                       W_std.astype(dtype=np.float32)]

    return net_weights


def build_model(model, net_weights):
    """Load network's weights to model.

    Parameters
    ----------
    model : keras.models.model
        The model structure of Keras
    net_weights : OrderedDict
        networ's weights
    """
    for layer in model.layers:
        if layer.name in net_weights:
            model.get_layer(layer.name).set_weights(net_weights[layer.name])
