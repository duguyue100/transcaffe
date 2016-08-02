"""Loading a .caffemodel and figure out the encoding.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
from keras.utils.visualize_util import plot

from transcaffe import parsing

# define model for testing
data_path = os.environ["TRANSCAFFE_DATA"]

# model_str = os.path.join(data_path,
#                          "VGG_ILSVRC_16_layers_deploy.prototxt.txt")
model_str = os.path.join(data_path, "lenet.prototxt.txt")
model_bin = os.path.join(data_path, "lenet_iter_10000.caffemodel")

net_param = parsing.parse_protobuf(model_str)
layers, version = parsing.get_layers(net_param)
input_dim = parsing.get_input_size(net_param)
model = parsing.get_model(layers, 1, tuple(input_dim[1:]), net_param.name)
model.summary()
plot(model, to_file='model.png')

param_layers, version = parsing.parse_caffemodel(model_bin)
net_weights = parsing.get_network_weights(param_layers, version)

parsing.build_model(model, net_weights)

print("[MESSAGE] Loaded")

# parsing.build_model(model, net_weights)
#
# network = parsing.get_network(layers, 0)
# # input_dim = parsing.get_input_size(net_param)
#
#
#
# print(type(model))
# print(model.summary())
#
# plot(model, to_file='model.png')
