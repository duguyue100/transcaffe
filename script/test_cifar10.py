"""Try to load CIFAR-10 network in network model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
from transcaffe import parsing

data_path = os.environ["TRANSCAFFE_DATA"]

model_str = os.path.join(data_path, "cifar10_nin.prototxt")
model_bin = os.path.join(data_path, "cifar10_nin.caffemodel")

net_param = parsing.parse_protobuf(model_str)
layers, version = parsing.get_layers(net_param)
input_dim = parsing.get_input_size(net_param)
network = parsing.get_network(layers, 0)
print network
# model = parsing.get_model(layers, 1, tuple(input_dim[1:]), net_param.name)

# model = tc.load(model_str, model_bin, target_lib="keras")
