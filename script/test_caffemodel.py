"""Loading a .caffemodel and figure out the encoding.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import struct
from google.protobuf.text_format import Merge

from transcaffe import caffe_pb2, parsing

# define model for testing
data_path = os.environ["TRANSCAFFE_DATA"]

model_str = os.path.join(data_path, "cifar10_nin.prototxt")
model_bin = os.path.join(data_path, "cifar10_nin.caffemodel")

f = open(model_str, mode="rb")
bin_content = f.read()

net = caffe_pb2.NetParameter()
Merge(bin_content, net)
print dir(net)

print net.layer
