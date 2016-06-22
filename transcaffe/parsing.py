"""Parse CaffeModel.

Helped by caffe2theano

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from transcaffe import caffe_pb2, utils


def parse_protobuf(filename):
    """Parse a given protobuf file.

    Parameters
    ----------
    filename : string
        absolute path of .prototxt file
    """
    utils.file_checker(filename)
