[![Build Status](https://travis-ci.org/duguyue100/transcaffe.svg?branch=master)](https://travis-ci.org/duguyue100/transcaffe)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/25e8a4861ed246b69d6576bf74d79221)](https://www.codacy.com/app/duguyue100/transcaffe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=duguyue100/transcaffe&amp;utm_campaign=Badge_Grade)

# TransCaffe

This is a minor task while I work on my other project.
I'm never a fan of using Caffe (although it's pretty fast and seems awesome!),
and I'm also annoyed by the fact they are using `protobuffer` and
there is no obvious way of porting to other languages.
And I hate building Caffe from scratch on my Mac.

This project is to provide a conversion tool for Caffe pre-trained
models. At first stage, I would like to call out all the weights, and at
the second stage, I would like to develop a more obvious network coding
and save it into HDF5 format. On the final stage, I hope that I can create
a model zoo where I deliver this HDF5 format data.

__NOTICE__: Apparently `protobuf` does not support `python 3` that well. I've
tested all `python 3.3, 3.4 and 3.5` and they are all failed. At this point,
please use this package with `Python 2.7`.

__DEVELOPMENT__: Keras obviously provides a much more mature organization on
saving trained models. My further development is inspired by one recent
development by [MarcBS's Caffe2Keras module](https://github.com/MarcBS/keras/tree/master/keras/caffe).
It seems that MarcBS's implementation has some unstable points.
Once I somehow replicated his parser, I would soon publish a set of
tested conversion. And if it's possible, I would like to poring it to other
popular libraries too.

## Todo List

-   [x] parse the prototxt
-   [x] parse the .caffemodel
-   [x] functions for call out parameters
-   [ ] possible drawing functions
-   [x] reading and writing utility functions
-   [ ] check on CIFAR-10 NIN models (why it's not loading correctly)
-   [ ] compare the activation of MNIST with torch model.
-   [ ] try to load with VGG-16, VGG-19 (compare with [this](https://github.com/fchollet/deep-learning-models))
-   [ ] More support layers for SNN (LRN, Batch Normalization, etc)
-   [ ] More support activation functiosn for SNN
-   [ ] Planning on lasagne loading functions
-   [ ] Possible to parse torch model as well?

## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com
