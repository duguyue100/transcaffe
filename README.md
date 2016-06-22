[![Build Status](https://travis-ci.org/duguyue100/transcaffe.svg?branch=master)](https://travis-ci.org/duguyue100/transcaffe)

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

## Todo List

-   [ ] parse the prototxt
-   [ ] parse the .caffemodel
-   [ ] functions for call out parameters
-   [ ] possible drawing functions
-   [ ] reading and writing utility functions

## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com
