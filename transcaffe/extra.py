"""Extra functions and classes related to the parsing.

This file documents some extra layers and functions that are not
supported by the original libraries by default.
One should also include this module while compiling the pretrained
models.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import theano.tensor as T

from keras.engine import Layer, InputSpec
from keras import initializations
from keras import backend as K


class KerasScale(Layer):
    """Learns a set of weights and biases used for scaling the input data.

    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.


    This class is directly taken from MarcBS's implementation.
    """

    def __init__(self, weights=None, axis=-1, momentum=0.9,
                 beta_init='zero', gamma_init='one', **kwargs):
        """Init a Scale layer.

        Parameters
        ----------
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        """
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(KerasScale, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the scale layer."""
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        """Carry out computation."""
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape)*x + \
            K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        """Get configuration of the layer."""
        config = {"momentum": self.momentum}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KerasLRN2D(Layer):
    """Local Response Normalization.

    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt

    This code is from MarcBS's implementation.
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """Init a LRN layer.

        Parameters
        ----------
        alpha : float
            the scaling parameter
        k : float
            K parameter
        beta : float
            the exponent
        n : int
             the number of channels to sum over (for cross channel LRN) or
             the side length of the square region to sum over (for within
             channel LRN)
        """
        super(KerasLRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def call(self, x, mask=None):
        """Carry out the computation."""
        X = x
        input_dim = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        b, ch, r, c = input_dim
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_output(self, train):
        """Get output."""
        X = self.get_input(train)
        input_dim = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        b, ch, r, c = input_dim
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        """Get configuration of the layer."""
        return {
            "alpha": self.alpha,
            "k": self.k,
            "beta": self.beta,
            "n": self.n}
