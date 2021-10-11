import tensorflow as tf
import numpy as np

class VNet(object):
    def __init__(self, num_classes=1,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation_fn=tf.nn.relu,
                 data_format = 'channels_last'):
        self.num_classes = num_classes
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation_fn
        self.data_format = data_format

        if data_format == 'channels_first':
            self.channel_axis = 1
        elif data_format == 'channels_last':
            self.channel_axis = -1

    def prelu(self, x):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

    def xavier_initializer_convolution(self, shape, dist='uniform', lambda_initializer=True):
        """
        Xavier initializer for N-D convolution patches. input_activations = patch_volume * in_channels;
        output_activations = patch_volume * out_channels; Uniform: lim = sqrt(3/(input_activations + output_activations))
        Normal: stddev =  sqrt(6/(input_activations + output_activations))
        :param shape: The shape of the convolution patch i.e. spatial_shape + [input_channels, output_channels]. The order of
        input_channels and output_channels is irrelevant, hence this can be used to initialize deconvolution parameters.
        :param dist: A string either 'uniform' or 'normal' determining the type of distribution
        :param lambda_initializer: Whether to return the initial actual values of the parameters (True) or placeholders that
        are initialized when the session is initiated
        :return: A numpy araray with the initial values for the parameters in the patch
        """
        s = len(shape) - 2
        num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
        if dist == 'uniform':
            lim = np.sqrt(6. / num_activations)
            if lambda_initializer:
                return np.random.uniform(-lim, lim, shape).astype(np.float32)
            else:
                return tf.random_uniform(shape, minval=-lim, maxval=lim)
        if dist == 'normal':
            stddev = np.sqrt(3. / num_activations)
            if lambda_initializer:
                return np.random.normal(0, stddev, shape).astype(np.float32)
            else:
                tf.truncated_normal(shape, mean=0, stddev=stddev)
        raise ValueError('Distribution must be either "uniform" or "normal".')

    def constant_initializer(self, value, shape, lambda_initializer=True):
        if lambda_initializer:
            return np.full(shape, value).astype(np.float32)
        else:
            return tf.constant(value, tf.float32, shape)

    def get_num_channels(self,x):
        input_shape = [int(x) for indx,x in enumerate(x.get_shape()) if indx>0]
        if self.data_format == 'channels_first':
            return input_shape[1]
        if self.data_format == 'channels_last':
            return input_shape[-1]

    def get_spatial_rank(self,x):
        """
        :param x: an input tensor with shape [batch_size, ..., num_channels]
        :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
        """
        return len(x.get_shape()) - 2

    def get_spatial_size(self,x):
        """
        :param x: an input tensor with shape [batch_size, ..., num_channels]
        :return: The spatial shape of x, excluding batch_size and num_channels.
        """
        return x.get_shape()[1:-1]

    def convolution(self, x, filter, padding='SAME', strides=None, dilation_rate=None):
        w = tf.get_variable(name='weights', initializer=self.xavier_initializer_convolution(shape=filter))
        b = tf.get_variable(name='biases', initializer=self.constant_initializer(0, shape=filter[-1]))

        return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b

    def down_convolution(self, x, factor, kernel_size):
        num_channels = self.get_num_channels(x)
        spatial_rank = self.get_spatial_rank(x)
        strides = spatial_rank * [factor]
        filter = kernel_size + [num_channels, num_channels * factor]
        x = self.convolution(x, filter, strides=strides)
        return x

    def convolution_block(self,layer_input, num_convolutions, keep_prob):
        x = layer_input
        n_channels = self.get_num_channels(x)
        spatial_rank = self.get_spatial_rank(x)
        for i in range(num_convolutions):
            with tf.variable_scope('conv_' + str(i+1)):
                filter = [5]*spatial_rank + [n_channels]+ [n_channels]
                x = self.convolution(x, filter)
                if i == num_convolutions - 1:
                    x = x + layer_input
                x = self.activation_fn(x)
                x = tf.nn.dropout(x, keep_prob)
        return x

    def deconvolution(self, x, filter, output_shape, strides, padding='SAME'):
        w = tf.get_variable(name='weights', initializer=self.xavier_initializer_convolution(shape=filter))
        b = tf.get_variable(name='biases', initializer=self.constant_initializer(0, shape=filter[-2]))

        spatial_rank = self.get_spatial_rank(x)
        if spatial_rank == 2:
            return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
        if spatial_rank == 3:
            return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
        raise ValueError('Only 2D and 3D images supported.')

    def up_convolution(self, x, output_shape, factor, kernel_size):
        num_channels = self.get_num_channels(x)
        spatial_rank = self.get_spatial_rank(x)
        strides = [1] + spatial_rank * [factor] + [1]
        filter = kernel_size + [num_channels // factor, num_channels]
        x = self.deconvolution(x, filter, output_shape, strides=strides)
        return x

    def convolution_block_2(self,layer_input, fine_grained_features, num_convolutions, keep_prob):
        n_channels = self.get_num_channels(layer_input)
        spatial_rank = self.get_spatial_rank(layer_input)
        x = tf.concat((layer_input, fine_grained_features), axis=self.channel_axis)
        if num_convolutions == 1:
            with tf.variable_scope('conv_' + str(1)):
                filter = [5]*spatial_rank + [n_channels * 2] + [n_channels]
                x = self.convolution(x, filter)
                x = x + layer_input
                x = self.activation_fn(x)
                x = tf.nn.dropout(x, keep_prob)
            return x
        with tf.variable_scope('conv_' + str(1)):
            filter = [5]*spatial_rank + [n_channels * 2] + [n_channels]
            x = self.convolution(x, filter)
            x = self.activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        for i in range(1, num_convolutions):
            with tf.variable_scope('conv_' + str(i+1)):
                filter = [5]*spatial_rank + [n_channels] + [n_channels]
                x = self.convolution(x, filter)
                if i == num_convolutions - 1:
                    x = x + layer_input
                x = self.activation_fn(x)
                x = tf.nn.dropout(x, keep_prob)
        return x

    def network_fn(self, x, keep_prob):
        input_shape = [int(kk) for indx,kk in enumerate(x.get_shape())]
        input_channels = input_shape[self.channel_axis]
        spatial_rank = self.get_spatial_rank(x)
        with tf.variable_scope('customNetwork/input_layer'):
            if input_channels == 1:
                kernel = [1]+[1]*spatial_rank + [self.num_channels]
                x = tf.tile(x, kernel)
            else:
                kernel = [5]*spatial_rank + [input_channels] + [self.num_channels]
                x = self.activation_fn(self.convolution(x, kernel))

        features = list()
        for l in range(self.num_levels):
            with tf.variable_scope('customNetwork/encoder/level_' + str(l + 1)):
                x = self.convolution_block(x, self.num_convolutions[l], keep_prob)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    kernel =  [2]*spatial_rank
                    x = self.activation_fn(self.down_convolution(x, factor=2, kernel_size=kernel))

        with tf.variable_scope('customNetwork/bottom_level'):
            x = self.convolution_block(x, self.bottom_convolutions, keep_prob)

        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('customNetwork/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    kernel =  [2]*spatial_rank
                    x = self.activation_fn(self.up_convolution(x, tf.shape(f), factor=2, kernel_size=kernel))

                x = self.convolution_block_2(x, f, self.num_convolutions[l], keep_prob)

        with tf.variable_scope('customNetwork/output_layer'):
            kernel =  [1]*spatial_rank + [self.num_channels] + [self.num_classes]
            logits = self.convolution(x, kernel)
        return logits
