# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np


def conv2d_bn_relu(x, filters, training, kernel_size=3, strides=1):
    """ Basic Conv + BN + ReLU unit """
    x_conv = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                              strides=strides,
                              padding='same', use_bias=False)
    x_bn = tf.layers.batch_normalization(x_conv, training=training)
    x_relu = tf.nn.relu(x_bn)
    return x_relu


def residual_unit(x, filters, training, strides=1):
    """
        Basic residual learning unit, which implements the unit illustrated
        by Figure 1(b) in He et al. Identity Mappings in Deep Residual
        Networks, ECCV 2016. https://arxiv.org/pdf/1603.05027
        """
    orig_x = x
    with tf.name_scope('sub1'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3,
                             strides=strides,
                             padding='same', use_bias=False)
    with tf.name_scope('sub2'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3,
                             strides=1,
                             padding='same', use_bias=False)
    # Use projection for increased dimension and identity mapping
    # for the same dimension i.e. option B in
    # He at al. Deep Residual Learning for Image Recognition, CVPR 2016.
    # https://arxiv.org/pdf/1512.03385
    with tf.name_scope('add'):
        if orig_x.shape[3] == x.shape[3] and strides == 1:
            shortcut = orig_x
        else:
            shortcut = tf.layers.conv2d(orig_x, filters=filters,
                                        kernel_size=1, strides=strides,
                                        padding='same', use_bias=False)
        x = shortcut + x
    return x


def bottleneck_unit(x, filters, training, strides=1):
    """
        Bottleneck residual learning unit, which implements the unit illustrated
        on the right of Figure 5 in
          He at al. Deep Residual Learning for Image Recognition, CVPR 2016.
          https://arxiv.org/pdf/1512.03385
        We also move batch_norm and relu before conv, as shown in He's next paper
          He at al. Identity Mappings in Deep Residual Networks, ECCV 2016.
          https://arxiv.org/pdf/1603.05027.
        """
    orig_x = x
    with tf.name_scope('sub1'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters / 4, kernel_size=1,
                             strides=strides,
                             padding='same', use_bias=False)
    with tf.name_scope('sub2'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters / 4, kernel_size=3,
                             strides=1,
                             padding='same', use_bias=False)
    with tf.name_scope('sub3'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=1,
                             strides=1,
                             padding='same', use_bias=False)
    with tf.name_scope('add'):
        if orig_x.shape[3] == x.shape[3] and strides == 1:
            shortcut = orig_x
        else:
            shortcut = tf.layers.conv2d(orig_x, filters=filters,
                                        kernel_size=1, strides=strides,
                                        padding='same', use_bias=False)
        x = shortcut + x
    return x


def linear_1d(sz):
    """ 1D linear interpolation kernel """
    if sz % 2 == 0:
        raise NotImplementedError('`Linear kernel` requires odd filter size.')
    c = int((sz + 1) / 2)
    h = np.array(list(range(1, c + 1)) + list(range(c - 1, 0, -1)), dtype=np.float32)
    h /= float(c)
    return h


def linear_2d(sz):
    """ 2D linear interpolation kernel """
    W = np.ones((sz, sz), dtype=np.float32)
    h = linear_1d(sz)
    for i in range(sz):
        W[i, :] *= h
    for j in range(sz):
        W[:, j] *= h
    return W


def transpose_upsample2d(x, factor, constant=True):
    """ 2D upsampling operator using transposed convolution """
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0],
                             x_shape[1] * factor,
                             x_shape[2] * factor,
                             x.shape[3].value])

    # The bilinear interpolation weight for the upsampling filter
    sz = factor * 2 - 1
    W = linear_2d(sz)
    n = x.shape[3].value
    filt_val = np.zeros((sz, sz, n, n), dtype=np.float32)
    for i in range(n):
        filt_val[:, :, i, i] = W

    # Currently, we simply use the fixed bilinear interpolation weights.
    # However, it is possible to set the filt to a trainable variable.
    if constant:
        filt = tf.constant(filt_val, dtype=tf.float32)
    else:
        filt = tf.Variable(filt_val, dtype=tf.float32)

    # Currently, if output_shape is an unknown shape, conv2d_transpose()
    # will output an unknown shape during graph construction. This will be
    # a problem for the next step tf.concat(), which requires a known shape.
    # A workaround is to reshape this tensor to the expected shape size.
    # Refer to https://github.com/tensorflow/tensorflow/issues/833#issuecomment-278016198
    x_up = tf.nn.conv2d_transpose(x, filter=filt, output_shape=output_shape,
                                  strides=[1, factor, factor, 1],
                                  padding='SAME')
    x_out = tf.reshape(x_up,
                       (x_shape[0],
                        x_shape[1] * factor,
                        x_shape[2] * factor,
                        x.shape[3].value))
    return x_out


def build_FCN(image, n_class, n_level, n_filter, n_block, training, same_dim=32, fc=64):
    """
        Build a fully convolutional network for segmenting an input image
        into n_class classes and return the logits map.
        """
    net = {}
    x = image

    # Learn fine-to-coarse features at each resolution level
    for l in range(0, n_level):
        with tf.name_scope('conv{0}'.format(l)):
            # If this is the first level (l = 0), keep the resolution.
            # Otherwise, convolve with a stride of 2, i.e. downsample
            # by a factor of 2。
            strides = 1 if l == 0 else 2
            # For each resolution level, perform n_block[l] times convolutions
            x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                               kernel_size=3, strides=strides)
            for i in range(1, n_block[l]):
                x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                                   kernel_size=3)
            net['conv{0}'.format(l)] = x

    # Before upsampling back to the original resolution level, map all the
    # feature maps to have same_dim dimensions. Otherwise, the upsampled
    # feature maps will have both a large size (e.g. 192 x 192) and a high
    # dimension (e.g. 256 features), which may exhaust the GPU memory (e.g.
    # 12 GB for Nvidia Titan K80).
    # Exemplar calculation:
    #   batch size 20 x image size 192 x 192 x feature dimension 256 x floating data type 4
    #   = 755 MB for a feature map
    #   Apart from this, there is also associated memory of the same size
    #   used for gradient calculation.
    with tf.name_scope('same_dim'):
        for l in range(0, n_level):
            net['conv{0}_same_dim'.format(l)] = conv2d_bn_relu(net['conv{0}'.format(l)],
                                                               filters=same_dim,
                                                               training=training,
                                                               kernel_size=1)

    # Upsample the feature maps at each resolution level to the original resolution
    with tf.name_scope('up'):
        net['conv0_up'] = net['conv0_same_dim']
        for l in range(1, n_level):
            net['conv{0}_up'.format(l)] = transpose_upsample2d(net['conv{0}_same_dim'.format(l)],
                                                               factor=int(pow(2, l)))

    # Concatenate the multi-level feature maps
    with tf.name_scope('concat'):
        list_up = []
        for l in range(0, n_level):
            list_up += [net['conv{0}_up'.format(l)]]
        net['concat'] = tf.concat(list_up, axis=-1)

    # Perform prediction using the multi-level feature maps
    with tf.name_scope('out'):
        # We only calculate logits, instead of softmax here because the loss
        # function tf.nn.softmax_cross_entropy() accepts the unscaled logits
        # and performs softmax internally for efficiency and numerical stability
        # reasons. Refer to https://github.com/tensorflow/tensorflow/issues/2462
        x = net['concat']
        x = conv2d_bn_relu(x, filters=fc, training=training, kernel_size=1)
        x = conv2d_bn_relu(x, filters=fc, training=training, kernel_size=1)
        logits = tf.layers.conv2d(x, filters=n_class, kernel_size=1, padding='same')
    return logits


def build_ResNet(image, n_class, n_level, n_filter, n_block, training,
                 use_bottleneck=False, same_dim=32, fc=64):
    """
        Build a fully convolutional network with residual learning units
        for segmenting an input image into n_class classes and return the
        logits map.
        """
    if use_bottleneck:
        res_func = bottleneck_unit
    else:
        res_func = residual_unit
    net = {}
    x = image

    # Learn fine-to-coarse features at each resolution level
    # As Figure 1 in
    #   He at al. Deep Residual Learning for Image Recognition, CVPR 2016.
    #   https://arxiv.org/pdf/1512.03385
    # shows, the original residual network for ImageNet classification only
    # starts using the residual units from the third resolution level.
    # We do the same here.
    for l in range(0, 2):
        with tf.name_scope('conv{0}'.format(l)):
            strides = 1 if l == 0 else 2
            x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                               kernel_size=3, strides=strides)
            for i in range(1, n_block[l]):
                x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                                   kernel_size=3)
            net['conv{0}'.format(l)] = x

    for l in range(2, n_level):
        with tf.name_scope('conv{0}'.format(l)):
            x = res_func(x, filters=n_filter[l], training=training, strides=2)
            for i in range(1, n_block[l]):
                x = res_func(x, filters=n_filter[l], training=training)
            net['conv{0}'.format(l)] = x

    # Before upsampling back to the original resolution level, map all the feature maps
    # to have same_dim dimensions.
    with tf.name_scope('same_dim'):
        for l in range(0, n_level):
            net['conv{0}_same_dim'.format(l)] = conv2d_bn_relu(net['conv{0}'.format(l)],
                                                               training=training,
                                                               filters=same_dim,
                                                               kernel_size=1)

    # Upsample the feature maps at each resolution level to the original resolution
    with tf.name_scope('up'):
        net['conv0_up'] = net['conv0_same_dim']
        for l in range(1, n_level):
            net['conv{0}_up'.format(l)] = transpose_upsample2d(net['conv{0}_same_dim'.format(l)],
                                                               factor=int(pow(2, l)))

    # Concatenate the multi-level feature maps
    with tf.name_scope('concat'):
        list_up = []
        for l in range(0, n_level):
            list_up += [net['conv{0}_up'.format(l)]]
        net['concat'] = tf.concat(list_up, axis=-1)

    # Perform prediction using the multi-level feature maps
    with tf.name_scope('out'):
        # We only calculate logits, instead of softmax here because the loss function
        # tf.nn.softmax_cross_entropy() accepts the unscaled logits and performs softmax
        # internally for efficiency and numerical stability reasons.
        # Refer to https://github.com/tensorflow/tensorflow/issues/2462
        x = net['concat']
        x = conv2d_bn_relu(x, filters=fc, training=training, kernel_size=1)
        x = conv2d_bn_relu(x, filters=fc, training=training, kernel_size=1)
        logits = tf.layers.conv2d(x, filters=n_class, kernel_size=1, padding='same')
    return logits
