from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.networks.segmentation.fcn import upscore_layer_3d


def fetal_fcn_3d(inputs,
                 num_classes,
                 filters=(32, 32, 32, 32, 128),
                 kernel_sizes=(5, 5, 5, 3, 1),
                 strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1)),
                 mode=tf.estimator.ModeKeys.EVAL,
                 use_bias=False,
                 kernel_initializer=tf.uniform_unit_scaling_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None):
    """Image segmentation network based on a modified [1] FCN architecture [2].

    [1] M. Rajchl et al. Learning under Distributed Weak Supervision. arXiv:1606.01100 2016.
    [2] K. He et al. Deep residual learning for image recognition. CVPR 2016.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        num_res_units (int, optional): Number of residual units at each
            resolution scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings:
            TRAIN, EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.
    Returns:
        dict: dictionary of output tensors
    """

    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    padding = 'same'

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    relu_op = tf.nn.relu6
    pool_op = tf.layers.max_pooling3d

    x = inputs

    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    res_scales = [x]

    for res_scale in range(0, len(filters)):

        # Use max pooling when required
        if np.prod(strides[res_scale]) > 1:

            with tf.variable_scope('pool_{}'.format(res_scale)):
                x = pool_op(
                    inputs=x,
                    pool_size=[2*s for s in strides[res_scale]],
                    strides=strides[res_scale],
                    padding=padding)

        # Add two blocks of conv/relu units for feature extraction
        with tf.variable_scope('enc_unit_{}'.format(res_scale)):
            for block in range(2):
                x = tf.layers.conv3d(
                    inputs=x,
                    filters=filters[res_scale],
                    kernel_size=[kernel_sizes[res_scale]] * 3,
                    strides=(1, 1, 1),
                    padding=padding,
                    **conv_params)

                x = tf.layers.batch_normalization(
                    x, training=mode == tf.estimator.ModeKeys.TRAIN)
                x = relu_op(x)

                # Dropout with 0.5 on the last scale
                if res_scale is len(filters) - 1:
                    with tf.variable_scope('dropout_{}'.format(res_scale)):
                        x = tf.layers.dropout(x)

                tf.logging.info('Encoder at res_scale {} shape: {}'.format(
                    res_scale, x.get_shape()))

            res_scales.append(x)

    # Upscore layers [2] reconstruct the predictions to higher resolution scales
    for res_scale in reversed(range(0, len(filters))):

        with tf.variable_scope('upscore_{}'.format(res_scale)):

            x = upscore_layer_3d(inputs=x,
                                 inputs2=res_scales[res_scale],
                                 out_filters=num_classes,
                                 strides=strides[res_scale],
                                 mode=mode,
                                 **conv_params)
        tf.logging.info('Decoder at res_scale {} shape: {}'.format(
            res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):
        x = tf.layers.conv3d(
            inputs=x,
            filters=num_classes,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            **conv_params)

    tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x

    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob

        y_ = tf.argmax(x, axis=-1) if num_classes > 1 \
            else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs
