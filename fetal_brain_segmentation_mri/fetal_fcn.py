"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.residual_unit import *
from dltk.core.upsample import *

def upscore_layer_3D(inputs, inputs2, out_filters, in_filters=None, strides=(2, 2, 2), mode=tf.estimator.ModeKeys.EVAL,
                     name='upscore', use_bias=False,
                     kernel_initializer=tf.uniform_unit_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None, bias_regularizer=None):
    """Summary
    
    Args:
        inputs (TYPE): Description
        inputs2 (TYPE): Description
        out_filters (TYPE): Description
        in_filters (None, optional): Description
        strides (tuple, optional): Description
        mode (TYPE, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    # Compute an upsampling shape dynamically from the input tensor. Input filters are required to be static.
    if in_filters is None:
        in_filters = inputs.get_shape().as_list()[-1]
        
    assert len(inputs.get_shape().as_list()) == 5, 'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), 'Ranks of input and input2 differ'
    
    # Account for differences in the number of input and output filters
    if in_filters != out_filters:
        x = tf.layers.conv3d(inputs, out_filters, (1, 1, 1), (1, 1, 1), padding='same', name='filter_conversion', **conv_params)
    else:
        x = inputs
    
    # Upsample inputs
    x = linear_upsample_3D(x, strides)    
        
    # Skip connection
    x2 = tf.layers.conv3d(inputs2, out_filters, (1, 1, 1), (1, 1, 1), padding='same', **conv_params)
    x2 = tf.layers.batch_normalization(x2, training=mode==tf.estimator.ModeKeys.TRAIN)
    
    # Return the element-wise sum
    return tf.add(x, x2)


def fetal_fcn_3D(inputs, num_classes, filters=(32, 32, 32, 32, 128), kernel_sizes=(5, 5, 5, 3, 1),
                     strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1)),
                     mode=tf.estimator.ModeKeys.EVAL, name='fetal_fcn_3D', use_bias=False,
                     kernel_initializer=tf.uniform_unit_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None, bias_regularizer=None):
    """Image segmentation network based on a modified [1] FCN architecture [2]. 

    [1] M. Rajchl et al. Learning under Distributed Weak Supervision. arXiv:1606.01100 2016.
    [2] K. He et al. Deep residual learning for image recognition. CVPR 2016.
    
    Args:
        inputs (TYPE): Description
        num_classes (TYPE): Description
        num_res_units (int, optional): Description
        filters (tuple, optional): Description
        strides (tuple, optional): Description
        mode (TYPE, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, 'inputs are required to have a rank of 5.'
    
    padding='same'

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
                x = pool_op(x, [2*s for s in strides[res_scale]], strides[res_scale], padding)
    
        # Add two blocks of conv/relu units for feature extraction
        with tf.variable_scope('enc_unit_{}'.format(res_scale)):
            for block in range(2):                
                x = tf.layers.conv3d(x, filters[res_scale], [kernel_sizes[res_scale]] * 3, (1, 1, 1), padding, **conv_params)
                x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
                x = relu_op(x)       
                
                # Dropout with 0.5 on the last scale
                if res_scale is len(filters) - 1:
                    with tf.variable_scope('dropout_{}'.format(res_scale)):
                        x = tf.layers.dropout(x)
                
                tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))

            res_scales.append(x)
            
    # Upscore layers [2] reconstruct the predictions to higher resolution scales
    for res_scale in reversed(range(0, len(filters))):
        with tf.variable_scope('upscore_{}'.format(res_scale)):
            x = upscore_layer_3D(x, res_scales[res_scale], num_classes, strides=strides[res_scale], mode=mode,
                                 **conv_params)
        tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):
        x = tf.layers.conv3d(x, num_classes, (1, 1, 1), (1, 1, 1), padding='same', **conv_params)
    tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x
    
    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob
        y_ = tf.argmax(x, axis=-1) if num_classes > 1 else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs