# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, LeakyReLU, ReLU, Reshape, UpSampling3D, concatenate
from tensorflow.python.keras.models import Model

from fastestimator.layers.sub_pixel_conv_2d import SubPixelConv2D


# Enhanced Deep Residual Networks for Single Image Super-Resolution
def res_block(layer, filters=64, res_scale=1):
    temp = layer
    layer = Conv2D(filters, 3, padding='same')(layer)
    layer = Activation('relu')(layer)
    layer = Conv2D(filters, 3, padding='same')(layer)
    layer = layer * res_scale
    layer = temp + layer
    return layer


def upsample_block(layer, filters, scale):
    for _ in range(int(np.math.log(scale, 2))):
        layer = Conv2D(filters * 4, 3, padding='same')(layer)
        layer = SubPixelConv2D(upsample_factor=2, nchannels=filters)(layer)
    return layer


def EDSR_Network(input_shape=(32, 32, 3), filters=64, res_blocks=32, scale=4):
    inputs = Input(input_shape)

    bias_neg = tf.constant_initializer(np.array([-0.4488, -0.4371, -0.4040]))
    bias_pos = tf.constant_initializer(np.array([0.4488, 0.4371, 0.4040]))
    kernel_init = tf.constant_initializer(np.array([[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]]))
    meanshifted_inputs = Conv2D(3, 1, bias_initializer=bias_neg, kernel_initializer=kernel_init,
                                trainable=False)(inputs)

    head = Conv2D(filters, 3, padding='same')(meanshifted_inputs)
    x = head
    #===============================================================

    for i in range(res_blocks):
        x = res_block(x, filters=filters, res_scale=0.1)

    x = Conv2D(filters, 3, padding='same')(x)
    x = x + head

    x = upsample_block(x, filters, scale)
    final_out = Conv2D(3, 3, padding='same')(x)
    #  reverse operation of mean shift
    final_out = Conv2D(3, 1, bias_initializer=bias_pos, kernel_initializer=kernel_init, trainable=False)(final_out)
    return Model(inputs=inputs, outputs=final_out)