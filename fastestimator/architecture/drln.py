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


def dcr_block(layer, filters):  # densely connected residul block
    temp = layer
    layer = Conv2D(filters,3,padding='same',activation='relu')(layer)
    layer = Conv2D(filters,3,padding='same')(layer)
    layer = Add()([layer, temp])
    layer = Activation('relu')(layer)
    return layer

def compression(layer, filters):
    layer = Conv2D(filters, 3, padding='same', activation='relu')(layer)
    return layer

def laplacian_attention(layer, filters, reduction):
    temp = layer
    layer = GlobalAveragePooling2D()(layer)
    layer = Reshape(target_shape=(1,1,filters))(layer)
    kernel_initializer = tf.keras.initializers.glorot_uniform()
    filter_init = kernel_initializer(shape=(3,3,filters, filters//reduction))
    c1 = tf.nn.conv2d(layer,filters=filter_init, strides=1, padding=[[0, 0], [3,3], [3,3], [0, 0]], dilations=3)
    c2 = tf.nn.conv2d(layer, filters=filter_init, strides=1, padding=[[0, 0], [5,5], [5,5], [0, 0]], dilations=5)
    c3 = tf.nn.conv2d(layer, filters=filter_init, strides=1, padding=[[0, 0], [7,7], [7,7], [0, 0]], dilations=7)
    c_out =  concatenate([c1,c2,c3], axis=-1)
    out =  Conv2D(filters,3,padding='same',activation='sigmoid')(c_out)
    return temp * out

def upsample_block(layer, filters, scale):
    for _ in range(int(np.math.log(scale,2))):
        layer = Conv2D(filters*4,3,padding='same')(layer)
        layer = SubPixelConv2D(upsample_factor=2, nchannels=filters)(layer)
    return layer

def drl_module(layer, filters=64):

    reduction = 16

    temp = layer
    layer = dcr_block(layer, filters)
    layer = concatenate([layer, temp], axis=-1)

    temp = layer
    layer = dcr_block(layer, 2*filters)
    layer = concatenate([layer, temp], axis=-1)

    temp = layer
    layer = dcr_block(layer, 4*filters)
    layer = concatenate([layer, temp], axis=-1)

    layer = compression(layer, filters)
    layer = laplacian_attention(layer, filters, reduction)
    return layer

def DRL_Network(input_shape=(32,32,3)):
    inputs = Input(input_shape)
    filters = 64

    bias_neg = tf.constant_initializer(np.array([-117.59686845, -113.37901035, -102.8904039 ]))
    bias_pos = tf.constant_initializer(np.array([117.59686845, 113.37901035, 102.8904039 ]))
    kernel_init = tf.constant_initializer(np.array([[[[1.,0.,0.],[0.,0.,1.],[0.,0.,1.]]]]))
    meanshifted_inputs= Conv2D(3,1, bias_initializer=bias_neg, kernel_initializer=kernel_init, trainable=False)(inputs)

    head = Conv2D(filters,3,padding='same')(meanshifted_inputs)
    x = head
    #===============================================================

    drlm_out = drl_module(head)    #1
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)          # 2
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)          # 3
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a1 = head + x

    #============================================================

    drlm_out = drl_module(a1)           # 4
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)           # 5
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)           # 6
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a2 = a1 + x

    #===================================================================

    drlm_out = drl_module(a2)           # 7
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)           # 8
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)            # 9
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a3 = a2 + x

    #===========================================================================

    drlm_out = drl_module(a3)            # 10
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)              # 11
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                # 12
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a4 = a3 + x

    #=================================================================

    drlm_out = drl_module(a4)                  # 13
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                      # 14
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                    # 15
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)


    drlm_out = drl_module(x)                       # 16
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a5 = a4 + x

    #=========================================================================



    drlm_out = drl_module(a5)                         # 17
    drlm_concat = concatenate([x, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                          # 18
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                           # 19
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    drlm_out = drl_module(x)                           # 20
    drlm_concat = concatenate([temp, drlm_out], axis=3)
    temp = drlm_concat
    x = Conv2D(filters,3,padding='same')(drlm_concat)

    a6 = a5 + x

    #=============================================================================

    out = a6 + head
    up_out = upsample_block(out, filters,scale=4)
    final_out = Conv2D(3,3,padding='same')(up_out)
    #  reverse operation of mean shift
    final_out= Conv2D(3,1, bias_initializer=bias_pos, kernel_initializer=kernel_init, trainable=False)(final_out)
    return Model(inputs=inputs, outputs=final_out)
