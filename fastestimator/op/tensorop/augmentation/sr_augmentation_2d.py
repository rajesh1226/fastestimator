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
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from fastestimator.op import TensorOp
from fastestimator.util.util import to_list


class SRAugmentation2D(TensorOp):
    """ This class supports augmentation such as  random rotation and flip. These are needed in super resolution task.

    Args:
        rotation_range: can be one of the following, valid value in [0.0, 360.0)
            * Float (x) that represents the range of random rotation (in degrees) from -x to x.
            * Tuple of floats ([x1, x2]) that represents  the range of random rotation(in degrees) between x1 and x2.
        flip_left_right: Boolean representing whether to flip the image horizontally with a probability of 0.5.
        mode: Augmentation on 'training' data or 'evaluation' data.
   """
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None):
        super().__init__(inputs, outputs, mode)
        self.thres = tf.constant(0.5,shape=(), dtype=tf.float32)


    def forward(self, data, state):
        """Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed
            state: Information about the current execution context

        Returns:
            Transformed (augmented) data

        """
        # ensure the data is list in order to prevent syntax error at 322
        if not isinstance(data, list):
            if isinstance(data, tuple):
                data = list(data)
            else:
                data = [data]
        hflip = tf.random.uniform([],maxval=1,minval=0)
        vflip = tf.random.uniform([],maxval=1,minval=0)
        rot90 = tf.random.uniform([],maxval=1,minval=0)


        for idx, single_data in enumerate(data):
            augment_data = single_data
            augment_data = tf.cond( tf.less(hflip, self.thres),
                                   lambda: tf.image.flip_left_right(augment_data),
                                   lambda: augment_data)
            augment_data = tf.cond( tf.less(vflip, self.thres),
                                   lambda: tf.image.flip_up_down(augment_data),
                                   lambda: augment_data)
            augment_data = tf.cond( tf.less(vflip, self.thres),
                                   lambda: tfa.image.rotate(augment_data, (np.pi*90)/180, interpolation='NEAREST'),
                                   lambda: augment_data)
            data[idx] = augment_data
        if not isinstance(self.inputs, (list, tuple)):
            data = data[0]
        return data
