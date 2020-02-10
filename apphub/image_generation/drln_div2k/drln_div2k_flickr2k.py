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

import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import DRL_Network
from fastestimator.dataset import div2k, flickr2k
from fastestimator.layers.sub_pixel_conv_2d import SubPixelConv2D
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.tensorop import Loss, ModelOp
from fastestimator.op.tensorop.augmentation import Augmentation2D, SRAugmentation2D
from fastestimator.schedule.lr_scheduler import LRSchedule
from fastestimator.trace import LRController, ModelSaver

MAX_RESOLUTION=2040  # max resolution of across div2k and flickr2k images is 2040x2040

class PadUptoTargetShape(NumpyOp):
    """Preprocessing class for padding the data

    Args:
        shape: target shape
    """
    def __init__(self, shape,inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.target_shape = shape

    def forward(self, data, state):
        """Pad data array

        Args:
            data: Data to be padded
            state: A dictionary containing background information such as 'mode'

        Returns:
            Padded array
        """
        input_shape = data.shape
        required_height_padding = self.target_shape[0] - input_shape[0]
        required_weight_padding = self.target_shape[1] - input_shape[1]
        pad_sequence = [(0, required_height_padding),(0, required_weight_padding),(0,0)]
        data_padded = np.pad(data,pad_sequence,mode='constant')
        return data_padded, input_shape

class RandomImagePatches(TensorOp):
    """ RandomImagePatches generates crops. These crops are defined by patch_size
    """
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 lr_patch_size=(48,48), scale=4):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.lr_patch_size = lr_patch_size
        self.scale = scale

    def get_image_patch(self, image_hr, image_lr, lr_shape):
        offset_height_lr = tf.random.uniform(shape=(1, ), minval=0,
                                          maxval= lr_shape[0]- self.lr_patch_size[0]+1, dtype=tf.dtypes.int32)
        offset_width_lr = tf.random.uniform(shape=(1, ), minval=0,
                                         maxval= lr_shape[1]- self.lr_patch_size[1]+1, dtype=tf.dtypes.int32)
        crop_img_lr = image_lr[offset_height_lr[0]:offset_height_lr[0]+ self.lr_patch_size[0],
                               offset_width_lr[0]:offset_width_lr[0]+ self.lr_patch_size[1]]

        offset_height_hr = offset_height_lr[0]* self.scale
        offset_width_hr =  offset_width_lr[0]* self.scale
        hr_patch_size = (self.lr_patch_size[0]*self.scale, self.lr_patch_size[1]*self.scale)
        crop_img_hr = image_hr[offset_height_hr:offset_height_hr + hr_patch_size[0],
                               offset_width_hr:offset_width_hr + hr_patch_size[1]]
        crop_img_hr, crop_img_lr = tf.cast(crop_img_hr, tf.float32), tf.cast(crop_img_lr, tf.float32)

        return crop_img_hr, crop_img_lr

    def forward(self, data, state):
        image_hr, hr_shape, image_lr, lr_shape = data
        image_hr = image_hr[0:hr_shape[0],0:hr_shape[1],:]
        image_lr = image_lr[0:lr_shape[0],0:lr_shape[1],:]
        crop_img_hr, crop_img_lr = self.get_image_patch(image_hr,image_lr, lr_shape)
        return crop_img_hr , crop_img_lr


class ContentLoss(Loss):
    """Compute generator loss."""
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.mae_loss =  tf.keras.losses.MeanAbsoluteError(reduction='none')

    def forward(self, data, state):
        superres, highres = data
        batch_size, _, _, _ = superres.shape
        superres = tf.reshape(superres, (batch_size, -1))
        highres = tf.reshape(highres, (batch_size, -1))
        mae_loss = self.mae_loss(highres, superres)
        return mae_loss


class MyLRSchedule(LRSchedule):
    """ lrschedule to modify lr for edsr.  """
    def __init__(self, schedule_mode):
        super().__init__(schedule_mode)
        self.current = 0
    def schedule_fn(self, current_step_or_epoch, lr):
        next = current_step_or_epoch//200000
        if self.current != next:
            lr = lr/2
            self.current = next
        return lr


def get_estimator(batch_size=16,
                  epochs=1,
                  steps_per_epoch=215,
                  model_dir=tempfile.mkdtemp(),
                  lr_scale=4,
                  path_div2k_flickr2k=os.path.join(os.getenv('HOME'), 'fastestimator_data', 'DRLN')):
    """Args:
    """
    print('model_dir:', model_dir)
    assert path_div2k_flickr2k is not None, 'Pass valid folder path having div2k and flickr2k dataset'
    train_flickr2k_csv, path_div2k_flickr2k = flickr2k.load_data( path_div2k_flickr2k)
    df_flickr2k = pd.read_csv(train_flickr2k_csv)
    train_div2k_csv, val_div2k_csv,  path_div2k_flickr2k = div2k.load_data( path_div2k_flickr2k, lr_scale= lr_scale)
    df_div2k  = pd.read_csv(train_div2k_csv)

    df = pd.concat([df_flickr2k, df_div2k])
    df = df.sample(frac=1).reset_index(drop=True)

    train_div2k_flickr2k_csv = os.path.join(path_div2k_flickr2k,'div2k_flickr2k_trian.csv')
    df.to_csv(train_div2k_flickr2k_csv,index=False)

    writer = fe.RecordWriter(
    save_dir=os.path.join(path_div2k_flickr2k, "tfrecords"),
    train_data=train_div2k_flickr2k_csv,
    validation_data=val_div2k_csv,
    ops=[ImageReader(inputs="image_hr", outputs="image_hr"),
            ImageReader(inputs="image_lr", outputs="image_lr"),
            PadUptoTargetShape((MAX_RESOLUTION//lr_scale, MAX_RESOLUTION//lr_scale),inputs='image_lr',outputs=['image_lr','lr_shape']),
            PadUptoTargetShape((MAX_RESOLUTION, MAX_RESOLUTION),inputs='image_hr',outputs=['image_hr','hr_shape'])
            ],
    compression="GZIP",
    write_feature=['image_hr', 'hr_shape', 'image_lr', 'lr_shape'])

    pipeline = fe.Pipeline(
    batch_size=batch_size,
    data=writer,
    ops=[  RandomImagePatches(inputs=['image_hr','hr_shape','image_lr','lr_shape'], outputs=['image_hr','image_lr']),
         SRAugmentation2D(inputs=['image_hr','image_lr'], outputs=['image_hr', 'image_lr'], mode='train')])


    model_drln = fe.build(model_def= lambda: DRL_Network(input_shape=(48,48,3)),
                        model_name="drln",
                        optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                        loss_name="mae_loss")

    network = fe.Network(ops=[
            ModelOp(inputs='image_lr', model=model_drln, outputs='image_sr'),
            ContentLoss(inputs=("image_sr", "image_hr"), outputs=("mae_loss")),
        ])

    estimator = fe.Estimator(
    network=network,
    pipeline=pipeline,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    traces=[
        ModelSaver(model_name="drln", save_dir=model_dir, save_best=True),
        LRController(model_name="drln", lr_schedule=MyLRSchedule(schedule_mode='step'))
    ])

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
