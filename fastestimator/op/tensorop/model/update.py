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
import tensorflow as tf
from tensorflow.python.framework import ops as tfops
from fastestimator.op import TensorOp


class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the model's loss value

    Args:
        model (keras.model): keras model compiled by fe.build
    """
    def __init__(self, model):
        super().__init__(inputs=model.loss_name, outputs=None, mode="train")
        self.model = model
        self.warmed = False

    def forward(self, data, state):
        loss = self._reduce_loss(element_wise_loss=data,
                                 global_batch_size=state["batch_size"],
                                 local_batch_size=state['local_batch_size'],
                                 warmup=state["warmup"])
        if state["warmup"] and not self.warmed:
            self.warmed = True
            with tfops.init_scope():  # pylint: disable=not-context-manager
                _ = self.model.optimizer.iterations
                self.model.optimizer._create_hypers()  # pylint: disable=protected-access
                self.model.optimizer._create_slots(self.model.trainable_variables)  # pylint: disable=protected-access
        else:
            tape = state['tape']
            with tape.stop_recording():
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @staticmethod
    def _reduce_loss(element_wise_loss, global_batch_size, local_batch_size, warmup):
        if warmup:
            assert element_wise_loss.ndim != 0 and element_wise_loss.shape[0] == local_batch_size, \
                "please make sure loss is element-wise loss"
        return tf.reduce_sum(element_wise_loss) / global_batch_size
