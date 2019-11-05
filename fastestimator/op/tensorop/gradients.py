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
from fastestimator.op import TensorOp
from fastestimator.util.util import to_list


class Gradients(TensorOp):
    """This class computes gradients

        Args:
            loss (str): The loss key to compute gradients from
            models (keras.model, list): A list of models to compute gradients against
            keys (str, list): A list of keys corresponding to variables to compute gradients against
            outputs (str, list): A list of output names (model gradients first, then key gradients)
        """
    def __init__(self, loss, models=None, keys=None, outputs=None):
        self.models = to_list(models) if models else []
        inputs = to_list(keys) if keys else []
        outputs = to_list(outputs) if outputs else []

        assert len(outputs) == len(inputs) + len(self.models)
        super().__init__(inputs=[loss] + inputs, outputs=outputs, mode="train")

    def forward(self, data, state):
        loss, *elems = data
        tape = state['tape']
        with tape.stop_recording():
            gradients = tape.gradient(loss, [model.trainable_variables for model in self.models] + elems)
        return gradients
