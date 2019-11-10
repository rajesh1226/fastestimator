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
import datetime
import json
import time
import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from fastestimator.trace import Trace
import pycocotools.mask as maskUtils


class MeanAveragePrecision(Trace):
    """Calculates mean average precision for object detection task and report it back to logger.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, selected_indices_key, valid_output_key, image_id_key, pred_cls_key, abs_loc_key, padimg_targimg_ratio_key, padding_key,
                    mode="eval", output_name="meanavgprecision", annFile=None, val_csv=None):
        super().__init__(outputs=output_name, mode=mode)
        self.selected_indices_key = selected_indices_key
        self.valid_output_key = valid_output_key
        self.image_id_key = image_id_key
        self.pred_cls_key = pred_cls_key
        self.abs_loc_key = abs_loc_key
        self.padimg_targimg_ratio_key = padimg_targimg_ratio_key
        self.padding_key = padding_key
        self.results = []
        assert annFile != None
        assert val_csv  != None
        self.set_name = 'val'
        self.coco=COCO(annFile)

        df = pd.read_csv(val_csv)
        self.val_imgIds = []


        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories.sort(key=lambda x: x['id'])
        #self.classes = {}
        self.coco_labels = {0:2, 1:3, 2:4 ,3:6 ,4:7, 5:8, 6:9, 7:16, 8:17, 9:18, 10:19, 11:20 }
        #for c in self.categories:
        #    self.coco_labels[len(self.classes)] = c['id']
        #    self.classes[c['name']] = len(self.classes)
        self.coco_labels[12]=21  # index 80 would be for background. ideally we shouldn't be needed key 80 if things are right in nms

    def on_epoch_begin(self, state):
        self.results = []
        self.val_imgIds = []

    def on_batch_end(self, state):

        batch_size = len(state["batch"][self.image_id_key])
        self.image_ids = state["batch"][self.image_id_key]
        self.image_ids = self.image_ids.numpy()
        selected_indices = state["batch"][self.selected_indices_key]
        selected_indices = selected_indices.numpy()
        valid_outputs = state["batch"][self.valid_output_key]
        valid_outputs = valid_outputs.numpy()
        pred_cls = state["batch"][self.pred_cls_key]
        pred_cls = pred_cls.numpy()
        abs_loc = state["batch"][self.abs_loc_key]
        abs_loc = abs_loc.numpy()
        padimg_targimg_ratio = state["batch"][self.padimg_targimg_ratio_key]
        padimg_targimg_ratio = padimg_targimg_ratio.numpy()
        padding = state["batch"][self.padding_key]
        padding = padding.numpy()


        for elem_image_id, elem_select_indices, elem_valid_outputs, elem_pred_cls, elem_abs_loc, elem_padding, elem_padimg_targimg_ratio\
            in zip(self.image_ids, selected_indices, valid_outputs, pred_cls, abs_loc, padding, padimg_targimg_ratio):
            valid_indices =  elem_select_indices[:elem_valid_outputs]
            pred_cls = elem_pred_cls[valid_indices]
            abs_loc = elem_abs_loc[valid_indices]

            scores = np.max(pred_cls, axis=-1)
            labels = np.argmax(pred_cls, axis=-1)

            # compute predicted labels and scores

            self.val_imgIds.append(int(elem_image_id))
            if valid_indices.size != 0:
                for box, score, label in zip(abs_loc, scores, labels):

                    box = box * elem_padimg_targimg_ratio - elem_padding  # equivalent to inversing all the preprocessing operation 
                    box[2:] = box[2:]-box[:2]   # x,y,w,h format

                    # append detection for each positively labeled class
                    label = int(self.coco_labels[label])
                    elem_image_id = int(elem_image_id)
                    image_result = {
                        'image_id'    : elem_image_id,
                        'category_id' : label,
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    self.results.append(image_result)




    def on_epoch_end(self, state):

        json.dump(self.results, open('{}_bbox_results.json'.format(self.set_name), 'w'), indent=4)
        # load results in COCO evaluation tool
        coco_true = self.coco
        coco_pred = None
        try:
            coco_pred = self.coco.loadRes('{}_bbox_results.json'.format(self.set_name))
            # run COCO evaluation
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = self.val_imgIds
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            return coco_eval.stats
        except:
            print('No records found to evaluate')

