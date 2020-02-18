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
"""Download flickr2k  dataset """
import os
import tarfile
from glob import glob
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(path=None):
    """Download the Flickr2k dataset to local storage, if not already downloaded. This will generate a flickr2k.csv
    file, which contains all the path information.

    Args:
        path (str, optional): The path to store the Flickr2k data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (csv_path, path) tuple, where

        * **csv_path** (str) -- Path to the summary csv file, containing the following columns:

            * image_hr (str): Images high resolution path.
            * image_lr (str): Image low resolution path.

        * **path** (str) -- Path to data directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'Flickr2K')
    os.makedirs(path, exist_ok=True)

    train_csv_path = os.path.join(path, "flickr2k_train.csv")

    image_compressed_path = os.path.join(path, 'Flickr2K.tar')
    image_extract_folder_path = os.path.join(path, 'Flickr2K')

    # download training data
    if not os.path.exists(image_compressed_path):
        print("Downloading data to {}".format(path))
        wget.download('https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar', path, bar=bar_custom)

    # extract training data
    if not os.path.exists(image_extract_folder_path):
        print("\nExtracting file ...")
        with tarfile.open(image_compressed_path, 'r') as tar_file:
            tar_file.extractall(path)

    # glob and generate csv
    if not os.path.exists(train_csv_path):
        img_hr_list = glob(os.path.join(image_extract_folder_path, 'Flickr2K_HR', '*.png'))
        img_hr_list = sorted(img_hr_list)
        img_lr_list = glob(os.path.join(image_extract_folder_path, 'Flickr2K_LR_bicubic', 'X4', '*.png'))
        img_lr_list = sorted(img_lr_list)
        df = pd.DataFrame(data={'image_hr': img_hr_list, 'image_lr': img_lr_list})
        df.to_csv(train_csv_path, index=False)

    return train_csv_path, path
