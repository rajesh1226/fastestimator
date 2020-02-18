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
"""Download div2k  dataset """
import os
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(path=None, lr_scale=8):
    """Download the DIV2k dataset to local storage, if not already downloaded. This will generate a div2k.csv
    file, which contains all the path information.

    Args:
        path (str, optional): The path to store the DIV2K data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.
        lr_scale(int): The lr_scale indicate the low resolution scale if 2x,4x or 8x

    Returns:
        tuple: (csv_path, path) tuple, where

        * **csv_path** (str) -- Path to the summary csv file, containing the following columns:

            * image_hr (str): Images high resolution path.
            * image_lr (str): Image low resolution path.

        * **path** (str) -- Path to data directory.

    """

    assert lr_scale == 8 or lr_scale == 4 or lr_scale == 2, 'low resolution should be either 8x,4x or 2x'
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'DIV2K')
    os.makedirs(path, exist_ok=True)

    train_csv_path = os.path.join(path, "div2k_train.csv")
    val_csv_path = os.path.join(path, "div2k_val.csv")

    train_image_hr_compressed_path = os.path.join(path, 'DIV2K_train_HR.zip')
    train_image_hr_extract_folder_path = os.path.join(path, 'DIV2K_train_HR')

    train_image_lr_compressed_path = None
    train_image_lr_extract_folder_path = None

    if lr_scale == 8:
        train_image_lr_compressed_path = os.path.join(path, 'DIV2K_train_LR_x8.zip')
        train_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_train_LR_x8')
    elif lr_scale == 4:
        train_image_lr_compressed_path = os.path.join(path, 'DIV2K_train_LR_bicubic_X4.zip')
        train_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_train_LR_bicubic', 'X4')
    elif lr_scale == 2:
        train_image_lr_compressed_path = os.path.join(path, 'DIV2K_train_LR_bicubic_X2.zip')
        train_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_train_LR_bicubic', 'X2')

    val_image_hr_compressed_path = os.path.join(path, 'DIV2K_valid_HR.zip')
    val_image_hr_extract_folder_path = os.path.join(path, 'DIV2K_valid_HR')

    val_image_lr_compressed_path = None
    val_image_lr_extract_folder_path = None
    if lr_scale == 8:
        val_image_lr_compressed_path = os.path.join(path, 'DIV2K_valid_LR_x8.zip')
        val_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_valid_LR_x8')
    elif lr_scale == 4:
        val_image_lr_compressed_path = os.path.join(path, 'DIV2K_valid_LR_bicubic_X4.zip')
        val_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_valid_LR_bicubic', 'X4')
    elif lr_scale == 2:
        val_image_lr_compressed_path = os.path.join(path, 'DIV2K_valid_LR_bicubic_X2.zip')
        val_image_lr_extract_folder_path = os.path.join(path, 'DIV2K_valid_LR_bicubic', 'X2')

    # download training data
    if not os.path.exists(train_image_hr_compressed_path):
        print("Downloading data to {}".format(path))
        wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', path, bar=bar_custom)
    if not os.path.exists(train_image_lr_compressed_path):
        print("Downloading data to {}".format(path))
        if lr_scale == 8:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip', path, bar=bar_custom)
        elif lr_scale == 4:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip', path, bar=bar_custom)
        elif lr_scale == 2:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip', path, bar=bar_custom)

    # extract training data
    if not os.path.exists(train_image_hr_extract_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(train_image_hr_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)

    if not os.path.exists(train_image_lr_extract_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(train_image_lr_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)

    # download validation data
    if not os.path.exists(val_image_hr_compressed_path):
        print("Downloading data to {}".format(path))
        wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip', path, bar=bar_custom)
    if not os.path.exists(val_image_lr_compressed_path):
        print("Downloading data to {}".format(path))
        if lr_scale == 8:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip', path, bar=bar_custom)
        elif lr_scale == 4:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip', path, bar=bar_custom)
        elif lr_scale == 2:
            wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip', path, bar=bar_custom)

    # extract training data
    if not os.path.exists(val_image_hr_extract_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(val_image_hr_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)

    if not os.path.exists(val_image_lr_extract_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(val_image_lr_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)

    # glob and generate csv
    if not os.path.exists(train_csv_path):
        img_hr_list = glob(os.path.join(train_image_hr_extract_folder_path, '*.png'))
        img_hr_list = sorted(img_hr_list)
        img_lr_list = glob(os.path.join(train_image_lr_extract_folder_path, '*.png'))
        img_lr_list = sorted(img_lr_list)
        df = pd.DataFrame(data={'image_hr': img_hr_list, 'image_lr': img_lr_list})
        df.to_csv(train_csv_path, index=False)

    if not os.path.exists(val_csv_path):
        img_hr_list = glob(os.path.join(val_image_hr_extract_folder_path, '*.png'))
        img_hr_list = sorted(img_hr_list)
        img_lr_list = glob(os.path.join(val_image_lr_extract_folder_path, '*.png'))
        img_lr_list = sorted(img_lr_list)
        df = pd.DataFrame(data={'image_hr': img_hr_list, 'image_lr': img_lr_list})
        df.to_csv(val_csv_path, index=False)

    return train_csv_path, val_csv_path, path
