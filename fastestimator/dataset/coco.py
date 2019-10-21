import os
import wget
import zipfile
from pycocotools.coco import COCO
import pdb
import numpy as np
import pandas as pd

classes = {}
coco_labels = {}
coco_labels_inverse = {}

def coco_labels_consecutive(coco):
    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x['id'])
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

def coco_dict_empty():
    coco_labels.clear()
    coco_labels_inverse.clear()
    classes.clear()

        
def category_2_category_consecutive(item):
    return [coco_labels_inverse[label] for label in item]


def format_bb(item):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for bb in item:
        if bb[2] < 1 or bb[2] < 1:
            continue
        else:
            x1s.append(bb[0])
            y1s.append(bb[1])
            x2s.append(bb[0]+bb[2])
            y2s.append(bb[1]+bb[3])
    return [x1s, y1s, x2s, y2s]
        

def generate_csv(path, annFile, datatype, dest_csv):
    coco=COCO(annFile)
    
    coco_dict_empty()
    coco_labels_consecutive(coco)
    
    image_ids = coco.getImgIds()
    image_paths = [ coco.loadImgs(img_ids)[0]['file_name'] for img_ids in image_ids]
    image_paths = [ os.path.join(path, datatype, img_path) for img_path in image_paths]
    annotations_ids =  [coco.getAnnIds(imgIds=img_ids, iscrowd=False) for img_ids in image_ids]
    annot_len = [len(annot_id) for annot_id in annotations_ids ]
    annot_len = np.array(annot_len)
    annot_filter = annot_len == 0  # filtering out cases where annotation is null
    annotations_details = [[ coco.loadAnns(ann_ids)[0]['bbox'] for ann_ids in ann_img] for ann_img in annotations_ids]
    annotations_details_filter =   [annot for  annot, cond in zip(annotations_details, annot_filter) if cond==False]
    
    category_details = [[ coco.loadAnns(ann_ids)[0]['category_id'] for ann_ids in ann_img] for ann_img in annotations_ids]
    category_details_filter = [label for label,cond in zip(category_details, annot_filter) if cond==False]
    
    category_consecutive_details_filter = [ category_2_category_consecutive(label) for label in category_details_filter]
    
    image_paths_filter = [image for image,cond in zip(image_paths, annot_filter) if cond==False]
    image_ids_filter = [image_id for image_id, cond in zip(image_ids, annot_filter) if cond==False]
    
    annotations_x1y1x2y2 = [ format_bb(item) for item in annotations_details_filter]
    
    x1 = [annotations[0] for annotations in  annotations_x1y1x2y2]
    y1 = [annotations[1] for annotations in  annotations_x1y1x2y2]
    x2 = [annotations[2] for annotations in  annotations_x1y1x2y2]
    y2 = [annotations[3] for annotations in  annotations_x1y1x2y2]
    
    row_list = []
    for img, img_id, x1_item, y1_item, x2_item, y2_item,label_item in \
                                        zip(image_paths_filter, image_ids_filter, x1, y1, x2, y2, category_consecutive_details_filter):
        row_dict = {'image':img, 'image_id': img_id, 'x1':x1_item, 'y1': y1_item, 'x2': x2_item, 'y2': y2_item, 'label': label_item}
        row_list.append(row_dict)
    df = pd.DataFrame(row_list, columns=['image','image_id', 'label', 'x1', 'y1', 'x2', 'y2'])
    df.to_csv(dest_csv, index=False)
    
    
def load_data(path=None):
    """Download the coco dataset to disk if it is not already downloaded. This will generate 2 csv files
    
    Args: 
        path (str, optional): The path to store the svhn data. Defaults to None, will save at `tempfile.gettempdir()`.
    
    Return:
        string: path to train csv file.
        string: path to test csv file.
        """
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "SVHN")
    if not os.path.exists(path):
        os.makedirs(path)
        
    train_csv = os.path.join(path,'train_coco.csv')
    val_csv = os.path.join(path,'val_coco.csv')
        
    train_dir = os.path.join(path,'train2014')
    val_dir = os.path.join(path, 'val2014')
    annotations_dir = os.path.join(path, 'annotations')
        
    if not os.path.exists(os.path.join(path,'train2014.zip')):
        print('Downloading training images')
        wget.download('http://images.cocodataset.org/zips/train2014.zip', path)
    if not os.path.exists(os.path.join(path,'val2014.zip')):
        print('Downloading val images')
        wget.download('http://images.cocodataset.org/zips/val2014.zip', path)
    if not os.path.exists(os.path.join(path,'annotations_trainval2014.zip')):
        print('Downloading annotation info')
        wget.download('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', path)
    
    if not (os.path.exists(train_dir) and len(os.listdir(train_dir))==82783):
        print('Extracting training images')
        with zipfile.ZipFile(os.path.join(path,'train2014.zip'),'r') as zip_file:
            zip_file.extractall(path)
    
    if not (os.path.exists(val_dir) and len(os.listdir(val_dir))==40504):
        print('Extracting validation images')
        with zipfile.ZipFile(os.path.join(path,'val2014.zip'),'r') as zip_file:
            zip_file.extractall(path)
            
    if not (os.path.exists(annotations_dir) and len(os.listdir(annotations_dir))!=0):
        print('Extracting annotations')
        with zipfile.ZipFile(os.path.join(path, 'annotations_trainval2014.zip'),'r') as zip_file:
            zip_file.extractall(path)
            
    # Generating train and val csv files
    train_annFile = os.path.join(annotations_dir,'instances_train2014.json')
    val_annFile = os.path.join(annotations_dir,'instances_val2014.json')
    
    generate_csv(path, train_annFile, 'train2014', train_csv)
    generate_csv(path, val_annFile, 'val2014', val_csv)
            
    return  train_csv, val_csv, path

