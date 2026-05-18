from pycocotools.coco import COCO
import requests
import os
import cv2
import numpy as np

ann_file = '/tmp/coco_ann/annotations/person_keypoints_val2017.json'
coco = COCO(ann_file)

# Get all images containing persons
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)

# Pick 10 images
subset_img_ids = sorted(img_ids)[:10]
imgs = coco.loadImgs(subset_img_ids)

data_dir = 'code/data/coco_subset'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

for img_info in imgs:
    # Download image
    img_url = img_info['coco_url']
    img_data = requests.get(img_url).content
    img_path = os.path.join(data_dir, 'images', img_info['file_name'])
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    # Get annotations
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    # Save annotations (simple format: one .npy file per image with list of dicts)
    # Actually, let's just save them as a single JSON for the subset to keep it simple.
    # Or just write a custom loader that uses the COCO object.
    
print(f"Downloaded 10 images to {data_dir}")
