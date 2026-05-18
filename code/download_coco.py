import os
import requests
from zipfile import ZipFile
from pycocotools.coco import COCO

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def setup_coco(base_dir='code/data/coco'):
    os.makedirs(base_dir, exist_ok=True)
    
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(base_dir, f"{name}.zip")
        download_file(url, zip_path)
        print(f"Unzipping {name}...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        # os.remove(zip_path) # Uncomment to save space

if __name__ == "__main__":
    setup_coco()
