
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np

# Leveraging Ultralytics conversion/validation ops if needed
from ultralytics.utils.ops import xywh2xyxy

class COCOPoseAndMaskDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=(640, 640)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.coco = COCO(ann_file)
        
        # Filter strictly for persons (category 1)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        
        # Get all images containing persons
        all_img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        
        # Strict pre-filtering: Only keep images that have instances with BOTH masks and keypoints
        self.valid_img_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            has_valid_instance = any(
                ann.get('num_keypoints', 0) > 0 and 
                'segmentation' in ann and 
                ann.get('iscrowd', 0) == 0 
                for ann in anns
            )
            if has_valid_instance:
                self.valid_img_ids.append(img_id)
                
        print(f"Found {len(self.valid_img_ids)} images with both valid keypoints and masks.")

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # 2. Load Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        keypoints_list = []
        masks_list = []
        bboxes_list = []
        
        for ann in anns:
            if ann.get('num_keypoints', 0) > 0 and 'segmentation' in ann and ann.get('iscrowd', 0) == 0:
                # Extract Keypoints: shape (17, 3) -> [x, y, visibility]
                kp = np.array(ann['keypoints']).reshape(-1, 3)
                keypoints_list.append(kp)
                
                # Extract Binary Mask: shape (H, W)
                mask = self.coco.annToMask(ann)
                masks_list.append(mask)
                
                # Bounding Box (YOLO format or raw)
                bboxes_list.append(ann['bbox'])
        
        # 3. Resize Image and Masks to target size (e.g., 640x640)
        # (For a production model, you should replace this with a letterbox pad 
        # using ultralytics.data.augment.LetterBox to preserve aspect ratio)
        image_resized = cv2.resize(image, self.img_size)
        
        final_masks = []
        final_kps = []
        
        for mask, kp in zip(masks_list, keypoints_list):
            # Resize mask (nearest neighbor to keep it binary)
            mask_resized = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            final_masks.append(mask_resized)
            
            # Scale keypoints coordinates to match the new image size
            scale_x = self.img_size[0] / orig_w
            scale_y = self.img_size[1] / orig_h
            
            kp_scaled = kp.copy().astype(np.float32)
            kp_scaled[:, 0] *= scale_x  # Scale X
            kp_scaled[:, 1] *= scale_y  # Scale Y
            final_kps.append(kp_scaled)
            
        # Convert to arrays/tensors
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0 # (C, H, W)
        masks_tensor = torch.from_numpy(np.array(final_masks)).float()                 # (N, H, W)
        kps_tensor = torch.from_numpy(np.array(final_kps)).float()                     # (N, 17, 3)
        
        return image_tensor, masks_tensor, kps_tensor


def custom_collate_fn(batch):
    images = []
    batch_masks = []
    batch_kps = []
    
    for i, (img, masks, kps) in enumerate(batch):
        images.append(img)
        
        # Optional: Append a batch index to targets so your custom loss function 
        # knows which instance belongs to which image in the batch (YOLO style)
        batch_masks.append(masks)
        batch_kps.append(kps)
        
    images = torch.stack(images, dim=0)
    
    # Keeping them as lists of tensors is often easiest for custom dynamic heads,
    # or you can pad them to a maximum number of instances per batch.
    return images, batch_masks, batch_kps

# Verification
dataset = COCOPoseAndMaskDataset(
    img_dir='path/to/coco/val2017', 
    ann_file='path/to/coco/annotations/person_keypoints_val2017.json'
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# Test a single batch loop
for imgs, masks, kps in dataloader:
    print("Images batch shape:", imgs.shape) # e.g., torch.Size([4, 3, 640, 640])
    print("Number of mask tensors in batch:", len(masks)) 
    break


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

# Import the native Ultralytics LetterBox utility
from ultralytics.data.augment import LetterBox

class COCOLetterboxDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=640):
        self.img_dir = img_dir
        self.img_size = img_size
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.valid_img_ids = []
        
        # Filter for images with both keypoints and masks
        all_img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        for img_id in all_img_ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids))
            if any(a.get('num_keypoints', 0) > 0 and 'segmentation' in a and a.get('iscrowd', 0) == 0 for a in anns):
                self.valid_img_ids.append(img_id)
                
        # Initialize the Ultralytics LetterBox transform
        # auto=False ensures it pads to exactly img_size x img_size, not a stride minimum
        self.letterbox = LetterBox(new_shape=(self.img_size, self.img_size), auto=False, scaleFill=False, scaleup=True)

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # 2. Apply LetterBox to Image
        # Ultralytics LetterBox outputs the modified image array
        image_padded = self.letterbox(image=image)
        
        # 3. Calculate dynamic gain and pads applied by LetterBox
        # Figure out the scale multiplier used
        gain = min(self.img_size / orig_h, self.img_size / orig_w)
        
        # Calculate exactly how much padding was added to the margins
        pad_x = (self.img_size - orig_w * gain) / 2
        pad_y = (self.img_size - orig_h * gain) / 2
        
        # 4. Process Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        final_masks = []
        final_kps = []
        
        for ann in anns:
            if ann.get('num_keypoints', 0) > 0 and 'segmentation' in ann and ann.get('iscrowd', 0) == 0:
                
                # --- A. MASKS PROCESSING ---
                # Generate original binary mask (0 and 1)
                mask = self.coco.annToMask(ann)
                
                # Resize mask using the same scaling gain (Nearest Neighbor to prevent float values)
                new_mask_w = int(round(orig_w * gain))
                new_mask_h = int(round(orig_h * gain))
                mask_scaled = cv2.resize(mask, (new_mask_w, new_mask_h), interpolation=cv2.INTER_NEAREST)
                
                # Pad the mask to match the exact dimensions of the letterboxed image
                # Canvas initialization
                mask_padded = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                
                # Drop the scaled mask directly onto the center where the scaled image lives
                start_y = int(round(pad_y))
                start_x = int(round(pad_x))
                mask_padded[start_y:start_y + new_mask_h, start_x:start_x + new_mask_w] = mask_scaled
                final_masks.append(mask_padded)
                
                # --- B. KEYPOINTS PROCESSING ---
                kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32) # (17, 3)
                kp_transformed = kp.copy()
                
                # Apply the scale gain and the padding shift to X and Y coordinates
                kp_transformed[:, 0] = (kp[:, 0] * gain) + pad_x  # Transform X
                kp_transformed[:, 1] = (kp[:, 1] * gain) + pad_y  # Transform Y
                
                # If a keypoint visibility flag is 0, it doesn't exist. Keep its coordinates at 0.
                for i in range(len(kp_transformed)):
                    if kp_transformed[i, 2] == 0:
                        kp_transformed[i, :2] = 0.0
                        
                final_kps.append(kp_transformed)
        
        # 5. Tensor Format Conversions (Ready for custom YOLO Backbone)
        image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).float() / 255.0
        
        # Fallbacks for empty images just in case
        if len(final_masks) > 0:
            masks_tensor = torch.from_numpy(np.array(final_masks)).float()
            kps_tensor = torch.from_numpy(np.array(final_kps)).float()
        else:
            masks_tensor = torch.zeros((0, self.img_size, self.img_size))
            kps_tensor = torch.zeros((0, 17, 3))
            
        return image_tensor, masks_tensor, kps_tensor
