import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PoseDetectionModel, mse_loss, nll_loss, invis_loss, mask_loss
import os

from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image

def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = target_size
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, (r, r), (left, top)

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_dir, target_size=(640, 640)):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.target_size = target_size
        self.output_size = (target_size[0] // 4, target_size[1] // 4)
        
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        # Filter to only use images we downloaded (subset of 10)
        self.img_ids = [iid for iid in self.img_ids if os.path.exists(os.path.join(img_dir, self.coco.loadImgs(iid)[0]['file_name']))]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Letterbox image
        img_padded, (ratio_x, ratio_y), (pad_x, pad_y) = letterbox(img, self.target_size)
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        
        # Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        # Person mask (segmentation) at output resolution (1/4)
        mask = np.zeros(self.output_size, dtype=np.float32)
        # Keypoints at output resolution
        kpts_x = np.zeros((17, *self.output_size), dtype=np.float32)
        kpts_y = np.zeros((17, *self.output_size), dtype=np.float32)
        kpts_mask = np.zeros((17, *self.output_size), dtype=np.float32)
        
        # Scale for output grid
        sx = self.output_size[1] / self.target_size[1]
        sy = self.output_size[0] / self.target_size[0]
        
        for ann in anns:
            # Mask
            m = self.coco.annToMask(ann)
            # Letterbox mask as well
            m_padded, _, _ = letterbox(m, self.target_size, color=0)
            m_resized = cv2.resize(m_padded, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
            person_mask_i = m_resized.astype(np.float32)
            mask = np.maximum(mask, person_mask_i)
            
            # Keypoints
            kp = np.array(ann['keypoints']).reshape(-1, 3)
            for k_idx in range(17):
                if kp[k_idx, 2] > 0:
                    kx, ky = kp[k_idx, 0], kp[k_idx, 1]
                    
                    # Map to padded image coordinates
                    kx_p = kx * ratio_x + pad_x
                    ky_p = ky * ratio_y + pad_y
                    
                    # Absolute position in [0, 4] scale
                    kx_scaled = (kx_p / self.target_size[1]) * 4
                    ky_scaled = (ky_p / self.target_size[0]) * 4
                    
                    # Dense supervision: all pixels of this person predict this keypoint
                    kpts_x[k_idx, person_mask_i > 0.5] = kx_scaled
                    kpts_y[k_idx, person_mask_i > 0.5] = ky_scaled
                    kpts_mask[k_idx, person_mask_i > 0.5] = 1.0
        
        return img_tensor, torch.from_numpy(mask).unsqueeze(0), \
               torch.from_numpy(kpts_x), torch.from_numpy(kpts_y), \
               torch.from_numpy(kpts_mask)

def train():
    device = torch.device('cpu') # Forced to CPU for verification
    model = PoseDetectionModel().to(device)
    
    # Dataset and Dataloader
    ann_file = '/tmp/coco_ann/annotations/person_keypoints_val2017.json'
    img_dir = 'code/data/coco_subset/images'
    dataset = CocoDataset(ann_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Phase 1: Freeze backbone
    print("Phase 1: Training heads only...")
    for param in model.full_base.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0)
    
    for epoch in range(200): # Increased epochs
        model.train()
        epoch_loss = 0
        for i, (imgs, masks_gt, kpts_x_gt, kpts_y_gt, kpts_mask_gt) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks_gt = masks_gt.to(device)
            kpts_x_gt = kpts_x_gt.to(device)
            kpts_y_gt = kpts_y_gt.to(device)
            kpts_mask_gt = kpts_mask_gt.to(device)
            
            optimizer.zero_grad()
            pred_mask, pred_x, pred_y, pred_ln_var = model(imgs)
            
            l_mask = mask_loss(pred_mask, masks_gt)
            l_mse = mse_loss(pred_x, pred_y, kpts_x_gt, kpts_y_gt, kpts_mask_gt)
            l_nll = nll_loss(pred_x, pred_y, pred_ln_var, kpts_x_gt, kpts_y_gt, kpts_mask_gt)
            l_inv = invis_loss(pred_ln_var, kpts_mask_gt)
            
            loss = l_mask + (l_mse + l_nll) / 2 + l_inv * 0.1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Phase 2: Fine-tune backbone
    print("Phase 2: Fine-tuning backbone...")
    for param in model.full_base.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    
    for epoch in range(200): # Increased epochs
        model.train()
        epoch_loss = 0
        for i, (imgs, masks_gt, kpts_x_gt, kpts_y_gt, kpts_mask_gt) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks_gt = masks_gt.to(device)
            kpts_x_gt = kpts_x_gt.to(device)
            kpts_y_gt = kpts_y_gt.to(device)
            kpts_mask_gt = kpts_mask_gt.to(device)
            
            optimizer.zero_grad()
            pred_mask, pred_x, pred_y, pred_ln_var = model(imgs)
            
            l_mask = mask_loss(pred_mask, masks_gt)
            l_nll = nll_loss(pred_x, pred_y, pred_ln_var, kpts_x_gt, kpts_y_gt, kpts_mask_gt)
            l_inv = invis_loss(pred_ln_var, kpts_mask_gt)
            
            loss = l_mask + l_nll + l_inv * 0.1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "pose_model.pth")

if __name__ == "__main__":
    train()
