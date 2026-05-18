import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, SPPF, Bottleneck
from ultralytics.nn.modules.block import C3k2, C2PSA, C3k

class PoseDetectionModel(nn.Module):
    def __init__(self, yolo_model_name='yolo26n-pose.pt', num_keypoints=17):
        super().__init__()
        # Load pre-trained YOLO26 model
        base_yolo = YOLO(yolo_model_name).model
        
        # Extract backbone and neck up to P3
        # Layer indices based on yolo26n.pt:
        # 0-9: Backbone + SPPF
        # 10: C2PSA
        # 11: Upsample
        # 12: Concat
        # 13: C3k2 (P4 neck)
        # 14: Upsample
        # 15: Concat
        # 16: C3k2 (P3 neck)
        
        self.backbone = base_yolo.model[:10]
        self.neck_p5_p4 = base_yolo.model[10:14] # C2PSA, Upsample, Concat, C3k2
        self.neck_p4_p3 = base_yolo.model[14:17] # Upsample, Concat, C3k2
        
        # P2 extension
        # P2 backbone features are from layer 2
        self.p2_backbone = base_yolo.model[0:3] 
        # Wait, I need to be careful with sharing layers.
        # Let's just use the full model and capture intermediate features.
        
        self.full_base = base_yolo.model
        
        # Number of channels for P3 neck output (layer 16) is 64 in yolo26n
        # Number of channels for P2 backbone (layer 2) is 64 in yolo26n
        p3_channels = 64
        p2_channels = 64
        
        self.upsample_p3_p2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_neck_c3k2 = C3k2(p3_channels + p2_channels, 64, n=1, shortcut=True)
        
        # Heads with CoordConv (grid concatenated to input)
        # Input to heads will be 64 (features) + 2 (grid) = 66 channels
        self.mask_head = nn.Sequential(
            Conv(p3_channels, 32, 3),
            Conv(32, 32, 3),
            nn.Conv2d(32, 1, 1)
        )
        
        self.keypoint_head = nn.Sequential(
            Conv(p3_channels, 64, 3),
            Conv(64, 64, 3),
            nn.Conv2d(64, num_keypoints * 3, 1)
        )
    
    def forward(self, x):
        # Manual forward based on YOLO26 structure
        x0 = self.full_base[0](x)
        x1 = self.full_base[1](x0)
        x2 = self.full_base[2](x1) # P2 backbone
        x3 = self.full_base[3](x2)
        x4 = self.full_base[4](x3) # P3 backbone
        x5 = self.full_base[5](x4)
        x6 = self.full_base[6](x5) # P4 backbone
        x7 = self.full_base[7](x6)
        x8 = self.full_base[8](x7) # P5 backbone
        x9 = self.full_base[9](x8) # SPPF
        x10 = self.full_base[10](x9) # C2PSA
        
        x11 = self.full_base[11](x10) # Upsample
        x12 = torch.cat([x11, x6], 1) # Concat with P4
        x13 = self.full_base[13](x12) # P4 neck
        
        x14 = self.full_base[14](x13) # Upsample
        x15 = torch.cat([x14, x4], 1) # Concat with P3
        x16 = self.full_base[16](x15) # P3 neck
        
        # Custom P2 extension
        x17 = self.upsample_p3_p2(x16)
        x18 = torch.cat([x17, x2], 1) # Concat with P2 backbone
        x19 = self.p2_neck_c3k2(x18)
        
        mask = self.mask_head(x19)
        kpts = self.keypoint_head(x19)

        B, _, H4, W4 = x19.shape
        y_c, x_c = torch.meshgrid(
            torch.linspace(0.5 / H4 * 4, 4 - 0.5 / H4 * 4, H4, device=x.device),
            torch.linspace(0.5 / W4 * 4, 4 - 0.5 / W4 * 4, W4, device=x.device),
            indexing='ij'
        )
        
        # kpts shape: [B, 17*3, H4, W4]
        kpts = kpts.view(B, 17, 3, H4, W4)
        
        dx = kpts[:, :, 0, :, :]
        dy = kpts[:, :, 1, :, :]
        ln_var = kpts[:, :, 2, :, :]
        
        # Add grid positions
        mean_x = dx + x_c.view(1, 1, H4, W4)
        mean_y = dy + y_c.view(1, 1, H4, W4)
        
        ln_var = torch.clamp(ln_var, -16, 6)
        
        return mask, mean_x, mean_y, ln_var

def invis_loss(pred_ln_var, kpt_mask):
    invis_loss = pred_ln_var * (kpt_mask - 1.0)
    return invis_loss.sum() / (kpt_mask.sum() + 1e-6)

def nll_loss(pred_mean_x, pred_mean_y, pred_ln_var, target_x, target_y, kpt_mask):
    var = torch.exp(pred_ln_var)
    dist_sq = (pred_mean_x - target_x)**2 + (pred_mean_y - target_y)**2
    nll_loss = dist_sq / (2 * var) + pred_ln_var
    visible_loss = nll_loss * kpt_mask
    return visible_loss.sum() / (kpt_mask.sum() + 1e-6)

def mse_loss(pred_mean_x, pred_mean_y, target_x, target_y, kpt_mask):
    dist_sq = (pred_mean_x - target_x)**2 + (pred_mean_y - target_y)**2
    visible_loss = dist_sq * kpt_mask
    return visible_loss.sum() / (kpt_mask.sum() + 1e-6)

def mask_loss(pred_mask, target_mask):
    return F.binary_cross_entropy_with_logits(pred_mask, target_mask)

if __name__ == "__main__":
    model = PoseDetectionModel()
    dummy_input = torch.randn(1, 3, 640, 640)
    mask, kpts = model(dummy_input)
    print(f"Mask shape: {mask.shape}")
    print(f"Keypoints shape: {kpts.shape}")
