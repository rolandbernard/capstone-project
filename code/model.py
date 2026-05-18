import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, SPPF, Bottleneck
from ultralytics.nn.modules.block import C3k2, C2PSA, C3k

class PoseDetectionModel(nn.Module):
    def __init__(self, yolo_model_name='yolo26n.pt', num_keypoints=17):
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
            Conv(66, 32, 3),
            nn.Conv2d(32, 1, 1)
        )
        
        self.keypoint_head = nn.Sequential(
            Conv(66, 64, 3),
            nn.Conv2d(64, num_keypoints * 3, 1)
        )
        
        # Initialize variance bias to something reasonable (e.g. -2 for ln_var)
        # This prevents the model from starting in a "high variance" trap
        with torch.no_grad():
            self.keypoint_head[-1].bias[2::3].fill_(-2.0)
        
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
        
        # Prepare CoordConv grid
        B, _, H4, W4 = x19.shape
        y_c, x_c = torch.meshgrid(
            torch.linspace(0.5 / H4 * 4, 4 - 0.5 / H4 * 4, H4, device=x.device),
            torch.linspace(0.5 / W4 * 4, 4 - 0.5 / W4 * 4, W4, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([x_c, y_c], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        x19_coord = torch.cat([x19, grid], dim=1)
        
        mask = self.mask_head(x19_coord)
        kpts = self.keypoint_head(x19_coord)
        
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

def gaussian_nll_loss(pred_mean_x, pred_mean_y, pred_ln_var, target_x, target_y, kpt_mask, person_mask):
    """
    pred_mean_x/y: [B, 17, H4, W4]
    pred_ln_var: [B, 17, H4, W4]
    target_x/y: [B, 17, H4, W4]
    kpt_mask: [B, 17, H4, W4] (visible keypoints)
    person_mask: [B, 1, H4, W4] (any person pixels)
    """
    var = torch.exp(pred_ln_var)
    dist_sq = (pred_mean_x - target_x)**2 + (pred_mean_y - target_y)**2
    
    # 1. Standard NLL + L2 for visible keypoints
    nll_loss = dist_sq / (2 * var + 1e-8) + pred_ln_var
    l2_loss = dist_sq * 10.0 
    visible_loss = (nll_loss + l2_loss) * kpt_mask
    
    # 2. Uncertainty supervision for invisible keypoints on person pixels
    # invis_mask = (person is present) AND (this specific kpt is NOT visible)
    invis_mask = (person_mask > 0.5) & (kpt_mask < 0.5)
    # Force ln_var to 6.0 (max uncertainty)
    uncertainty_loss = (pred_ln_var - 6.0)**2 * 0.1 # Scaled down for stability
    invisible_loss = uncertainty_loss * invis_mask
    
    total_loss = visible_loss.sum() + invisible_loss.sum()
    
    return total_loss / (person_mask.sum() * 17 + 1e-6)

def mask_loss(pred_mask, target_mask):
    """
    pred_mask: [B, 1, H4, W4]
    target_mask: [B, 1, H4, W4]
    """
    return F.binary_cross_entropy_with_logits(pred_mask, target_mask)

if __name__ == "__main__":
    model = PoseDetectionModel()
    dummy_input = torch.randn(1, 3, 640, 640)
    mask, kpts = model(dummy_input)
    print(f"Mask shape: {mask.shape}")
    print(f"Keypoints shape: {kpts.shape}")
