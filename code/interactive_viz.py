import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from model import PoseDetectionModel
from train import CocoDataset, letterbox

def run_interactive(model_path, dataset_path, img_idx=0):
    device = torch.device('cpu')
    
    # Load model
    model = PoseDetectionModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset sample
    ann_file = '/tmp/coco_ann/annotations/person_keypoints_val2017.json'
    img_dir = 'code/data/coco_subset/images'
    dataset = CocoDataset(ann_file, img_dir)
    img_tensor, mask_gt, kpts_x_gt, kpts_y_gt, kpts_mask_gt = dataset[img_idx]
    
    # Run inference
    with torch.no_grad():
        pred_mask_logit, pred_x, pred_y, pred_ln_var = model(img_tensor.unsqueeze(0))
        
    img = img_tensor.permute(1, 2, 0).numpy()
    mask_pred = torch.sigmoid(pred_mask_logit).squeeze().numpy()
    mx = pred_x.squeeze().numpy()
    my = pred_y.squeeze().numpy()
    lv = pred_ln_var.squeeze().numpy()
    
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Image
    ax_img = axes[0]
    ax_img.imshow(img)
    ax_img.set_title('Image (Click to show skeleton)')
    ax_img.axis('off')
    
    # Plot Mask
    ax_mask = axes[1]
    ax_mask.imshow(mask_pred > 0.5, cmap='gray', vmin=0, vmax=1)
    ax_mask.set_title('Predicted Mask (Click here too)')
    ax_mask.axis('off')
    
    current_skeleton_lines = []
    current_kpt_dots = []

    def update_skeleton(event):
        if event.inaxes not in axes:
            return
            
        # Get coordinates in 160x160 grid
        if event.inaxes == ax_img:
            # event.xdata, ydata are in 0-640 range
            rx = int(event.xdata / 640 * 160)
            ry = int(event.ydata / 640 * 160)
        else:
            # event.xdata, ydata are in 0-160 range
            rx = int(event.xdata)
            ry = int(event.ydata)
            
        if not (0 <= rx < 160 and 0 <= ry < 160):
            return
            
        print(f"Selected Cell: ({rx}, {ry}), Mask Conf: {mask_pred[ry, rx]:.2f}")
        
        # Clear previous
        for line in current_skeleton_lines:
            line.remove()
        for dot in current_kpt_dots:
            dot.remove()
        current_skeleton_lines.clear()
        current_kpt_dots.clear()
        
        # Extract skeleton from this cell
        lv_cell = lv[:, ry, rx]
        conf_thresh = -1.0 # Lenient for exploration
        
        pkpts = {}
        for k in range(17):
            if lv_cell[k] < conf_thresh:
                kx, ky = mx[k, ry, rx] / 4 * 640, my[k, ry, rx] / 4 * 640
                pkpts[k] = [kx, ky]
                dot = ax_img.scatter(kx, ky, s=30, color='lime', zorder=5)
                current_kpt_dots.append(dot)
                
        for i, j in skeleton:
            if i in pkpts and j in pkpts:
                line, = ax_img.plot([pkpts[i][0], pkpts[j][0]], [pkpts[i][1], pkpts[j][1]], color='lime', linewidth=2, zorder=4)
                current_skeleton_lines.append(line)
        
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', update_skeleton)
    
    print("Interactive mode active. Click on the plots to see skeletons.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help='Image index in subset (0-9)')
    args = parser.parse_args()
    
    run_interactive('pose_model.pth', 'code/data/coco_subset', img_idx=args.idx)
