import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from model import PoseDetectionModel
from train import CocoDataset, letterbox
import torch.nn.functional as F

def visualize_extended(model_path, dataset_path, img_idx=0, suffix=''):
    device = torch.device('cpu')
    viz_dir = '.visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
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
    
    H4, W4 = mask_pred.shape
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    # 1. viz_gt.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(mask_gt.squeeze().numpy(), cmap='gray')
    axes[0].set_title('GT Mask')
    axes[0].axis('off')
    axes[1].imshow(img)
    img_info = dataset.coco.loadImgs(dataset.img_ids[img_idx])[0]
    ann_ids = dataset.coco.getAnnIds(imgIds=img_info['id'], catIds=dataset.cat_ids)
    anns = dataset.coco.loadAnns(ann_ids)
    orig_img = cv2.imread(os.path.join(img_dir, img_info['file_name']))
    _, (ratio_x, ratio_y), (pad_x, pad_y) = letterbox(orig_img)
    for ann in anns:
        kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
        kp[:, 0] = kp[:, 0] * ratio_x + pad_x
        kp[:, 1] = kp[:, 1] * ratio_y + pad_y
        for i, j in skeleton:
            if kp[i, 2] > 0 and kp[j, 2] > 0:
                axes[1].plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]], color='red', linewidth=2)
        for k in range(17):
            if kp[k, 2] > 0:
                axes[1].scatter(kp[k, 0], kp[k, 1], s=20, color='blue')
    axes[1].set_title('GT Skeleton')
    axes[1].axis('off')
    plt.savefig(os.path.join(viz_dir, f'viz_gt{suffix}.png'))
    plt.close()

    # 2. viz_pred.png (Single Best Cell based on Mask * Avg Confidence)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(mask_pred, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Predicted Mask')
    axes[0].axis('off')
    axes[1].imshow(img)
    
    # Selection logic: mask * avg(precision)
    # precision = exp(-lv)
    precision = np.exp(-lv)
    avg_precision = precision.mean(axis=0)
    selection_metric = mask_pred * avg_precision
    
    best_idx = np.argmax(selection_metric)
    ry, rx = np.unravel_index(best_idx, mask_pred.shape)
    
    lv_best = lv[:, ry, rx]
    conf_thresh = -2.0 # Still filter individual points for skeleton drawing
    
    pkpts_plot = {}
    for k in range(17):
        if lv_best[k] < conf_thresh:
            kx, ky = mx[k, ry, rx] / 4 * 640, my[k, ry, rx] / 4 * 640
            pkpts_plot[k] = [kx, ky]
            axes[1].scatter(kx, ky, s=20, color='lime')
            
    for i, j in skeleton:
        if i in pkpts_plot and j in pkpts_plot:
            axes[1].plot([pkpts_plot[i][0], pkpts_plot[j][0]], [pkpts_plot[i][1], pkpts_plot[j][1]], color='lime', linewidth=2)

    axes[1].set_title(f'Predicted Skeleton (Best Cell @ {rx},{ry})')
    axes[1].axis('off')
    plt.savefig(os.path.join(viz_dir, f'viz_pred{suffix}.png'))
    plt.close()

    # 3. Absolute Position Heatmaps
    k_viz = 0
    for k in range(17):
        if kpts_mask_gt[k].sum() > 0:
            k_viz = k
            break
            
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    m_gt_np = (mask_gt.squeeze() > 0.5).numpy()
    m_pred_np = (mask_pred > 0.5)
    
    gx_m = np.full_like(mx[k_viz], np.nan); gy_m = np.full_like(my[k_viz], np.nan)
    gx_m[m_gt_np] = kpts_x_gt[k_viz].numpy()[m_gt_np]
    gy_m[m_gt_np] = kpts_y_gt[k_viz].numpy()[m_gt_np]
    im0 = axes[0, 0].imshow(gx_m, cmap='jet', vmin=0, vmax=4); axes[0, 0].set_facecolor('black')
    im1 = axes[0, 1].imshow(gy_m, cmap='jet', vmin=0, vmax=4); axes[0, 1].set_facecolor('black')
    fig.colorbar(im0, ax=axes[0, 0]); fig.colorbar(im1, ax=axes[0, 1])
    
    px_m = np.full_like(mx[k_viz], np.nan); py_m = np.full_like(my[k_viz], np.nan)
    px_m[m_pred_np] = mx[k_viz][m_pred_np]
    py_m[m_pred_np] = my[k_viz][m_pred_np]
    im2 = axes[1, 0].imshow(px_m, cmap='jet', vmin=0, vmax=4); axes[1, 0].set_facecolor('black')
    im3 = axes[1, 1].imshow(py_m, cmap='jet', vmin=0, vmax=4); axes[1, 1].set_facecolor('black')
    fig.colorbar(im2, ax=axes[1, 0]); fig.colorbar(im3, ax=axes[1, 1])
    
    for ax in axes.flat:
        for ann in anns:
            kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
            if kp[k_viz, 2] > 0:
                kx_p = (kp[k_viz, 0] * ratio_x + pad_x) / 4
                ky_p = (kp[k_viz, 1] * ratio_y + pad_y) / 4
                ax.scatter(kx_p, ky_p, marker='x', color='white', s=100, linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'viz_abs_kpt{suffix}.png'))
    plt.close()

if __name__ == "__main__":
    for i in range(10):
        print(f"Generating visualization for image {i}...")
        visualize_extended('pose_model.pth', 'code/data/coco_subset', img_idx=i, suffix=f'_{i}')
    print("Visualizations saved in .visualization/")
