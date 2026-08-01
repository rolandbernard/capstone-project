
import sys

import dataset
import visualize
import kflearn
import tracker
import kalman

data = dataset.KalmanDataset("./data/kalman/train")
data_val = dataset.KalmanDataset("./data/kalman/val")
raw_data = dataset.CmuPanopticDataset(path="./data/panoptic")
cams = raw_data.get_some_cams()

print("Number of training samples:", len(data))
print("Number of validation samples:", len(data_val))

fps, track = data[int(sys.argv[1])]
print(f"FPS: {fps.item():.2f}")
# plot = visualize.MinimalSkeletonPlayer(track, fps.item())
model = kalman.LearnedPhysics(tracker.build_constrained_physics())
_, pre_cov, sim_track, post_cov = kflearn.simulate_kalman_filter(model, fps, track, cams)
for pre, post in zip(pre_cov, post_cov):
    print("::", pre.diag().sqrt().tolist())
    print("=>", post.diag().sqrt().tolist())
plot = visualize.MinimalSkeletonPlayer(sim_track.detach().cpu().view_as(track), fps.item())
plot.setup_cameras(cams)
plot.show()
