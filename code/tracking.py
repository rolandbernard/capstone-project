import numpy as np

class PoseEKF:
    def __init__(self, num_keypoints=17, dt=0.033):
        self.num_kpts = num_keypoints
        self.dt = dt
        
        # State: [X1, Y1, Z1, ..., X17, Y17, Z17, dX1, dY1, dZ1, ..., dX17, dY17, dZ17]
        self.dim_state = num_keypoints * 3 * 2
        self.x = np.zeros(self.dim_state)
        self.P = np.eye(self.dim_state) * 10.0 # Initial uncertainty
        
        # Transition matrix F
        self.F = np.eye(self.dim_state)
        for i in range(num_keypoints * 3):
            self.F[i, i + num_keypoints * 3] = dt
            
        # Process noise Q
        self.Q = np.eye(self.dim_state) * 0.01
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def project(self, P_mat, point_3d):
        """
        P_mat: 3x4 projection matrix
        point_3d: [X, Y, Z]
        Returns: [u, v], Jacobian H_i (2x3)
        """
        X, Y, Z = point_3d
        hom_point = np.array([X, Y, Z, 1.0])
        proj = P_mat @ hom_point
        
        w = proj[2]
        u = proj[0] / (w + 1e-8)
        v = proj[1] / (w + 1e-8)
        
        # Jacobian H_i = d(u,v)/d(X,Y,Z)
        # u = N1/D, v = N2/D
        # du/dX = (dN1/dX * D - N1 * dD/dX) / D^2
        D = w
        N1 = proj[0]
        N2 = proj[1]
        
        dD = P_mat[2, :3]
        dN1 = P_mat[0, :3]
        dN2 = P_mat[1, :3]
        
        H_u = (dN1 * D - N1 * dD) / (D**2 + 1e-8)
        H_v = (dN2 * D - N2 * dD) / (D**2 + 1e-8)
        
        return np.array([u, v]), np.vstack([H_u, H_v])

    def update(self, observations, cam_matrices):
        """
        observations: list of {kpt_idx: {'pos': [u, v], 'var': sigma2}, ...} for each camera
        cam_matrices: list of 3x4 projection matrices
        """
        for obs, P_mat in zip(observations, cam_matrices):
            for kpt_idx, data in obs.items():
                z = data['pos']
                R = np.eye(2) * data['var']
                
                # Get current 3D pos for this kpt
                start_idx = kpt_idx * 3
                point_3d = self.x[start_idx:start_idx+3]
                
                z_pred, H_i = self.project(P_mat, point_3d)
                
                # Full Jacobian H
                H = np.zeros((2, self.dim_state))
                H[:, start_idx:start_idx+3] = H_i
                
                # Innovation
                y = z - z_pred
                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S)
                
                print(f"z: {z}, z_pred: {z_pred}, y: {y}")
                print(f"H_i: {H_i}")
                print(f"K @ y max: {(K @ y).max()}")
                
                self.x = self.x + K @ y
                self.P = (np.eye(self.dim_state) - K @ H) @ self.P

def track_persons(skeletons_by_cam, cam_matrices, existing_tracks):
    """
    skeletons_by_cam: list of lists of skeletons (from run_inference)
    cam_matrices: list of 3x4 projection matrices
    existing_tracks: list of PoseEKF objects
    """
    # 1. Prediction for all tracks
    for track in existing_tracks:
        track.predict()
        
    # 2. Association and Update
    # This part is complex (matching skeletons to tracks).
    # For now, let's assume one person for simplicity or use a greedy match.
    
    if existing_tracks:
        # Update first track with observations from all cameras
        observations = []
        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            if skeletons:
                # Pick the best skeleton (placeholder)
                observations.append(skeletons[0])
            else:
                observations.append({})
        
        existing_tracks[0].update(observations, cam_matrices)
    else:
        # Initialize a new track if skeletons are found
        new_track = PoseEKF()
        # Initialize state from triangulated points if possible...
        existing_tracks.append(new_track)

if __name__ == "__main__":
    ekf = PoseEKF()
    # Mock update
    cam_mat = np.array([
        [500, 0, 320, 0],
        [0, 500, 240, 0],
        [0, 0, 1, 0]
    ])
    # Set initial depth to 1.0 for kpt 0
    ekf.x[2] = 1.0
    obs = {0: {'pos': [330, 250], 'var': 1.0}}
    ekf.update([obs], [cam_mat])
    print(f"Updated 3D pos for kpt 0: {ekf.x[:3].tolist()}")
