import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# --- 1. Load Images ---
print("--- 1. Loading Images ---")
img1 = cv2.imread('data/im2.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/im6.png', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Could not load images. Check 'data' folder and file paths.")
    exit()
else:
    print("Images loaded successfully.")

# --- 2. Apply SIFT for Correspondence Points ---
print("\n--- 2. Applying SIFT for Correspondence Points ---")
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance: # Ratio test
        good_matches.append(m)

pts1_sift = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2_sift = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

print(f"Found {len(good_matches)} good SIFT matches.")

# Optional: Draw matches to visualize
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title("SIFT Matches")
plt.axis('off')
plt.show()

# --- 3. Implement the Normalized 8-Point Algorithm for the Fundamental Matrix ---
print("\n--- 3. Implementing Normalized 8-Point Algorithm for Fundamental Matrix ---")

def normalize_points(pts):
    """
    Normalizes a set of 2D points to have centroid at origin and average distance sqrt(2).
    """
    pts = pts.astype(np.float64) # Ensure float type

    # Calculate centroid
    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])
    centroid = np.array([mean_x, mean_y])

    # Calculate scale factor
    # This is equivalent to dividing by RMS distance to origin after translation
    s = np.sqrt(2) / np.mean(np.sqrt(np.sum((pts - centroid)**2, axis=1)))

    # Transformation matrix
    T = np.array([
        [s, 0, -s * mean_x],
        [0, s, -s * mean_y],
        [0, 0, 1]
    ])

    # Apply transformation
    # Add a column of ones to make points homogeneous for matrix multiplication
    homogeneous_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_pts_h = (T @ homogeneous_pts.T).T
    normalized_pts = normalized_pts_h[:, :2] # Convert back to non-homogeneous for algorithm

    return normalized_pts, T

def eight_point_algorithm(pts1, pts2):
    """
    Estimates the Fundamental Matrix F using the normalized 8-point algorithm.
    pts1 and pts2 are Nx2 arrays of corresponding points.
    """
    print("Normalizing points...")
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)
    print("Points normalized.")

    # Construct matrix A for the linear system Af = 0
    # Each correspondence (x1, y1) <-> (x2, y2) gives one row in A
    # The equation is x2' F x1 = 0
    # [x2 x2y1 x2y1 y2x1 y2y1 y2 x1 y1 1] . f = 0
    A = []
    for i in range(len(norm_pts1)):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    A = np.array(A)

    # Solve Af = 0 using SVD to find f (vectorized F)
    # F is the last column of V (or V.T[-1])
    U, S, V = np.linalg.svd(A)
    F_norm = V[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    # The fundamental matrix must have rank 2.
    # Set the smallest singular value to 0.
    U_F, S_F, V_F = np.linalg.svd(F_norm)
    S_F[2] = 0
    F_rank2_norm = U_F @ np.diag(S_F) @ V_F

    # Denormalize F
    # F = T2.T @ F_norm @ T1
    F = T2.T @ F_rank2_norm @ T1
    return F

# Use the points from SIFT matches
pts1_alg = pts1_sift.squeeze()
pts2_alg = pts2_sift.squeeze()

# Estimate Fundamental Matrix using your implementation
F_custom = eight_point_algorithm(pts1_alg, pts2_alg)
print("\nCustom Estimated Fundamental Matrix F:")
print(F_custom)

# For comparison, OpenCV's findFundamentalMat
F_cv, mask = cv2.findFundamentalMat(pts1_sift, pts2_sift, cv2.FM_8POINT)
print("\nOpenCV Estimated Fundamental Matrix F:")
print(F_cv)

# --- 4. Estimate 3D Structure (Simplified Gold Standard / Triangulation) ---
print("\n--- 4. Estimating 3D Structure (Triangulation) ---")

image_width = img1.shape[1]
image_height = img1.shape[0]

# Placeholder K - you should get this from camera calibration
K = np.array([
    [1.2 * image_width, 0, image_width / 2],
    [0, 1.2 * image_width, image_height / 2],
    [0, 0, 1]
], dtype=np.float64)
print("\nAssumed Camera Intrinsic Matrix K:")
print(K)

# We need the Essential Matrix E from F and K
E = K.T @ F_custom @ K
print("\nEstimated Essential Matrix E:")
print(E)

# Recover Pose
points, R_rec, t_rec, inliers = cv2.recoverPose(E, pts1_sift, pts2_sift, K)
print(f"\nRecovered Pose (R, t) from Essential Matrix:")
print("R_rec:\n", R_rec)
print("t_rec:\n", t_rec)

P1 = K @ np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float64)

P2 = K @ np.hstack((R_rec, t_rec))
print("\nProjection Matrix P1 (Camera 1):")
print(P1)
print("\nProjection Matrix P2 (Camera 2):")
print(P2)

# Triangulate 3D points
points_4D_hom = cv2.triangulatePoints(P1, P2, pts1_sift.squeeze().T, pts2_sift.squeeze().T)
points_3D_initial = (points_4D_hom / points_4D_hom[3]).T[:, :3]

print(f"\nTriangulated {len(points_3D_initial)} 3D points (initial).")
print("First 5 initial 3D points:")
print(points_3D_initial[:5])

# --- 5. Implement Bundle Adjustment ---
# --- 5. Implement Bundle Adjustment ---
print("\n--- 5. Implementing Bundle Adjustment ---")

# Define the error function for least_squares
def bundle_adjustment_cost(params, K, pts1, pts2, num_points):
    """
    Cost function for bundle adjustment.
    params: [rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z, P1_x, P1_y, P1_z, P2_x, P2_y, P2_z, ...]
    """
    # Extract camera pose parameters for camera 2
    rvec = params[0:3]
    tvec = params[3:6].reshape(3, 1) # Ensure tvec is (3,1)

    # Extract 3D points
    points_3D_optimized = params[6:].reshape(-1, 3)

    # Camera 1 is fixed at origin [I|0]
    P1 = K @ np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64)

    # Camera 2 projection matrix
    R_optimized, _ = cv2.Rodrigues(rvec)
    P2_optimized = K @ np.hstack((R_optimized, tvec))

    # Reproject 3D points to 2D for both cameras
    # cv2.projectPoints expects 3D points as (N, 1, 3) or (N, 3) and returns (N, 1, 2)
    projected_pts1, _ = cv2.projectPoints(points_3D_optimized, np.zeros(3), np.zeros(3), K, None)
    projected_pts2, _ = cv2.projectPoints(points_3D_optimized, rvec, tvec, K, None)

    # Calculate reprojection errors
    error1 = (projected_pts1.squeeze() - pts1).flatten()
    error2 = (projected_pts2.squeeze() - pts2).flatten()

    # Combine errors
    return np.hstack((error1, error2))

# Prepare data for optimization
# Convert R to Rodrigues vector and ensure it's a 1D array
rvec_initial = cv2.Rodrigues(R_rec)[0].flatten()
# Ensure t is a 1D array
tvec_initial = t_rec.flatten()

# Initial parameters: [rvec_c2, tvec_c2, Pts_3D_x, Pts_3D_y, Pts_3D_z, ...]
initial_params = np.hstack((rvec_initial, tvec_initial, points_3D_initial.flatten()))

print(f"Initial parameters shape: {initial_params.shape}")
print(f"Number of 3D points being optimized: {len(points_3D_initial)}")
print(f"Number of 2D correspondences: {len(pts1_sift)}")

# Perform optimization
# We are passing the original 2D points (pts1_sift.squeeze(), pts2_sift.squeeze())
# as fixed observations to the cost function.
# The `args` tuple contains extra arguments to the cost function.
# `loss='huber'` can make it more robust to outliers
# `ftol` and `xtol` define the termination tolerance
result = least_squares(bundle_adjustment_cost, initial_params,
                       args=(K, pts1_sift.squeeze(), pts2_sift.squeeze(), len(points_3D_initial)),
                       verbose=2, x_scale='jac', ftol=1e-8, xtol=1e-8, loss='huber')

# Extract optimized parameters
optimized_params = result.x
rvec_optimized = optimized_params[0:3]
tvec_optimized = optimized_params[3:6].reshape(3, 1)
points_3D_optimized = optimized_params[6:].reshape(-1, 3)

# Convert optimized rvec back to rotation matrix
R_optimized, _ = cv2.Rodrigues(rvec_optimized)

print("\nBundle Adjustment Complete.")
print("Optimized Camera 2 Rotation (Rodrigues):\n", rvec_optimized)
print("Optimized Camera 2 Translation:\n", tvec_optimized)
print(f"Optimized {len(points_3D_optimized)} 3D points.")
print("First 5 Optimized 3D points:")
print(points_3D_optimized[:5])

# --- 6. Plot the Final 3D Scene as a Dense Point Cloud (now optimized sparse cloud) ---
print("\n--- 6. Plotting Optimized 3D Point Cloud ---")

# Filter out points that are too far, behind the camera, or have unrealistic Z values
max_depth = 100.0 # Example maximum depth
min_depth = 0.1   # Example minimum depth
valid_points_mask = (points_3D_optimized[:, 2] > min_depth) & (points_3D_optimized[:, 2] < max_depth)
valid_points_optimized = points_3D_optimized[valid_points_mask]

print(f"Plotting {len(valid_points_optimized)} valid 3D points after filtering (optimized).")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(valid_points_optimized[:, 0], valid_points_optimized[:, 1], valid_points_optimized[:, 2], c='blue', marker='.', s=1)

# Plot camera positions
# Camera 1 is at origin [0,0,0]
ax.scatter(0, 0, 0, c='red', marker='o', s=100, label='Camera 1 Position')
# Camera 2 position is -R.T @ t (inverse transformation)
cam2_pos_optimized = -R_optimized.T @ tvec_optimized
ax.scatter(cam2_pos_optimized[0], cam2_pos_optimized[1], cam2_pos_optimized[2], c='green', marker='o', s=100, label='Camera 2 Position (Optimized)')

ax.set_xlabel('X (Horizontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Depth)')
ax.set_title('Optimized 3D Point Cloud Reconstruction (Blue points, Cameras: Red, Green)')
ax.legend()

max_range = np.array([valid_points_optimized[:,0].max()-valid_points_optimized[:,0].min(),
                      valid_points_optimized[:,1].max()-valid_points_optimized[:,1].min(),
                      valid_points_optimized[:,2].max()-valid_points_optimized[:,2].min()]).max() / 2.0

mid_x = (valid_points_optimized[:,0].max()+valid_points_optimized[:,0].min()) * 0.5
mid_y = (valid_points_optimized[:,1].max()+valid_points_optimized[:,1].min()) * 0.5
mid_z = (valid_points_optimized[:,2].max()+valid_points_optimized[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

print("\nReconstruction complete and plotted (with Bundle Adjustment).")

# --- Save SIFT Match Visualization ---
plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title("SIFT Matches (used for BA)")
plt.axis('off')
plt.savefig('sift_matches_ba.png', dpi=300, bbox_inches='tight')
print("✅ SIFT match visualization saved to 'sift_matches_ba.png'")

# --- Save Point Cloud ---
def save_ply(filename, points):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for p in points:
            f.write(f'{p[0]} {p[1]} {p[2]}\n')
    print(f"✅ Optimized 3D point cloud saved to '{filename}'")

save_ply('reconstruction_2_optimized.ply', valid_points_optimized)