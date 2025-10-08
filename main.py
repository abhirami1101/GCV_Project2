import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Load Images ---
print("--- 1. Loading Images ---")
img1 = cv2.imread('data/r_0.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/r_2.png', cv2.IMREAD_GRAYSCALE)

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

# Assuming a standard camera intrinsic matrix K.
# For metric reconstruction, accurate K is critical.
# For simplicity, we'll assume a pinhole camera with known (or estimated) intrinsics.
# The focal length (fx, fy) is often in the range of image width/height.
# The principal point (cx, cy) is typically near the image center.
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
# E = K.T @ F @ K
E = K.T @ F_custom @ K
print("\nEstimated Essential Matrix E:")
print(E)

# Decompose Essential Matrix into Rotation R and Translation t
# There are four possible (R, t) pairs. We need to select the physically correct one.
# retval, R1, R2, t = cv2.decomposeEssentialMat(E)
R1, R2, t = cv2.decomposeEssentialMat(E)


# For a purely horizontal motion along the x-axis, the translation vector `t`
# should ideally be [tx, 0, 0] or similar. Let's check `t`:
print(f"\nDecomposed Translation Vector t:\n{t}")

# Construct projection matrices P1 and P2
# Camera 1 is at the origin, looking along the Z-axis (identity rotation)
P1 = K @ np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float64)

# We need to choose the correct R and t.
# The best pair results in most points being in front of both cameras (positive Z depth).
# OpenCV's recoverPose can help verify, but here we'll simplify and pick one.
# Generally, for purely horizontal motion, t will have a significant x component and small y, z.
# And R will be close to identity or a small rotation.
# Let's try R1 and t and ensure points are in front.

# The `recoverPose` function from OpenCV can give us a correct R and t pair
# and also returns the number of inliers.
points, R_rec, t_rec, inliers = cv2.recoverPose(E, pts1_sift, pts2_sift, K)
print(f"\nRecovered Pose (R, t) from Essential Matrix:")
print("R_rec:\n", R_rec)
print("t_rec:\n", t_rec)

P2 = K @ np.hstack((R_rec, t_rec))
print("\nProjection Matrix P1 (Camera 1):")
print(P1)
print("\nProjection Matrix P2 (Camera 2):")
print(P2)

# Triangulate 3D points
# OpenCV's triangulatePoints expects points as 2xN arrays (x,y for each point).
points_4D_hom = cv2.triangulatePoints(P1, P2, pts1_sift.squeeze().T, pts2_sift.squeeze().T)

# Convert from homogeneous coordinates to 3D Cartesian coordinates
points_3D = (points_4D_hom / points_4D_hom[3]).T[:, :3]

print(f"\nTriangulated {len(points_3D)} 3D points.")
print("First 5 3D points:")
print(points_3D[:5])

# --- 5. Plot the Final 3D Scene as a Dense Point Cloud ---
print("\n--- 5. Plotting 3D Point Cloud ---")

# Filter out points that are too far, behind the camera, or have unrealistic Z values
# These can occur due to noise or poor matches.
# A common filter is to keep points where Z > 0 for both cameras.
# Here, we check Z > 0 for the first camera, and also filter by a reasonable depth range.
# For the second camera, reprojection would ensure it's also in front.
# You might need to adjust these thresholds based on your scene.
max_depth = 100.0 # Example maximum depth
min_depth = 0.1   # Example minimum depth
valid_points_mask = (points_3D[:, 2] > min_depth) & (points_3D[:, 2] < max_depth)
valid_points = points_3D[valid_points_mask]

print(f"Plotting {len(valid_points)} valid 3D points after filtering.")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c='blue', marker='.', s=1)

# Plot camera positions (optional, but good for context)
# Camera 1 is at origin [0,0,0]
ax.scatter(0, 0, 0, c='red', marker='o', s=100, label='Camera 1 Position')
# Camera 2 position is -R.T @ t (inverse transformation)
cam2_pos = -R_rec.T @ t_rec
ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c='green', marker='o', s=100, label='Camera 2 Position')


ax.set_xlabel('X (Horizontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Depth)')
ax.set_title('3D Point Cloud Reconstruction (Blue points, Cameras: Red, Green)')
ax.legend()

# Set equal aspect ratio to prevent distortion in perception of depth
# Need to find max range for each axis
max_range = np.array([valid_points[:,0].max()-valid_points[:,0].min(),
                      valid_points[:,1].max()-valid_points[:,1].min(),
                      valid_points[:,2].max()-valid_points[:,2].min()]).max() / 2.0

mid_x = (valid_points[:,0].max()+valid_points[:,0].min()) * 0.5
mid_y = (valid_points[:,1].max()+valid_points[:,1].min()) * 0.5
mid_z = (valid_points[:,2].max()+valid_points[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

print("\nReconstruction complete and plotted.")

# --- Save SIFT Match Visualization ---
plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title("SIFT Matches")
plt.axis('off')
plt.savefig('sift_matches.png', dpi=300, bbox_inches='tight')
print("✅ SIFT match visualization saved to 'sift_matches.png'")

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
    print(f"✅ 3D point cloud saved to '{filename}'")

save_ply('reconstruction_2.ply', valid_points)
