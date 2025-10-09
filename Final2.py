import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# --- Configuration ---
image_path_1 = 'data/im1_real.jpeg'
image_path_2 = 'data/im2_real.jpeg'
sift_matches_output_path = 'sift_matches_all.png'
sift_selected_matches_output_path = 'sift_selected_matches_numbered.png'  # NEW
image1_numbered_output_path = 'image1_selected_points_numbered.png'  # NEW
image2_numbered_output_path = 'image2_selected_points_numbered.png'  # NEW
reconstructed_scene_output_path = 'reconstructed_scene_dense.ply'
reconstructed_scene_render_output_path = 'reconstructed_scene_render_numbered.png'  # UPDATED

# --- 1. Load Images ---
print("="*60)
print("3D RECONSTRUCTION USING GOLD STANDARD METHOD")
print("="*60)

try:
    img1_color = cv2.imread(image_path_1)
    img2_color = cv2.imread(image_path_2)
    img1_gray = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

    if img1_color is None or img2_color is None or img1_gray is None or img2_gray is None:
        raise FileNotFoundError("Could not load images. Check paths.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

print("\n[1] Images loaded successfully.")
print(f"    Image 1 shape: {img1_color.shape}")
print(f"    Image 2 shape: {img2_color.shape}")

# --- 2. SIFT Feature Detection and Matching ---
print("\n[2] Running SIFT feature detection and matching...")
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

print(f"    Detected {len(kp1)} keypoints in image 1")
print(f"    Detected {len(kp2)} keypoints in image 2")

# BFMatcher with ratio test (Lowe's ratio test)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"    Found {len(good_matches)} good matches after ratio test.")

# Additional filtering using geometric consistency (optional but recommended)
if len(good_matches) >= 8:
    pts1_test = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2_test = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Use RANSAC to find geometrically consistent matches
    _, inlier_mask = cv2.findHomography(pts1_test, pts2_test, cv2.RANSAC, 5.0)
    
    if inlier_mask is not None:
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
        print(f"    Found {len(inlier_matches)} geometrically consistent matches after RANSAC filtering")
        
        # Use inlier matches for visualization and reconstruction
        good_matches = inlier_matches

# Save SIFT Matches Visualization (show up to 200 best matches)
if len(good_matches) > 0:
    # Sort by distance to show best matches
    good_matches_sorted = sorted(good_matches, key=lambda x: x.distance)
    num_matches_to_show = min(200, len(good_matches_sorted))
    
    img_matches_all = cv2.drawMatches(
        img1_color, kp1, 
        img2_color, kp2, 
        good_matches_sorted[:num_matches_to_show], 
        None,
        matchColor=(0, 255, 0),  # Green lines
        singlePointColor=(255, 0, 0),  # Red points
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(sift_matches_output_path, img_matches_all)
    print(f"    SIFT matches visualization saved to {sift_matches_output_path}")
    print(f"    Showing {num_matches_to_show} best matches out of {len(good_matches)} total")

# --- 3. Extract Correspondence Points ---
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

print(f"\n[3] Extracted {len(pts1)} correspondence point pairs")

# ============================================================================
# ALGORITHM 11.1: NORMALIZED 8-POINT ALGORITHM
# ============================================================================

def normalize_points(pts):
    """
    Normalize points by translating to centroid and scaling.
    Returns normalized points and transformation matrix.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Algorithm 4.2
    """
    # Compute centroid
    centroid = np.mean(pts, axis=0)
    
    # Translate points to origin
    pts_centered = pts - centroid
    
    # Compute average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))
    
    # Scale factor: make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist
    
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_normalized = (T @ pts_homogeneous.T).T
    
    return pts_normalized[:, :2], T

def compute_fundamental_matrix_normalized_8point(pts1, pts2):
    """
    Algorithm 11.1: Normalized 8-Point Algorithm for Fundamental Matrix.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Algorithm 11.1
    
    Args:
        pts1: Nx2 array of points in image 1
        pts2: Nx2 array of points in image 2
    
    Returns:
        F: 3x3 Fundamental matrix
    """
    assert len(pts1) >= 8, "Need at least 8 point correspondences"
    
    # Step 1: Normalize point coordinates
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Step 2: Construct matrix A for the linear system
    N = len(pts1_norm)
    A = np.zeros((N, 9))
    
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Step 3: Solve using SVD (homogeneous least squares)
    # F is the last column of V (corresponding to smallest singular value)
    U, S, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)
    
    # Step 4: Enforce rank-2 constraint (singularity constraint)
    # Set smallest singular value to zero
    U_f, S_f, Vt_f = np.linalg.svd(F_norm)
    S_f[2] = 0  # Enforce rank 2
    F_norm = U_f @ np.diag(S_f) @ Vt_f
    
    # Step 5: Denormalize
    F = T2.T @ F_norm @ T1
    
    # Normalize so that ||F|| = 1
    F = F / np.linalg.norm(F)
    
    return F

# ============================================================================
# ALGORITHM 11.3: GOLD STANDARD METHOD (Geometric Error Minimization)
# ============================================================================

def sampson_error(F, pts1, pts2):
    """
    Compute Sampson distance (first-order geometric error approximation).
    
    Reference: Hartley & Zisserman, Section 11.4.3
    """
    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    
    errors = []
    for i in range(len(pts1)):
        p1 = pts1_h[i]
        p2 = pts2_h[i]
        
        # Epipolar line in image 2: l2 = F * p1
        l2 = F @ p1
        
        # Epipolar line in image 1: l1 = F^T * p2
        l1 = F.T @ p2
        
        # Sampson error
        numerator = (p2.T @ F @ p1) ** 2
        denominator = l2[0]**2 + l2[1]**2 + l1[0]**2 + l1[1]**2
        
        error = numerator / denominator if denominator > 1e-10 else 0
        errors.append(error)
    
    return np.array(errors)

def fundamental_matrix_from_vector(f_vec):
    """Convert 9-element vector to 3x3 matrix and enforce rank-2 constraint."""
    F = f_vec.reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    return F

def vector_from_fundamental_matrix(F):
    """Convert 3x3 matrix to 9-element vector."""
    return F.flatten()

def residuals_for_optimization(f_vec, pts1, pts2):
    """
    Residual function for least squares optimization.
    Returns Sampson errors for all point correspondences.
    """
    F = fundamental_matrix_from_vector(f_vec)
    errors = sampson_error(F, pts1, pts2)
    return np.sqrt(errors)  # Return square root for least_squares

def gold_standard_fundamental_matrix(pts1, pts2, F_init=None):
    """
    Algorithm 11.3: Gold Standard Algorithm for Fundamental Matrix.
    
    Minimizes geometric error (Sampson distance) using iterative refinement.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Algorithm 11.3
    
    Args:
        pts1: Nx2 array of points in image 1
        pts2: Nx2 array of points in image 2
        F_init: Initial estimate of F (if None, uses normalized 8-point)
    
    Returns:
        F_optimal: Optimized 3x3 Fundamental matrix
    """
    # Step 1: Linear solution (initialization)
    if F_init is None:
        F_init = compute_fundamental_matrix_normalized_8point(pts1, pts2)
    
    print("\n[4] Running Gold Standard Algorithm...")
    print(f"    Initial Sampson error: {np.mean(sampson_error(F_init, pts1, pts2)):.6f}")
    
    # Step 2: Minimize geometric error using Levenberg-Marquardt
    f_init = vector_from_fundamental_matrix(F_init)
    
    result = least_squares(
        residuals_for_optimization,
        f_init,
        args=(pts1, pts2),
        method='lm',  # Levenberg-Marquardt
        max_nfev=1000,
        verbose=0
    )
    
    # Step 3: Extract optimized F and enforce rank-2 constraint
    F_optimal = fundamental_matrix_from_vector(result.x)
    
    print(f"    Final Sampson error: {np.mean(sampson_error(F_optimal, pts1, pts2)):.6f}")
    print(f"    Optimization converged: {result.success}")
    
    return F_optimal

# ============================================================================
# RUN THE ALGORITHMS
# ============================================================================

# Apply Normalized 8-Point Algorithm (Algorithm 11.1)
print("\n[4a] Computing Fundamental Matrix using Normalized 8-Point Algorithm...")
F_linear = compute_fundamental_matrix_normalized_8point(pts1, pts2)
print("     Fundamental Matrix (Linear - Algorithm 11.1):")
print(F_linear)

# Apply Gold Standard Method (Algorithm 11.3)
F_gold = gold_standard_fundamental_matrix(pts1, pts2, F_init=F_linear)
print("\n[4b] Fundamental Matrix (Gold Standard - Algorithm 11.3):")
print(F_gold)

# --- 5. Camera Intrinsics ---
# Assuming generic intrinsics (you should calibrate or provide actual values)
focal_length = max(img1_gray.shape) * 1.2  # Heuristic estimate
cx1, cy1 = img1_gray.shape[1] / 2, img1_gray.shape[0] / 2
cx2, cy2 = img2_gray.shape[1] / 2, img2_gray.shape[0] / 2

K1 = np.array([[focal_length, 0, cx1], [0, focal_length, cy1], [0, 0, 1]])
K2 = np.array([[focal_length, 0, cx2], [0, focal_length, cy2], [0, 0, 1]])

print("\n[5] Camera Intrinsic Matrix K:")
print(K1)

# --- 6. Essential Matrix and Pose Recovery ---
print("\n[6] Computing Essential Matrix and recovering camera pose...")
E = K2.T @ F_gold @ K1
print("    Essential Matrix E:")
print(E)

# Recover pose from Essential Matrix
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K1)
print(f"    Recovered rotation R and translation t")
print(f"    Number of points in front of both cameras: {np.sum(mask_pose)}")

# --- 7. Triangulation (High-Quality Points Only) ---
print("\n[7] Selecting HIGH-QUALITY correspondence points for triangulation...")

# Strategy: Use only the BEST matches with lowest reprojection error
# This gives better 3D reconstruction quality

# Define projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, t))

# First, compute reprojection errors for ALL matches using the estimated F matrix
pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

# Compute symmetric epipolar distance for each correspondence
epipolar_errors = []
for i in range(len(pts1)):
    p1 = pts1_h[i]
    p2 = pts2_h[i]
    
    # Distance from point to epipolar line
    l2 = F_gold @ p1  # Epipolar line in image 2
    l1 = F_gold.T @ p2  # Epipolar line in image 1
    
    # Point-to-line distance
    dist1 = np.abs(p2.T @ F_gold @ p1) / np.sqrt(l2[0]**2 + l2[1]**2)
    dist2 = np.abs(p2.T @ F_gold @ p1) / np.sqrt(l1[0]**2 + l1[1]**2)
    
    # Symmetric epipolar distance
    epipolar_errors.append((dist1 + dist2) / 2)

epipolar_errors = np.array(epipolar_errors)

# Select top N points with lowest epipolar error
NUM_SELECTED_POINTS = 50  # Small number of high-quality points
best_indices = np.argsort(epipolar_errors)[:NUM_SELECTED_POINTS]

# Extract selected high-quality correspondences
pts1_selected = pts1[best_indices]
pts2_selected = pts2[best_indices]
selected_matches = [good_matches[i] for i in best_indices]

print(f"    Selected {NUM_SELECTED_POINTS} highest-quality correspondence points")
print(f"    Mean epipolar error of selected points: {np.mean(epipolar_errors[best_indices]):.4f} pixels")
print(f"    Max epipolar error of selected points: {np.max(epipolar_errors[best_indices]):.4f} pixels")

# Triangulate ONLY the selected high-quality points
pts1_triang = pts1_selected.T
pts2_triang = pts2_selected.T

X_homogeneous = cv2.triangulatePoints(P1, P2, pts1_triang, pts2_triang)
X_3D = (X_homogeneous[:3] / X_homogeneous[3]).T

# Filter valid points (reasonable depth)
z_threshold = 0.1
max_z = 50
valid_indices = (X_3D[:, 2] > z_threshold) & (X_3D[:, 2] < max_z)

X_3D_filtered = X_3D[valid_indices]
pts1_final = pts1_selected[valid_indices]
pts2_final = pts2_selected[valid_indices]
point_numbers = np.arange(len(X_3D_filtered))

print(f"    Final 3D points after depth filtering: {len(X_3D_filtered)}")

# --- 8. Visualization with NUMBERING ---
print("\n[8] Creating numbered visualizations...")

# 8a. Numbered matches visualization
img_matches_numbered = cv2.drawMatches(
    img1_color, kp1, 
    img2_color, kp2, 
    [selected_matches[i] for i in range(len(selected_matches)) if valid_indices[i]], 
    None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Add numbers to matched points
for i, match_idx in enumerate(np.where(valid_indices)[0]):
    m = selected_matches[match_idx]
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    
    # Number on image 1 (left side)
    cv2.circle(img_matches_numbered, (int(p1[0]), int(p1[1])), 8, (0, 0, 255), 2)
    cv2.putText(img_matches_numbered, str(i), 
                (int(p1[0]) - 10, int(p1[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Number on image 2 (right side, offset by image1 width)
    p2_offset = (int(p2[0] + img1_color.shape[1]), int(p2[1]))
    cv2.circle(img_matches_numbered, p2_offset, 8, (255, 0, 0), 2)
    cv2.putText(img_matches_numbered, str(i), 
                (p2_offset[0] - 10, p2_offset[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imwrite(sift_selected_matches_output_path, img_matches_numbered)
print(f"    Numbered matches saved to {sift_selected_matches_output_path}")

# 8b. Numbered points on Image 1
img1_numbered = img1_color.copy()
for i in range(len(pts1_final)):
    p = pts1_final[i]
    cv2.circle(img1_numbered, (int(p[0]), int(p[1])), 8, (0, 0, 255), 2)
    cv2.putText(img1_numbered, str(i), 
                (int(p[0]) + 10, int(p[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imwrite(image1_numbered_output_path, img1_numbered)
print(f"    Image 1 with numbered points saved to {image1_numbered_output_path}")

# 8c. Numbered points on Image 2
img2_numbered = img2_color.copy()
for i in range(len(pts2_final)):
    p = pts2_final[i]
    cv2.circle(img2_numbered, (int(p[0]), int(p[1])), 8, (255, 0, 0), 2)
    cv2.putText(img2_numbered, str(i), 
                (int(p[0]) + 10, int(p[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imwrite(image2_numbered_output_path, img2_numbered)
print(f"    Image 2 with numbered points saved to {image2_numbered_output_path}")
if len(X_3D_filtered) > 0:
    print("\n[9] Creating 3D visualization and saving results...")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X_3D_filtered)
    
    # Assign colors based on image 1
    colors = []
    for i in range(len(pts1_final)):
        x, y = int(pts1_final[i, 0]), int(pts1_final[i, 1])
        if 0 <= x < img1_color.shape[1] and 0 <= y < img1_color.shape[0]:
            color = img1_color[y, x] / 255.0  # Normalize to [0, 1]
            colors.append(color[::-1])  # BGR to RGB
        else:
            colors.append([0.5, 0.5, 0.5])  # Gray for out-of-bounds
    
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Save to PLY
    o3d.io.write_point_cloud(reconstructed_scene_output_path, pcd)
    print(f"    3D point cloud saved to {reconstructed_scene_output_path}")
    
    # Create detailed depth report
    print("\n" + "="*70)
    print("DEPTH ANALYSIS FOR NUMBERED POINTS")
    print("="*70)
    print(f"{'Point #':<10} {'X (m)':<12} {'Y (m)':<12} {'Z (m)':<12} {'Depth Rank':<12}")
    print("-"*70)
    
    # Sort by depth (Z coordinate)
    depth_order = np.argsort(X_3D_filtered[:, 2])
    
    for rank, idx in enumerate(depth_order):
        x, y, z = X_3D_filtered[idx]
        print(f"{idx:<10} {x:>11.4f} {y:>11.4f} {z:>11.4f} {rank+1:<12}")
    
    print("-"*70)
    print(f"Closest point: #{depth_order[0]} at Z={X_3D_filtered[depth_order[0], 2]:.4f}m")
    print(f"Farthest point: #{depth_order[-1]} at Z={X_3D_filtered[depth_order[-1], 2]:.4f}m")
    print(f"Mean depth: {np.mean(X_3D_filtered[:, 2]):.4f}m")
    print("="*70)
    
    # Matplotlib visualization with numbers
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors
    ax.scatter(X_3D_filtered[:, 0], X_3D_filtered[:, 1], X_3D_filtered[:, 2], 
               c=colors, s=100, marker='o', edgecolors='black', linewidths=1.5)
    
    # Add numbers to each 3D point
    for i in range(len(X_3D_filtered)):
        ax.text(X_3D_filtered[i, 0], X_3D_filtered[i, 1], X_3D_filtered[i, 2], 
                str(i), fontsize=10, weight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, weight='bold')
    ax.set_zlabel('Z (Depth in m)', fontsize=12, weight='bold')
    ax.set_title(f'3D Reconstructed Scene with Numbered Points ({len(X_3D_filtered)} points)\n'
                 f'Gold Standard Method - High Quality Correspondences Only', 
                 fontsize=14, weight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([X_3D_filtered[:, 0].max()-X_3D_filtered[:, 0].min(),
                          X_3D_filtered[:, 1].max()-X_3D_filtered[:, 1].min(),
                          X_3D_filtered[:, 2].max()-X_3D_filtered[:, 2].min()]).max() / 2.0
    
    mid_x = (X_3D_filtered[:, 0].max()+X_3D_filtered[:, 0].min()) * 0.5
    mid_y = (X_3D_filtered[:, 1].max()+X_3D_filtered[:, 1].min()) * 0.5
    mid_z = (X_3D_filtered[:, 2].max()+X_3D_filtered[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(reconstructed_scene_render_output_path, dpi=300, bbox_inches='tight')
    print(f"\n    Numbered 3D visualization saved to {reconstructed_scene_render_output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("RECONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Algorithm Used: Gold Standard Method (Hartley & Zisserman)")
    print(f"Quality Strategy: Small number of high-quality correspondences")
    print(f"Total 3D points reconstructed: {len(X_3D_filtered)}")
    print(f"\nOutput files:")
    print(f"  1. {sift_matches_output_path} - All SIFT matches")
    print(f"  2. {sift_selected_matches_output_path} - Selected numbered matches")
    print(f"  3. {image1_numbered_output_path} - Image 1 with numbered points")
    print(f"  4. {image2_numbered_output_path} - Image 2 with numbered points")
    print(f"  5. {reconstructed_scene_output_path} - 3D point cloud (PLY)")
    print(f"  6. {reconstructed_scene_render_output_path} - Numbered 3D visualization")
    
else:
    print("\nERROR: No valid 3D points reconstructed!")

# ============================================================================
# REFERENCES AND ACKNOWLEDGMENTS
# ============================================================================
print("\n" + "="*60)
print("REFERENCES:")
print("="*60)
print("1. Hartley, R. and Zisserman, A. (2004)")
print("   'Multiple View Geometry in Computer Vision', 2nd Edition")
print("   - Algorithm 11.1: Normalized 8-Point Algorithm")
print("   - Algorithm 11.3: Gold Standard Algorithm")
print("   Cambridge University Press")
print()
print("2. OpenCV Library (cv2) - used for:")
print("   - SIFT feature detection and matching")
print("   - Image I/O operations")
print("   - Pose recovery (cv2.recoverPose)")
print("   - Triangulation (cv2.triangulatePoints)")
print()
print("3. Open3D Library - used for:")
print("   - Point cloud storage and visualization")
print("="*60)