import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
image_path_1 = 'data/im2.png'
image_path_2 = 'data/im6.png'
sift_matches_output_path = 'sift_matches_all.png' # All matches
sift_selected_matches_output_path = 'sift_selected_matches_numbered.png' # NEW: Selected & numbered matches
reconstructed_scene_output_path = 'reconstructed_scene.ply'
reconstructed_scene_render_output_path = 'reconstructed_scene_render_numbered.png' # NEW: Numbered 3D render
image1_numbered_output_path = 'image1_selected_points_numbered.png' # NEW: Image 1 with numbered points
image2_numbered_output_path = 'image2_selected_points_numbered.png' # NEW: Image 2 with numbered points

# --- 1. Load Images ---
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

print("Images loaded successfully.")

# --- 2. SIFT Feature Detection and Matching ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Found {len(good_matches)} good matches after ratio test.")

# Save ALL SIFT Matches Image
if len(good_matches) > 0:
    img_matches_all = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(sift_matches_output_path, img_matches_all)
    print(f"All SIFT matches visualization saved to {sift_matches_output_path}")

# --- 3. Extract ALL Correspondence Points for Fundamental Matrix Estimation ---
pts1_all = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts2_all = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# --- 4. Fundamental Matrix Estimation (Using OpenCV's RANSAC for Robustness) ---
# It's better to use RANSAC to find F, as it's robust to outliers.
# This also gives us a mask to identify inlier matches.
if len(pts1_all) >= 8:
    F, fundamental_mask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.0, 0.99) # Threshold 1.0, confidence 0.99
    print("\nFundamental Matrix (OpenCV RANSAC):\n", F)

    # --- Select a SMALL number of HIGH-QUALITY correspondence points ---
    # Filter points based on the fundamental_mask (RANSAC inliers)
    inlier_matches = []
    if fundamental_mask is not None:
        for i, m in enumerate(good_matches):
            if fundamental_mask[i] == 1:
                inlier_matches.append(m)
        print(f"Found {len(inlier_matches)} inlier matches after Fundamental Matrix RANSAC.")

        # Now, select a small, fixed number of these high-quality inliers
        # You can adjust 'num_selected_points' based on your preference
        num_selected_points = min(50, len(inlier_matches)) # Max 50 points, or fewer if not enough inliers
        
        # Shuffle and select to get a diverse set, or pick deterministically
        np.random.seed(42) # For reproducibility
        selected_indices = np.random.choice(len(inlier_matches), num_selected_points, replace=False)
        selected_matches = [inlier_matches[i] for i in selected_indices]
        
        pts1_selected = np.float32([kp1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 2)
        pts2_selected = np.float32([kp2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 2)

        print(f"Selected {len(pts1_selected)} high-quality correspondence points for 3D reconstruction.")
    else:
        print("RANSAC mask is None. Cannot select high-quality points.")
        pts1_selected = np.array([])
        pts2_selected = np.array([])
        F = None # If RANSAC failed, F might be unreliable
else:
    print("Not enough good matches to compute Fundamental Matrix or select points.")
    pts1_selected = np.array([])
    pts2_selected = np.array([])
    F = None

# Ensure we have selected points for the next steps
if len(pts1_selected) == 0 or F is None:
    print("Exiting: No selected points or Fundamental Matrix is unavailable.")
    exit()

# --- 5. Visualize Selected SIFT Matches with Numbers ---
img_matches_selected = cv2.drawMatches(img1_color, kp1, img2_color, kp2, selected_matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Add numbers to the selected points on the match visualization
for i, m in enumerate(selected_matches):
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    
    # Draw number on image 1 side
    cv2.putText(img_matches_selected, str(i), (int(p1[0]), int(p1[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Red for Image 1 points
    # Draw number on image 2 side (remember to offset x by image width)
    cv2.putText(img_matches_selected, str(i), (int(p2[0] + img1_color.shape[1]), int(p2[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Blue for Image 2 points

cv2.imwrite(sift_selected_matches_output_path, img_matches_selected)
print(f"Selected SIFT matches with numbers saved to {sift_selected_matches_output_path}")

plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(img_matches_selected, cv2.COLOR_BGR2RGB))
plt.title(f"Selected SIFT Matches ({len(selected_matches)} points)")
plt.show()

# --- NEW: Save Image 1 and Image 2 with Numbered Points ---
img1_numbered = img1_color.copy()
img2_numbered = img2_color.copy()

for i in range(len(pts1_selected)):
    p1 = pts1_selected[i]
    p2 = pts2_selected[i]

    # Draw circle and text on img1
    cv2.circle(img1_numbered, (int(p1[0]), int(p1[1])), 5, (0, 0, 255), -1) # Red circle
    cv2.putText(img1_numbered, str(i), (int(p1[0]) + 7, int(p1[1]) - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw circle and text on img2
    cv2.circle(img2_numbered, (int(p2[0]), int(p2[1])), 5, (255, 0, 0), -1) # Blue circle
    cv2.putText(img2_numbered, str(i), (int(p2[0]) + 7, int(p2[1]) - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imwrite(image1_numbered_output_path, img1_numbered)
cv2.imwrite(image2_numbered_output_path, img2_numbered)
print(f"Image 1 with numbered points saved to {image1_numbered_output_path}")
print(f"Image 2 with numbered points saved to {image2_numbered_output_path}")

# --- 6. Camera Intrinsics (Adjust if necessary) ---
# Assuming generic intrinsics based on image size.
focal_length = 700
cx1, cy1 = img1_gray.shape[1] / 2, img1_gray.shape[0] / 2
cx2, cy2 = img2_gray.shape[1] / 2, img2_gray.shape[0] / 2

K1 = np.array([[focal_length, 0, cx1], [0, focal_length, cy1], [0, 0, 1]])
K2 = np.array([[focal_length, 0, cx2], [0, focal_length, cy2], [0, 0, 1]])

print("\nAssumed Camera Intrinsic Matrix K1:\n", K1)
print("Assumed Camera Intrinsic Matrix K2:\n", K2)

# --- 7. Recover Pose (R, t) from Essential Matrix (derived from F and K) ---
# Ensure F is not None before proceeding
E, _ = cv2.findEssentialMat(pts1_selected, pts2_selected, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
points, R, t, mask_pose = cv2.recoverPose(E, pts1_selected, pts2_selected, K1)

print("\nRecovered Rotation Matrix R:\n", R)
print("Recovered Translation Vector t:\n", t.T)

# --- 8. Define Camera Projection Matrices ---
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, t))

# --- 9. Triangulate 3D Points for the SELECTED points ---
pts1_triang = pts1_selected.T
pts2_triang = pts2_selected.T

X_homogeneous = cv2.triangulatePoints(P1, P2, pts1_triang, pts2_triang)
X_3D_raw = (X_homogeneous[:3] / X_homogeneous[3]).T

# Filter out points behind cameras or at infinity
z_threshold = 0.1 # Minimum Z depth
# Also filter for points whose Z is not excessively large (bad triangulations can happen)
max_z = 100 # Adjust this based on expected scene scale
valid_indices = (X_3D_raw[:, 2] > z_threshold) & (X_3D_raw[:, 2] < max_z)
X_3D_filtered = X_3D_raw[valid_indices]
selected_point_numbers = np.arange(len(pts1_selected))[valid_indices] # Keep track of original point numbers

print(f"Reconstructed {len(X_3D_filtered)} 3D points from selected high-quality matches.")

# --- 10. Plotting and Saving the 3D Scene with Numbered Points ---
if len(X_3D_filtered) > 0:
    # --- Matplotlib Plot with Numbers ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_3D_filtered[:, 0], X_3D_filtered[:, 1], X_3D_filtered[:, 2], s=50, c='red', marker='o')

    # Add numbers to the 3D points
    for i, (point, num) in enumerate(zip(X_3D_filtered, selected_point_numbers)):
        ax.text(point[0], point[1], point[2], str(num), color='black', fontsize=10, weight='bold')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Reconstructed Point Cloud (Numbered)')
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

    # --- Save to PLY File (using Open3D) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X_3D_filtered)
    o3d.io.write_point_cloud(reconstructed_scene_output_path, pcd)
    print(f"3D point cloud saved to {reconstructed_scene_output_path}")

    # --- Save Reconstructed Scene Image (Render using Open3D with Numbers) ---
    # Open3D doesn't natively support rendering text labels on 3D points in the visualizer
    # in a way that's easily captured off-screen.
    # So, we'll save the Matplotlib plot as the "rendered scene image with numbers".
    # For a higher quality render, you might export the PLY and use a dedicated 3D software (Blender, MeshLab).
    fig.savefig(reconstructed_scene_render_output_path, dpi=300)
    print(f"Numbered 3D scene render saved to {reconstructed_scene_render_output_path} (from Matplotlib).")

else:
    print("\nNo valid 3D points to plot, save to PLY file, or render a scene image.")