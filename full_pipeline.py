#!/usr/bin/env python3
"""
reconstruct_gold_standard.py
Usage:
    python reconstruct_gold_standard.py image1.jpg image2.jpg
"""

import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------
# Utils: Normalized 8-point
# ----------------------------
def normalize_points(pts):
    """
    pts: Nx2
    returns pts_norm Nx2 and 3x3 transform T so that pts_hom_norm = T * pts_hom
    """
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    dists = np.sqrt(np.sum(centered**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist == 0:
        s = 1.0
    else:
        s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    pts_n = (T @ pts_h.T).T
    return pts_n[:, :2], T

def eight_point_normalized(pts1, pts2):
    """
    Normalized 8-point algorithm. pts1, pts2 are Nx2 arrays of corresponding points.
    Returns fundamental matrix 3x3 (rank-2 enforced).
    """
    if pts1.shape[0] < 8:
        raise ValueError("Need at least 8 points for the 8-point algorithm.")
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)
    N = pts1_n.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    _, _, Vt = np.linalg.svd(A)
    F_n = Vt[-1].reshape(3,3)
    # Enforce rank-2
    U, S, Vt2 = np.linalg.svd(F_n)
    S[-1] = 0
    F_n = U @ np.diag(S) @ Vt2
    # Denormalize
    F = T2.T @ F_n @ T1
    return F / np.linalg.norm(F)

# ----------------------------
# RANSAC wrapper around 8-pt
# ----------------------------
def ransac_fundamental(pts1, pts2, iters=2000, tol=1e-2):
    N = pts1.shape[0]
    bestF = None
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        ids = np.random.choice(N, 8, replace=False)
        try:
            F_candidate = eight_point_normalized(pts1[ids], pts2[ids])
        except Exception:
            continue
        # Sampson distance for all points
        ones = np.ones((N,1))
        p1 = np.hstack([pts1, ones])
        p2 = np.hstack([pts2, ones])
        # Epipolar lines
        Fx1 = (F_candidate @ p1.T).T       # lines in image2
        Ftx2 = (F_candidate.T @ p2.T).T    # lines in image1
        # Sampson distance
        num = np.sum(p2 * Fx1, axis=1)**2
        denom = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2
        denom[denom==0] = 1e-12
        sd = num / denom
        inliers = sd < tol
        cnt = np.sum(inliers)
        if cnt > best_count:
            best_count = cnt
            bestF = F_candidate
            best_inliers = inliers
    return bestF, best_inliers

# ----------------------------
# Compute camera matrices from F
# ----------------------------
def compute_camera_matrices_from_F(F):
    """
    P1 = [I|0], P2 = [ [e']_x F | e']  (Hartley & Zisserman)
    """
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    # Right epipole (null space of F^T)
    U, S, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]
    e2 = e2 / e2[2]
    e2x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    P2 = np.hstack([e2x @ F, e2.reshape(3,1)])
    return P1, P2

# ----------------------------
# Linear triangulation (DLT)
# ----------------------------
def linear_triangulation(P1, P2, pts1, pts2):
    """
    pts1, pts2 are Nx2
    returns Nx3 3D points in homogeneous -> non-homog coordinates
    """
    N = pts1.shape[0]
    X = np.zeros((N, 3))
    for i in range(N):
        x1 = pts1[i]
        x2 = pts2[i]
        A = np.vstack([
            x1[0] * P1[2,:] - P1[0,:],
            x1[1] * P1[2,:] - P1[1,:],
            x2[0] * P2[2,:] - P2[0,:],
            x2[1] * P2[2,:] - P2[1,:]
        ])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        Xh = Xh / Xh[3]
        X[i] = Xh[:3]
    return X

# ----------------------------
# Bundle adjust: optimize P2 (12 params) + 3D points (3N)
# minimize reprojection error in both images. P1 fixed as canonical.
# ----------------------------
def pack_params(P2, X):
    return np.hstack([P2.ravel(), X.ravel()])

def unpack_params(params, n_points):
    P2 = params[:12].reshape(3,4)
    X = params[12:].reshape(n_points, 3)
    return P2, X

def reprojection_residuals(params, P1, pts1, pts2):
    n = pts1.shape[0]
    P2, X = unpack_params(params, n)
    # project
    Xh = np.hstack([X, np.ones((n,1))])
    proj1 = (P1 @ Xh.T).T
    proj2 = (P2 @ Xh.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    res = np.hstack([proj1 - pts1, proj2 - pts2]).ravel()
    return res

def bundle_adjustment(P1, P2_init, X_init, pts1, pts2, max_nfev=2000):
    n = pts1.shape[0]
    params0 = pack_params(P2_init, X_init)
    def fun(p): return reprojection_residuals(p, P1, pts1, pts2)
    res = least_squares(fun, params0, method='lm', max_nfev=max_nfev, verbose=2)
    P2_opt, X_opt = unpack_params(res.x, n)
    return P2_opt, X_opt

# ----------------------------
# Full pipeline
# ----------------------------
def full_pipeline(img1_path, img2_path, use_top_matches=500):
    # 1) SIFT + match
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)
    if img1_color is None or img2_color is None:
        raise FileNotFoundError("Couldn't load one of the images.")
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # KNN + ratio
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)
    if len(good) < 8:
        raise ValueError("Not enough good matches found - need at least 8.")
    good = good[:use_top_matches]  # limit

    pts1 = np.array([kp1[m.queryIdx].pt for m in good])
    pts2 = np.array([kp2[m.trainIdx].pt for m in good])

    print(f"Matches (after ratio test): {len(pts1)}")

    # 2) Estimate F with RANSAC + normalized 8-point
    F_init, inliers = ransac_fundamental(pts1, pts2, iters=2000, tol=1e-3)
    if F_init is None:
        raise RuntimeError("RANSAC failed to find F.")
    inliers = inliers.astype(bool)
    print(f"RANSAC inliers: {np.sum(inliers)} / {len(inliers)}")

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # refine F on inliers with normalized 8-point (no normalization bug)
    F_refined = eight_point_normalized(pts1_in, pts2_in)

    # 3) Compute camera matrices
    P1, P2_init = compute_camera_matrices_from_F(F_refined)

    # 4) Triangulate
    X_init = linear_triangulation(P1, P2_init, pts1_in, pts2_in)

    # 5) Bundle adjustment (Gold-Standard style) — refine P2 and X to minimize reprojection error
    print("Running bundle adjustment (refining P2 and 3D points)... This may take a while.")
    P2_opt, X_opt = bundle_adjustment(P1, P2_init, X_init, pts1_in, pts2_in, max_nfev=1000)
    print("Bundle adjustment finished.")

    # 6) Plot 3D point cloud
    return X_opt, pts1_in, pts2_in, img1_color, img2_color, P1, P2_opt

# ----------------------------
# Plotting
# ----------------------------
def plot_3d_cloud(X, show=True, savepath=None):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Reconstruction (Gold-Standard refined)")
    # nice equal aspect (approx)
    max_range = np.ptp(X, axis=0).max()
    mid = np.mean(X, axis=0)
    ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
    ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
    ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Saved 3D plot to {savepath}")
    if show:
        plt.show()

def save_ply(filename, points, colors=None):
    """
    Save Nx3 points (+ optional Nx3 uint8 colors) to a PLY file.
    """
    n_points = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    with open(filename, 'w') as f:
        f.write("\n".join(header) + "\n")
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z}\n")
    print(f"[✓] Saved PLY point cloud to: {filename}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python reconstruct_gold_standard.py path/to/image1 path/to/image2")
        sys.exit(1)
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    np.random.seed(0)
    X_opt, pts1_in, pts2_in, _, _, P1, P2_opt = full_pipeline(img1_path, img2_path, use_top_matches=600)
    print(f"Reconstructed {X_opt.shape[0]} points.")
    plot_3d_cloud(X_opt, show=True, savepath="pointcloud.png")
    # Optional: save as PLY (without color)
    save_ply("reconstruction.ply", X_opt)

