import cv2
import numpy as np

# Load images
img1 = cv2.imread('data/im2.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/im6.png', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("Check your image paths!")

# SIFT keypoints + descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match features using KNN + Lowe's ratio test
bf = cv2.BFMatcher()
matches_knn = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Create a side-by-side visualization
h1, w1 = img1.shape
h2, w2 = img2.shape
vis = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
vis[:h1, :w1] = img1
vis[:h2, w1:w1 + w2] = img2
vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# Precompute match coordinates for quick lookup
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

# Convert pts2 to shifted coords for right image display
pts2_shifted = pts2.copy()
pts2_shifted[:, 0] += w1

# Radius threshold for selecting points
CLICK_RADIUS = 10

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is in left or right image
        if x < w1:  # Left image click
            distances = np.linalg.norm(pts1 - np.array([x, y]), axis=1)
            idx = np.argmin(distances)
            if distances[idx] < CLICK_RADIUS:
                pt1 = tuple(np.int32(pts1[idx]))
                pt2 = tuple(np.int32(pts2_shifted[idx]))
                temp = vis_color.copy()
                cv2.circle(temp, pt1, 8, (0, 0, 255), -1)
                cv2.circle(temp, pt2, 8, (0, 255, 0), -1)
                cv2.line(temp, pt1, pt2, (255, 0, 0), 2)
                print(f"✅ Match clicked: Left({pt1}) ↔ Right({pts2[idx]})")
                cv2.imshow("SIFT Correspondences", temp)

        else:  # Right image click
            distances = np.linalg.norm(pts2_shifted - np.array([x, y]), axis=1)
            idx = np.argmin(distances)
            if distances[idx] < CLICK_RADIUS:
                pt1 = tuple(np.int32(pts1[idx]))
                pt2 = tuple(np.int32(pts2_shifted[idx]))
                temp = vis_color.copy()
                cv2.circle(temp, pt1, 8, (0, 0, 255), -1)
                cv2.circle(temp, pt2, 8, (0, 255, 0), -1)
                cv2.line(temp, pt1, pt2, (255, 0, 0), 2)
                print(f"✅ Match clicked: Right({pts2[idx]}) ↔ Left({pt1})")
                cv2.imshow("SIFT Correspondences", temp)

# Setup window
cv2.namedWindow("SIFT Correspondences", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SIFT Correspondences", on_mouse)
cv2.imshow("SIFT Correspondences", vis_color)

print("🖱 Click near a feature point to highlight its correspondence.")
print("Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
