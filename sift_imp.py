import cv2

# Load images
img1 = cv2.imread('data/im2.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/im6.png', cv2.IMREAD_GRAYSCALE)


if img1 is None or img2 is None:
    raise FileNotFoundError("Check your image paths!")

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show result interactively
cv2.imshow("SIFT Correspondences", matched_img)

print("👉 Press any key on the image window to close it.")
cv2.waitKey(0)      # Wait for any key press
cv2.destroyAllWindows()
