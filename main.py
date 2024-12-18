import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

image = cv2.imread('morocco mosque.jpg', cv2.IMREAD_GRAYSCALE)

radius = 1
n_points = 8 * radius

lbp = local_binary_pattern(image, radius, n_points, method='uniform')

lbp_histogram, _ = np.histogram(lbp.ravel(), bins= np.arange(0, n_points+3), range=(0,n_points+2))
lbp_histogram = lbp_histogram.astype(float)
lbp_histogram /= (lbp_histogram.sum() + 0.0001)

fx,axes = plt.subplots(1, 2, figsize=(12, 6))

# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Original Image")
# axes[0].axis("off")
#
# axes[1].imshow(image, cmap='gray')

# axes[1].set_title("LBP Image")
# axes[1].axis("off")

plt.figure(figsize=(8,6))
plt.bar(np.arange(0, len(lbp_histogram)), lbp_histogram, color='blue')
plt.title("Local Binray Patterns")
plt.xlabel("LBP Prototype #")
plt.ylabel("% of Pixels")
plt.show()