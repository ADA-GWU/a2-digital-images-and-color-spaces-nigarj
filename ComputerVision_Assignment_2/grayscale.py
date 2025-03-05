import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

# Load the image
image_path = "redgirlbib.jpeg" 
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
else:
    # Convert BGR to RGB for proper visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ========== Convert to Grayscale (Weighted & Averaging) ==========
    gray_weighted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Uses NTSC formula
    gray_avg = np.mean(image, axis=2).astype(np.uint8)  # Simple average method

    # Calculate difference (Quantitative Comparison)
    difference = np.abs(gray_weighted.astype(np.float32) - gray_avg.astype(np.float32))
    mse = np.mean(difference ** 2)
    print(f"Mean Squared Error (MSE) between the methods: {mse:.2f}")

    # Display grayscale images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(gray_weighted, cmap='gray')
    axes[1].set_title("Grayscale (Weighted)")
    axes[2].imshow(gray_avg, cmap='gray')
    axes[2].set_title("Grayscale (Averaging)")
    plt.show()
