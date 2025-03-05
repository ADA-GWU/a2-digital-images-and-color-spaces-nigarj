import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

# Load the image
image_path = "redgirlbib.jpeg"  # Change this to your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
else:
    # Convert BGR to RGB for proper visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # ========== Adjust Hue, Saturation, Brightness & Lightness ==========
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Modify parameters (tweak values as needed)
    hue_shift = 20       # Adjust Hue
    sat_factor = 1.2     # Saturation Increase
    val_factor = 1.1     # Brightness Increase

    # Apply modifications
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat_factor, 0, 255).astype(np.uint8)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val_factor, 0, 255).astype(np.uint8)

    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(modified_image)
    axes[1].set_title("Hue/Saturation/Brightness Adjusted")
    plt.show()