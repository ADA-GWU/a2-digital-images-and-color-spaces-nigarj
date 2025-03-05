import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab

# Load the image
image_path = "flower.jpeg"  # Change this to your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
    exit()

# Convert BGR to RGB for proper visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to handle mouse clicks and get pixel color
def mouse_callback(event, x, y, flags, param):
    global ref_color, ref_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pixel = (x, y)  # Store the position of the clicked pixel
        ref_color = image[y, x]  # Get the BGR color of the clicked pixel
        ref_color = cv2.cvtColor(np.uint8([[ref_color]]), cv2.COLOR_BGR2LAB)[0][0]  # Convert to LAB
        print(f"Reference color at ({x}, {y}): {ref_color}")
        
        # Calculate DeltaE for the entire image with the selected reference color
        delta_e_map = np.linalg.norm(lab_image - ref_color, axis=2)  # Vectorized DeltaE calculation

        # Define threshold for similar colors
        threshold = 230  # You can try adjusting this threshold to get more visible results
        mask = delta_e_map < threshold

        # Highlight similar colors with transparency (blue in this case)
        highlighted_image = image_rgb.copy()
        highlighted_image[mask] = [0, 0, 255]  # Blue color for similar pixels (BGR format)

        # Add opacity (overlay effect) for highlighting
        alpha = 0.5  # Adjust alpha for transparency
        overlay = highlighted_image * alpha + image_rgb * (1 - alpha)

        # Display the updated result with the overlay
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title("Color Closeness Highlighted")
        plt.show()

# Convert image to Lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

cv2.imshow('Click on the reference pixel', image)
cv2.setMouseCallback('Click on the reference pixel', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
