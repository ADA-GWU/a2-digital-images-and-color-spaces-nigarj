import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab

# Loading the image
image_path = "input_images/adacenter.jpeg"  # Change this to desired image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
    exit()

# Convert BGR to RGB for proper visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to Lab color space
lab_image = rgb2lab(image_rgb)  

# Function for mouse clicks and get pixel color
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_color = lab_image[y, x]  # Get the LAB color of the clicked pixel
        print(f"Reference color at ({x}, {y}): {ref_color}")

        # Vectorized DeltaE calculation using broadcasting (for quick calclation)
        ref_color_reshaped = ref_color.reshape(1, 1, 3) 
        # Compute DeltaE for all pixels 
        delta_e_map = deltaE_ciede2000(ref_color_reshaped, lab_image)  

        # Threshold for similar colors
        threshold = 20  # You can adjust this for sensitivity
        mask = delta_e_map < threshold

        # Highlight similar colors
        highlighted_image = image_rgb.copy()
        highlighted_image[mask] = [0, 0, 255]  # Blue color for similar pixels (RGB format)

        # Add opacity (overlay effect)
        alpha = 0.5
        overlay = (highlighted_image * alpha + image_rgb * (1 - alpha)).astype(np.uint8)

        # Display the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[1].imshow(overlay)
        axes[1].set_title("Color Closeness Highlighted")
        plt.show()

# Display the image in an OpenCV window
cv2.imshow('Click on the reference pixel', image)
cv2.setMouseCallback('Click on the reference pixel', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()