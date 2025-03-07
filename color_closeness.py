import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab
import os

# Function to create the output folder if it doesn't exist
def create_output_folder(task_name):
    output_dir = f"output_{task_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Load the image
image_path = "input_images/bikeshare.jpeg"  # Change this to desired image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
    exit()

# Resize image for consistent display
screen_width, screen_height = 800, 600  
scale_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])  # Maintain aspect ratio
display_image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

# Convert BGR to RGB for proper visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

# Convert image to Lab color space
lab_image = rgb2lab(image_rgb)

# Function for mouse clicks and get pixel color
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates back to the original image scale
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        ref_color = lab_image[orig_y, orig_x]  # Get the LAB color of the clicked pixel
        print(f"Reference color at ({orig_x}, {orig_y}): {ref_color}")

        # Vectorized DeltaE calculation using broadcasting
        ref_color_reshaped = ref_color.reshape(1, 1, 3)
        delta_e_map = deltaE_ciede2000(ref_color_reshaped, lab_image)

        # Threshold for similar colors
        threshold = 20  # Adjust this for sensitivity
        mask = delta_e_map < threshold

        # Highlight similar colors
        highlighted_image = image_rgb.copy()
        highlighted_image[mask] = [0, 0, 255]  # Blue color for similar pixels

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

        # Save the overlay image
        output_dir = create_output_folder("color_closeness")
        cv2.imwrite(f"{output_dir}/highlighted_image.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Highlighted image saved in 'output' folder")

# Create a resizable window and display image
cv2.namedWindow('Click on the reference pixel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Click on the reference pixel', display_image.shape[1], display_image.shape[0])
cv2.imshow('Click on the reference pixel', display_image)
cv2.setMouseCallback('Click on the reference pixel', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()