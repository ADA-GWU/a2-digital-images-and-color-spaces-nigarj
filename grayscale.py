import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to create the output folder if it doesn't exist
def create_output_folder(task_name):
    output_dir = f"output_{task_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Loading the image
image_path = "input_images/yellowcar.jpeg"  # Change this to desired image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
else:
    # Convert BGR to RGB for proper visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale (Weighted & Averaging)
    gray_weighted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Uses NTSC formula
    gray_avg = np.mean(image, axis=2).astype(np.uint8)  # Simple average method

    # Quantitative Comparison
    difference = np.abs(gray_weighted.astype(np.float32) - gray_avg.astype(np.float32))
    mse = np.mean(difference ** 2)

    # Display grayscale images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gray_weighted, cmap='gray')
    axes[1].set_title("Grayscale (Weighted)")
    axes[1].axis("off")

    axes[2].imshow(gray_avg, cmap='gray')
    axes[2].set_title("Grayscale (Averaging)")
    axes[2].axis("off")

    # Display MSE text
    fig.text(0.5, 0.02, f"Mean Squared Error (MSE): {mse:.2f}", ha="center", fontsize=12, color='red')

    # Automatically save images
    output_dir = create_output_folder("grayscale")
    cv2.imwrite(f"{output_dir}/grayscale_weighted.png", gray_weighted)
    cv2.imwrite(f"{output_dir}/grayscale_avg.png", gray_avg)

    print("Images saved in 'output' folder")

    plt.show()
