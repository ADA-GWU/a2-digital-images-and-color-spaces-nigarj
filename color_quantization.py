import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans
import os

# Function to create the output folder if it doesn't exist
def create_output_folder():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Loading the image
image_path = "input_images/adacenter.jpeg"  # Change this to desired image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image")
else:
    # Convert BGR to RGB for proper visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Function for K-Means Quantization
    def kmeans_quantization(img, K):
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=K, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        quantized_img = centers[labels].reshape(img.shape)
        return quantized_img

    # Function for Uniform Quantization
    def uniform_quantization(img, levels):
        scale = 256 // levels
        return (img // scale * scale).astype(np.uint8)

    # Apply Quantizations
    quant_kmeans = kmeans_quantization(image_rgb, K=5)  # You can adjust the k here
    quant_uniform = uniform_quantization(image_rgb, levels=5) # You can adjust the levels here
    quant_median = cv2.xphoto.applyChannelGains(image_rgb, 0.5, 0.5, 0.5)  

    # Display the quantized images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(quant_kmeans)
    axes[1].set_title("K-Means Quantization")
    axes[2].imshow(quant_median)
    axes[2].set_title("Median Cut Approx.")
    axes[3].imshow(quant_uniform)
    axes[3].set_title("Uniform Quantization")
    plt.show()

    # Automatically save the quantized images
    output_dir = create_output_folder()

    # Saving the quantized images with logical filenames
    cv2.imwrite(f"{output_dir}/original_image.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/kmeans_quantization.png", cv2.cvtColor(quant_kmeans, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/median_cut_quantization.png", cv2.cvtColor(quant_median, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/uniform_quantization.png", cv2.cvtColor(quant_uniform, cv2.COLOR_RGB2BGR))

    print("Quantized images saved in 'output' folder")
