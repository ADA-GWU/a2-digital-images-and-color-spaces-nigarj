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

    # ==========  Color Quantization (K-Means, Median Cut, Uniform) ==========
    
    # Function for K-Means Quantization
    def kmeans_quantization(img, K=8):
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=K, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        quantized_img = centers[labels].reshape(img.shape)
        return quantized_img

    # Function for Uniform Quantization
    def uniform_quantization(img, levels=8):
        scale = 256 // levels
        return (img // scale * scale).astype(np.uint8)

    # Apply K-Means, Median Cut (using OpenCV), and Uniform Quantization
    quant_kmeans = kmeans_quantization(image_rgb, K=8)
    quant_uniform = uniform_quantization(image_rgb, levels=8)
    quant_median = cv2.xphoto.applyChannelGains(image_rgb, 0.5, 0.5, 0.5)  # Approximate Median Cut

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
