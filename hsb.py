import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

# Function to create the output folder if it doesn't exist
def create_output_folder():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Loading the image
image_path = "input_images/adacenter.jpeg"
original_bgr = cv2.imread(image_path)

# Convert images to different color spaces
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)  
original_hls = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)

# UI SETUP: FIGURE & IMAGE DISPLAY
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, right=0.95, bottom=0.1, top=0.9)

# Display the original image
image_display = ax.imshow(original_rgb)
ax.axis("off")

# UI CONTROLS: SLIDERS
slider_positions = {
    "hue": [0.03, 0.6, 0.2, 0.03],
    "saturation": [0.03, 0.5, 0.2, 0.03],
    "brightness": [0.03, 0.4, 0.2, 0.03],
    "lightness": [0.03, 0.3, 0.2, 0.03]
}

# Create the sliders
ax_hue = fig.add_axes(slider_positions["hue"])
slider_hue = Slider(ax=ax_hue, label=" Hue", valmin=-180, valmax=180, valinit=0)

slider_saturation = Slider(ax=fig.add_axes(slider_positions["saturation"]), label="S", valmin=0.1, valmax=3, valinit=1)
slider_brightness = Slider(ax=fig.add_axes(slider_positions["brightness"]), label="B", valmin=0.1, valmax=3, valinit=1)
slider_lightness = Slider(ax=fig.add_axes(slider_positions["lightness"]), label="L", valmin=0.1, valmax=3, valinit=1)

# FUNCTION TO CHANGE SLIDER COLOR
def update_slider_color(hue_value):
    """
    Converts the given Hue value (-180 to 180) into an RGB color and updates the slider's track color.
    """
    hue_opencv = (hue_value + 180) % 180  # Convert to OpenCV's 0-180 range

    # Convert HLS -> RGB using OpenCV (Full saturation and mid lightness)
    hls_color = np.uint8([[[hue_opencv, 127, 255]]])  # (H, L, S)
    rgb_color = cv2.cvtColor(hls_color, cv2.COLOR_HLS2RGB)[0][0] / 255.0  # Normalize to (0,1) for Matplotlib

    # Apply the color to the slider track
    slider_hue.poly.set_color(rgb_color)
    fig.canvas.draw_idle()

# IMAGE PROCESSING FUNCTION
def apply_color_adjustments(hue_val, saturation_val, brightness_val, lightness_val):
    modified_hls = original_hls.copy()

    # Apply Hue shift and wrap it within the valid range
    modified_hls[:, :, 0] = np.mod(modified_hls[:, :, 0] + hue_val, 180)

    # Apply Saturation and Lightness scaling
    modified_hls[:, :, 1] = np.clip(modified_hls[:, :, 1] * lightness_val, 0, 255)
    modified_hls[:, :, 2] = np.clip(modified_hls[:, :, 2] * saturation_val, 0, 255)

    # Convert back to RGB
    image_bgr = cv2.cvtColor(modified_hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply brightness
    final_image = np.clip(image_rgb * brightness_val, 0, 255).astype(np.uint8)

    return final_image

# CALLBACK FUNCTION
def update(val):
    hue_val = slider_hue.val
    saturation_val = slider_saturation.val
    brightness_val = slider_brightness.val
    lightness_val = slider_lightness.val

    adjusted_image = apply_color_adjustments(hue_val, saturation_val, brightness_val, lightness_val)
    image_display.set_data(adjusted_image)

    # Update the slider track color
    update_slider_color(hue_val)

    fig.canvas.draw_idle()

    # Save the adjusted image
    save_adjusted_image(hue_val, saturation_val, brightness_val, lightness_val, adjusted_image)

# Connect sliders to update function
slider_hue.on_changed(update)
slider_saturation.on_changed(update)
slider_brightness.on_changed(update)
slider_lightness.on_changed(update)

# RESET BUTTON
reset_button_ax = fig.add_axes([0.03, 0.15, 0.2, 0.05])
reset_button = Button(reset_button_ax, "Reset", color='lightgray', hovercolor='blue')

def reset_all(event):
    slider_hue.reset()
    slider_saturation.reset()
    slider_brightness.reset()
    slider_lightness.reset()
    image_display.set_data(original_rgb)
    fig.canvas.draw_idle()

reset_button.on_clicked(reset_all)

# Function to save the adjusted image
def save_adjusted_image(hue_val, saturation_val, brightness_val, lightness_val, adjusted_image):
    output_dir = create_output_folder()

    # Generate a filename based on the current slider values
    filename = f"output/hue_{hue_val}_sat_{saturation_val}_bright_{brightness_val}_light_{lightness_val}.png"

    # Save the adjusted image
    cv2.imwrite(filename, cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR))
    print(f"Image saved as: {filename}")

plt.show()
