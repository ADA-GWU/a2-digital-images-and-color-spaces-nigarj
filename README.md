# Overview

This project demonstrates various color manipulations on images. The following functionalities are implemented:

Grayscale Conversion: Convert a color image to grayscale using proper RGB coefficients and compare it with an averaging method.

Color Quantization: Reduce the number of colors in an image using 3 different quantization techniques.

Hue, Saturation, Brightness, and Lightness Adjustments: Modifying color properties while maintaining valid ranges.

CIEDE Color Closeness Algorithm: Identifies similar colors using the DeltaE metric based on a selected reference color.


## Folder Structure
```
📂 input_images        # Folder for original images
📂 output_grayscale    # Output for grayscale conversion
📂 output_color_quantization  # Output for color quantization
📂 output_hsb         # Output for HSB adjustments
📂 output_color_closeness  # Output for color similarity detection
📜 grayscale.py       # Script for grayscale conversion
📜 color_quantization.py  # Script for color quantization
📜 hsb.py             # Script for HSB adjustment
📜 color_closeness.py # Script for color similarity detection
📜 README.md          # This documentation
```

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn
```

## Scripts Description & Usage

### 1. Grayscale Conversion (`grayscale.py`)
This script converts an image to grayscale using two methods:
- **Weighted method** (NTSC formula)
- **Averaging method** (equal weights for RGB channels)

It then calculates the **Mean Squared Error (MSE)** between the two results.

**Usage:**
```bash
python grayscale.py
```

### 2. Color Quantization (`color_quantization.py`)
This script applies three color quantization techniques:
- **K-Means clustering** (default: 5 clusters)
- **Uniform quantization** (default: 5 levels)
- **Median cut approximation**

**Usage:**
```bash
python color_quantization.py
```

### 3. Hue, Saturation, Brightness, and Lightness Adjustment (`hsb.py`)
This script provides an interactive **Matplotlib UI** to adjust Hue, Saturation, Brightness, and Lightness values.
- Uses sliders to modify the image in real-time.
- Allows resetting values to original settings.
- Saves the final adjusted image upon closing the window.

**Usage:**
```bash
python hsb.py
```

### 4. Color Closeness Detection (`color_closeness.py`)
This script uses **CIEDE2000** to find similar colors in an image.
- Click on any part of the image to select a reference color.
- The script highlights all pixels that are within a set **DeltaE threshold** (default: 20).

**Usage:**
```bash
python color_closeness.py
```

## Output Files
Each script saves the modified images in its respective output folder (`output_*`).

## Notes
- Ensure all images are stored in the `input_images/` folder.
- Adjust parameters in the scripts if needed.
- When running `color_closeness.py`, a **mouse click** on the displayed image is required to pick a reference color. If you use touchpad make sure you actually click, not just tap.




