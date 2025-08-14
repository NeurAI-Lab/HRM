import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the .npy file
images = np.load('/home/pbhat1/projects/NeurAI/HRM/data/sudoku-extreme-1k-aug-1000/train/all__images.npy')

# Save the first image
first_image = images[0]  # Get the first image

# Ensure the image is in uint8 format if necessary
first_image = (first_image * 255).astype(np.uint8)

# Rearrange dimensions from (C, H, W) to (H, W, C)
first_image = first_image.transpose(1, 2, 0)

# Option 1: Save using PIL
pil_image = Image.fromarray(first_image)
pil_image.save('first_image_pil.png')
