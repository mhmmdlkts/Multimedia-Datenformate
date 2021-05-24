from SSIM_PIL import compare_ssim
from PIL import Image



image1 = Image.open("Bilder2.0/1-1/JPEG/s1/1.jpg")
image2 = Image.open("Bilder2.0/1-1/JPEG/s1/2.jpg")

value = compare_ssim(image1, image2) # Compare images using OpenCL by default
print(value)

value = compare_ssim(image1, image2, GPU=False) #  Compare images using CPU-only version
print(value)

import numpy as np
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))