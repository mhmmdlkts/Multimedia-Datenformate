import pillow_avif

from matplotlib.image import imread

from sewar.full_ref import uqi, mse, psnr, ssim
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)

img1 = imread("kimono.avif")
img2 = imread("kimono.avif", 1)
val = uqi(img1,img2)
print(val)