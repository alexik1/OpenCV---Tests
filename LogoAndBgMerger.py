import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython.display import Image

#Set the main image and show it
img_bgr = cv2.imread("cocacola.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
print(img_rgb.shape)
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

#Set background
img_bg_bgr = cv2.imread("background.png")
img_bg_rgb = cv2.cvtColor(img_bg_bgr, cv2.COLOR_RGB2BGR)
aspect_ratio = logo_w / img_bg_rgb.shape[1]
dim = (logo_w, int(img_bg_rgb.shape[0] * aspect_ratio))

#resize and show it
img_bg_rgb = cv2.resize(img_bg_rgb, dim, interpolation=cv2.INTER_AREA)

plt.imshow(img_bg_rgb)
print(img_bg_rgb.shape)

#mask of main image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#threshold and show it
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_mask, cmap="gray")
print(img_mask.shape)

#invert the mask and show it
img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray")

#apply bg to mask and show it
img_bg = cv2.bitwise_and(img_bg_rgb, img_bg_rgb, mask=img_mask)
plt.imshow(img_bg)

#isolate foreground and show it
img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.imshow(img_foreground)

#merge foreground and bg
result = cv2.add(img_bg, img_foreground)
plt.imshow(result)
cv2.imwrite("Result.png", result[:, :, ::-1])
