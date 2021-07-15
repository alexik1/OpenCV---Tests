import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

#insert file of photo/image to the same directory than this py file, set name of file below, or so
img = cv2.imread("SheetMusic.png", cv2.IMREAD_GRAYSCALE)

#threshold as adaptive, to focus on making the background white and bolden the black pixels
img_thres_adp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=[18, 15])
plt.subplot(221); plt.imshow(img, cmap="gray"); plt.title("Original")
plt.subplot(222); plt.imshow(img_thres_adp, cmap="gray"); plt.title("Thresholded (Adaptive)")
