import cv2
import torch
import numpy as np
import cv2

img = cv2.imread("powerplant_sketch.jpg")


# Bilateral filter preserves edges while blurring textures
cartoon = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Save output
cv2.imwrite("cartoon.jpg", cartoon)