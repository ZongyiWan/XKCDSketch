import cv2
import numpy as np

# Load the image
image = cv2.imread("image7.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1. Grayscale Image", gray)
cv2.imwrite("step1_grayscale.jpg", gray)

# Apply Edge Detection (Canny)
edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds for stronger/weaker edges
cv2.imshow("2. Edge Detection (Canny)", edges)
cv2.imwrite("step2_edges.jpg", edges)


# Invert the grayscale image
inverted = 255 - gray
cv2.imshow("3. Inverted Grayscale", inverted)
cv2.imwrite("step3_inverted.jpg", inverted)

bilateral = cv2.bilateralFilter(inverted, 15, 75, 75)
cv2.imshow("2.5 bilateral", bilateral)
cv2.imwrite("bilateral.jpg", bilateral)

# Apply Gaussian Blur to the inverted image
blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
cv2.imshow("4. Gaussian Blur", blurred)
cv2.imwrite("step4_blur.jpg", blurred)

# Invert the blurred image
inverted_blurred = 255 - bilateral
cv2.imshow("5. Inverted Blurred", inverted_blurred)
cv2.imwrite("step5_inverted_blur.jpg", inverted_blurred)

# Create the pencil sketch using Color Dodge blending
pencil_sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
cv2.imshow("6. Pencil Sketch (Before Edge Enhancement)", pencil_sketch)
cv2.imwrite("step6_pencil_sketch.jpg", pencil_sketch)

# Blend the sketch with the edge map to enhance edges
final_sketch = cv2.addWeighted(pencil_sketch, 0.8, edges, 0.2, 0)
cv2.imshow("7. Final Pencil Sketch (With Edge Enhancement)", final_sketch)
cv2.imwrite("step7_final_sketch.jpg", final_sketch)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()