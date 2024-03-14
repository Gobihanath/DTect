import os

import cv2
import numpy as np

# Load the image
image = cv2.imread('image_path')

# Resize the image to a specific width and height
new_width = 300
new_height = 400
resized_image = cv2.resize(image, (new_width, new_height))

# Convert the resized image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and help contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the large rectangular box
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Extract the region of interest (ROI) from the original image
largest_box_image = resized_image[y:y + h, x:x + w]

# Divide the largest_box_image into 10 equal parts
num_parts = 10
part_height = h // num_parts


# Save each part as a separate image
for i in range(num_parts):
    part = largest_box_image[i * part_height: (i + 1) * part_height, :]
    ext_image = cv2.resize(part, (256, 256))

    # If you want to see the outputs and save the outputs
    #cv2.imshow(f'Part {i+1}', ext_image)
    #cv2.waitKey(0)



#cv2.destroyAllWindows()
