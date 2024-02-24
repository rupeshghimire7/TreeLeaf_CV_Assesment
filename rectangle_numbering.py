
'''
- Detect rectangle
    - Use a rectangle detection algorithm (e.g., contour detection or Hough transform) to identify the rectangles in the image

- Detect lines inside rectangles
    - Use Hough transform for line detection

- Measure line lengths
    - Calculate the length of each detected line

- Index rectangles based on line lengths
    - Sort rectangles based on the lengths of the lines contained within them
    - Assign smaller indices to rectangles with shorter lines
'''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('rect.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Grayscale the image

ret,thresh = cv.threshold(gray,200,255,cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

contour_image = image.copy()
cv.drawContours(contour_image, contours[1:], -1, (0, 255, 0), 2)

for i, contour in enumerate(contours):
    # Print contour index
    print(f"Contour {i}:")

    # Print number of points
    print(f"Number of Points: {len(contour)}")

    # Calculate area of contour
    area = cv.contourArea(contour)
    print(f"Area: {area}")

    # Calculate perimeter of contour
    perimeter = cv.arcLength(contour, True)
    print(f"Perimeter: {perimeter}")

    # Calculate bounding rectangle
    x, y, w, h = cv.boundingRect(contour)
    print(f"Bounding Rectangle: (x={x}, y={y}, w={w}, h={h})")

    # Calculate aspect ratio of bounding rectangle
    aspect_ratio = float(w) / h
    print(f"Aspect Ratio: {aspect_ratio}")

    vertices = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    num_vertices = len(vertices)
    print(f"Number of vertices: {num_vertices}")


    # Print newline for separation
    print()


rectangles = []
lines = []

def get_shape(contour):
    # Calculate area of contour
    area = cv.contourArea(contour)

    vertices = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    num_vertices = len(vertices)

    if area<1000 and num_vertices<3:
      return 'Line'
    else: return 'Rectangle'


for contour in contours:
    shape = get_shape(contour)
    if shape == 'Rectangle':
      rectangles.append(contour)
    elif shape == 'Line':
        lines.append(contour)
    else:
      pass



peri_dict = {}
for i, contour in enumerate(lines):
  perimeter = cv.arcLength(contour, True)
  peri_dict[i+1] = perimeter


peri_dict = {k: v for k, v in sorted(peri_dict.items(), key=lambda item: item[1])}
ranks = ['1st', '2nd','3rd', '4th' ]
i=0
for idx, val in peri_dict.items():
  peri_dict[idx] = [val,ranks[i]]
  i+=1

img = image.copy()
for i, contour in enumerate(lines):
    perimeter = peri_dict[i+1][0]  # Get the perimeter value for the current contour
    x, y, w, h = cv.boundingRect(contour)  # Get the bounding rectangle coordinates

    text = peri_dict[i+1][1]

    # Calculate the position for placing the text beside the rectangle
    text_position = (x + w + 10, y + h-35)

    # Draw the text on the image
    cv.putText(img, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Ranked")
plt.imshow(img)

plt.show()