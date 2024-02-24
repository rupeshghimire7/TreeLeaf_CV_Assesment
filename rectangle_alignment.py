'''
SOLUTION:

- Separate all rectangles
    Cropping

- Thresholding and find contours
    Binary threshold with findContour()

- Get Center and Angle for Rotation
    image w,h/2 and define angles as needed

- Get Rotation Matrix and Rotate image to make it straight
    displayed using a function

'''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('rect.jpg')
height,width = image.shape[:2]

# Top Left
start_row, start_col = int(height*.15), int(width*.15)
end_row, end_col = int(height * .35), int(width*.5)
top_left = image[start_row:end_row , start_col:end_col]
top_left = cv.cvtColor(top_left, cv.COLOR_BGR2GRAY)
# plt.title("TOP LEFT RECT")
# plt.imshow(top_left)


# Bottom Left
start_row, start_col = int(height*.4), int(width*.10)
end_row, end_col = int(height * .8), int(width*.5)
bottom_left = image[start_row:end_row , start_col:end_col]
bottom_left = cv.cvtColor(bottom_left, cv.COLOR_BGR2GRAY)
# plt.title("BOTTOM LEFT RECT")
# plt.imshow(bottom_left)


# Top Right
start_row, start_col = int(height*.15), int(width*.5)
end_row, end_col = int(height * .4), int(width*.85)
top_right = image[start_row:end_row , start_col:end_col]
top_right = cv.cvtColor(top_right, cv.COLOR_BGR2GRAY)
# plt.title("TOP RIGHT RECT")
# plt.imshow(top_right)


# Bottom Right
start_row, start_col = int(height*.5), int(width*.5)
end_row, end_col = int(height * .9), int(width*.9)
bottom_right = image[start_row:end_row , start_col:end_col]
bottom_right = cv.cvtColor(bottom_right, cv.COLOR_BGR2GRAY)
plt.title("BOTTOM RIGHT RECT")
# plt.imshow(bottom_right)



def rect_alignment(images:list, titles:list, angles:list):

    '''
        Transform and plot images

        Parameters
        -------------
        images: list of images to straighten/align through rotation

        titiles: title for each image as needed while plotting

        angles: Angles of rotation for each image

        
        Returns
        --------------
        None
        
    '''



    fig, axes = plt.subplots(len(images), 2, figsize=(20, 20))

    for i, (img, title,angle) in enumerate(zip(images, titles,angles)):

        # Threshold image
        ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

        # Find Contours
        contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Draw Contour
        for j in range(1,len(contours)):
            cv.drawContours(img, contours[j], -1, (0,255,0), 1)

        # Get center coordinates of the image
        height, width = img.shape[:2]
        center = (width/2, height/2)

        # Get rotation Matrix
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
        # Rotate Image
        rotated_image = cv.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

        # Plot original Image
        axes[i, 0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2BGR))  # Convert BGR to RGB for plotting
        axes[i, 0].set_title(f"Original Image: {title}")

        # Plot rotated image
        axes[i, 1].imshow(cv.cvtColor(rotated_image, cv.COLOR_GRAY2BGR))  # Convert BGR to RGB for plotting
        axes[i, 1].set_title(f"Rotated image: {title}")

    plt.tight_layout()
    plt.show()
    

images = [top_left.copy(), bottom_left.copy(), top_right.copy(), bottom_right.copy()]
titles = ["Top Left", "Bottom Left", "Top Right", "Bottom Right"]
angles=[15, -30,-15,30]

rect_alignment(images, titles,angles)