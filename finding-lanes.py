import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Functions


def greyscale(image):
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return greyscale


def darken(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Process
image = mpimg.imread('test_images/solidWhiteRight.jpg')
lane_image = image

greyscale_image = greyscale(lane_image)
darkened_image = darken(greyscale_image, 0.4)
blur_image = gaussian_blur(darkened_image, 5)


plt.imshow(blur_image, cmap="gray")
plt.show()
