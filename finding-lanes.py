import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Functions


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def darken(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


# Process
image = mpimg.imread('test_images/solidWhiteRight.jpg')
lane_image = image

greyscale_image = greyscale(lane_image)
darken_image = darken(greyscale_image, 0.4)
blur_image = gaussian_blur(darken_image, 7)
canny_image = canny(blur_image, 50, 150)


plt.imshow(canny_image, cmap="gray")
plt.show()
