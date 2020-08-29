from timeit import default_timer as timer

import imutils
from skimage.filters import threshold_local
import cv2
import list_images

# Testing local thresholding


image_files = list_images.files("test-data ", " png")
for image_file in image_files:
    image = cv2.imread(image_file)
    image_display = imutils.resize(image, width=500)

    start = timer()
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    print("[INFO]: Conversion Elapsed time", (timer() - start) * 1000, "[ms]")
    # Get the lightness
    image_bw: object = image_lab[:, :, 1]

    # There is one threshold value for each pixels
    start = timer()
    threshold_mask = threshold_local(image_bw, 201, offset=50, method="gaussian")
    print("[INFO]: Elapsed time", (timer()-start)*1000, "[ms]")
    image_bw_display = imutils.resize(image_bw, width=500)
    cv2.imshow("Current", image_display)
    cv2.imshow("Lightness", image_bw_display)
    th_max = threshold_mask.max()
    th_min = threshold_mask.min()

    plot_mask = threshold_mask - th_min
    plot_mask = plot_mask * (255/th_max)
    threshold_mask_display = imutils.resize(plot_mask, width=500)
    cv2.imshow("mask", threshold_mask_display)

    # Thresholding to binary. The result is boolean, astype change it to u-int
    # but it is still 0, 1. So we multiply by 255 to scale it as an image
    image_binary = (image_bw > threshold_mask).astype("uint8")*255
    image_binary_display = imutils.resize(image_binary, width=500)
    cv2.imshow("binary", image_binary_display)

    cv2.waitKey(0)


