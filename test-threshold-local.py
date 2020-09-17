from timeit import default_timer as timer
import pandas as pd
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
    # Offset is subtracted to the mask, so when we check we make faint structure white,
    # because they are brighter than the threshold.
    threshold_mask = threshold_local(image_bw, 21, offset=25, method="gaussian")
    print("[INFO]: Elapsed time", (timer()-start)*1000, "[ms]")
    image_bw_display = imutils.resize(image_bw, width=500)
    cv2.imshow("Current", image_display)
    cv2.imshow("Lightness", image_bw_display)
    threshold_mask_display = imutils.resize(threshold_mask, width=500).astype("uint8")
    cv2.imshow("mask", threshold_mask_display)
    # Check the mask content
    df = pd.DataFrame(threshold_mask_display)
    # print(df.head(200))

    # Thresholding to binary. The result is boolean, astype change it to u-int
    # but it is still 0, 1. So we multiply by 255 to scale it as an image
    image_binary = (image_bw > threshold_mask).astype("uint8")*255
    image_binary_display = imutils.resize(image_binary, width=500)
    cv2.imshow("binary", image_binary_display)

    cv2.waitKey(0)


