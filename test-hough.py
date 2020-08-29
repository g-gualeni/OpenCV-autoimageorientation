import imutils
from skimage.filters import threshold_local
import cv2
import list_images


image_files = list_images.files("test-data ", " png")
for image_file in image_files:
    image = cv2.imread(image_file)
    image_display = imutils.resize(image, width=600)

    image_lab = cv2.cvtColor(image_display, cv2.COLOR_BGR2LAB)
    # Get the lightness
    image_bw: object = image_lab[:, :, 0]
    cv2.imshow("Current", image_display)
    cv2.imshow("Lightness", image_bw)
    threshold_mask = threshold_local(image_bw, 11, offset=10, method="gaussian")
    cv2.imshow("mask", threshold_mask)
    image_binary = (image_bw > threshold_mask).astype("uint8") * 255
    cv2.imshow("binary", image_binary)

    cv2.waitKey(0)


