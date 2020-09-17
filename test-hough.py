# Test image orientation using hough
import collections

import imutils
from timeit import default_timer as timer
import list_images
import numpy as np
import cv2


def nothing(x):
    pass


# START
image_files = list_images.files("test-data-cutted", " png")
image_files_d = collections.deque(image_files)
erosion_size = 2
new_erosion_size = erosion_size

# Global variables
command = "read"
image = np.zeros(shape=[1, 1, 3], dtype=np.uint8)
while 1:
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    if k == 32:
        # Move to next image
        command = "read"
        image_files_d.rotate(1)
        print("Next image please: " + image_files_d[0])

    if command == "read":
        image_file = image_files_d[0]
        image = cv2.imread(image_file)
        image_display = imutils.resize(image, width=500)
        cv2.imshow("Hough: Original Image", image_display)
        command = "process"

    if command == "process":
        # Get the lightness
        start = timer()
        image_wk = imutils.resize(image, width=500)
        image_wk_hls = cv2.cvtColor(image_wk, cv2.COLOR_BGR2HLS)
        range_min, range_max = list_images.image_range(image_wk_hls, 1)
        image_wk_bw = list_images.image_stretching(image_wk_hls[:, :, 1], range_min, range_max)
        print("[INFO]: image_stretching Elapsed time", (timer() - start) * 1000, "[ms]")

        start = timer()
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        image_wk_erode = cv2.erode(image_wk_bw, element)
        print("[INFO]: erode Elapsed time", (timer() - start) * 1000, "[ms]")

        start = timer()
        image_wk_canny = imutils.auto_canny(image_wk_erode)
        print("[INFO]: Canny Elapsed time", (timer() - start) * 1000, "[ms]")

        start = timer()
        lines = cv2.HoughLinesP(image_wk_canny, 4, np.pi / 90, 50)
        # print(lines[:3])

        image_wk_col = cv2.cvtColor(image_wk_bw, cv2.COLOR_GRAY2BGR)
        image_wk_col = list_images.lines_hough_p_plot(image_wk_col, lines[:10])

        cv2.imshow("Stretching+Lines", image_wk_col)
        cv2.imshow("Canny", image_wk_canny)
        cv2.createTrackbar("Kernel", "Erosion", erosion_size, 20, nothing)
        cv2.imshow("Erosion", image_wk_erode)

        command = "idle"

    # check for a new erosion kernel
    # Mandatory After read and process
    new_erosion_size = max(1, cv2.getTrackbarPos("Kernel", "Erosion"))
    if new_erosion_size != erosion_size:
        erosion_size = new_erosion_size
        command = "process"
