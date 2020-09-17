from timeit import default_timer as timer
import cv2
import imutils
import collections
import list_images
import numpy as np
from matplotlib import pyplot as plt

import transform


def nothing(x):
    pass


# START
image_files = list_images.files("test-data ", " png")
image_files_d = collections.deque(image_files)

# Global variables
kernel = 5
new_kernel = kernel
command = "read"
image = np.zeros(shape=[1, 1, 3], dtype=np.uint8)
ratio = 1

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
        ratio = image.shape[0] / 500
        cv2.imshow("Contour: Current", image_display)
        image_display_hls = cv2.cvtColor(image_display, cv2.COLOR_BGR2HLS)
        range_min, range_max = list_images.image_range(image_display_hls, 1, 3)
        image_display_gray = list_images.image_stretching(image_display_hls[:, :, 1], range_min, range_max)
        cv2.imshow("Contour: BW Equalized", image_display_gray)
        histogram = cv2.calcHist([image_display_gray], [0], None, [256], [0, 256])
        plt.plot(histogram, color='g')
        plt.xlim([0, 256])
        plt.title("Gray Histogram")
        # plt.show()
        command = "process"

    if command == "process":
        # Get the lightness
        start = timer()
        image_wk = imutils.resize(image, width=500)
        image_wk_hls = cv2.cvtColor(image_wk, cv2.COLOR_BGR2HLS)
        range_min, range_max = list_images.image_range(image_wk_hls, 1, 3)
        image_wk_bw = list_images.image_stretching(image_wk_hls[:, :, 1], range_min, range_max)
        print("[INFO]: Sampling and Conversion Elapsed time", (timer() - start) * 1000, "[ms]")

        start = timer()
        image_wk_bw_blurred = cv2.GaussianBlur(image_wk_bw, (kernel, kernel), 0)
        print("[INFO]: GaussianBlur Elapsed time", (timer() - start) * 1000, "[ms] - Kernel:", kernel)
        cv2.imshow("Contour: GaussianBlur", image_wk_bw_blurred)
        cv2.createTrackbar("Kernel", "Contour: GaussianBlur", kernel, 55, nothing)

        start = timer()
        image_wk_bw_canny = imutils.auto_canny(image_wk_bw_blurred)
        print("[INFO]: Canny Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow("Contour: Canny", image_wk_bw_canny)

        # Contour search
        start = timer()
        min_fill_percentage = 20
        # 1 using canny and contour approximation
        screen_contour = list_images.contours_from_edges(image_wk_bw_canny, min_fill_percentage)
        # 2 using canny, contour, convex hull and minAreaRect
        if len(screen_contour) == 0:
            screen_contour = list_images.contours_from_edges_convex_hull(image_wk_bw_canny, min_fill_percentage)
        # 3 using image shape
        if len(screen_contour) == 0:
            screen_contour = list_images.contours_from_image(image_wk_bw_canny, 10, 10)

        print("[INFO]: Contour Search", (timer() - start) * 1000, "[ms]")

        image_wk_bw_warped = transform.four_point_transform(image_wk_bw, screen_contour.reshape(4, 2))
        # I don't like the result of the binarization
        # threshold_mask = threshold_local(image_wk_bw_warped, 21, offset=5, method="gaussian")
        # image_wk_bw_warped = (image_wk_bw_warped > threshold_mask).astype("uint8") * 255

        cv2.drawContours(image_wk, [screen_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Contour: Outline", image_wk)
        cv2.imshow("Warped Binary Image", image_wk_bw_warped)

        list_images.image_save(image_wk_bw_warped, image_files_d[0],
                               suffix='_cut',
                               destination_folder="test-data-cutted")

        command = "idle"

    # check for a new kernel
    new_kernel = max(1, cv2.getTrackbarPos("Kernel", "Contour: GaussianBlur"))
    new_kernel = (new_kernel & 0xFE)+1
    if new_kernel != kernel:
        kernel = new_kernel
        command = "process"
