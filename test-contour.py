from timeit import default_timer as timer
import cv2
import imutils
import collections
import list_images
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


# START
image_files = list_images.files("test-data ", " png")
image_files_d = collections.deque(image_files)

# Global variables
kernel = 5
new_kernel = 0
command = "read"
image = np.zeros(shape=[1, 1, 3], dtype=np.uint8)
ratio = 1
canny_min_val = 75
canny_max_val = 200

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
        plt.show()
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
        image_wk_bw_canny = cv2.Canny(image_wk_bw_blurred, canny_min_val, canny_max_val, False)
        print("[INFO]: Canny Elapsed time", (timer() - start) * 1000)
        cv2.imshow("Contour: Canny", image_wk_bw_canny)
        cv2.createTrackbar("Th MinVal", "Contour: Canny", canny_min_val, 255, nothing)
        cv2.createTrackbar("Th MaxVal", "Contour: Canny", canny_max_val, 255, nothing)

        command = "idle"

    # check for a new kernel
    new_kernel = max(1, cv2.getTrackbarPos("Kernel", "Contour: GaussianBlur"))
    new_kernel = (new_kernel & 0xFE)+1
    if new_kernel != kernel:
        kernel = new_kernel
        command = "process"

    new_canny_min_val = cv2.getTrackbarPos("Th MinVal", "Contour: Canny")
    new_canny_max_val = cv2.getTrackbarPos("Th MaxVal", "Contour: Canny")
    if(new_canny_min_val != canny_min_val ) or (new_canny_max_val != canny_max_val):
        canny_min_val = new_canny_min_val
        canny_max_val = new_canny_max_val
        command = "process"
