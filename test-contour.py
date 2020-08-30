from timeit import default_timer as timer
import cv2
import imutils

import list_images


def nothing(x):
    pass


image_files = list_images.files("test-data ", " png")
for image_file in image_files:
    image = cv2.imread(image_file)
    image_display = imutils.resize(image, width=500)
    ratio = image.shape[0] / 500

    start = timer()
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    print("[INFO]: Conversion Elapsed time", (timer() - start) * 1000, "[ms]")
    # Get the lightness
    image_bw: object = image_lab[:, :, 0]
    image_bw_display = imutils.resize(image_bw, width=500)

    cv2.imshow("Contour: Current", image_display)
    cv2.imshow("Contour: Lightness", image_bw_display)
    kernel = 5
    while 1:
        if kernel
        start = timer()
        image_bw_gaussian = cv2.GaussianBlur(image_bw_display, (kernel, kernel), 0)
        print("[INFO]: GaussianBlur Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow("Contour: GaussianBlur", image_bw_gaussian)
        cv2.createTrackbar("Kernel", "Contour: GaussianBlur", kernel, 55, nothing)
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
        new_kernel = max(1, cv2.getTrackbarPos("Kernel", "Contour: GaussianBlur"))
        print(kernel)
        kernel = (kernel & 0xFE)+1

    print("Next image please")
