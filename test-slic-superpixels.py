import collections
import imutils
from timeit import default_timer as timer
import list_images
import cv2

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

# START
image_files = list_images.files("test-data", " png")
image_files_d = collections.deque(image_files)

# Global variables
command = "read"
image = []

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
        cv2.imshow("SLIC: Original Image", image_display)
        command = "process"

    if command == "process":
        # Get the lightness
        start = timer()
        image_wk = imutils.resize(image, width=500)
        image_wk_float = img_as_float(image_wk)

        # loop over the number of segments
        numSegments = 30
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image_wk_float, n_segments=numSegments, sigma=2)

        # show the output of SLIC
        image_wk_display = mark_boundaries(image_wk, segments, color=(1, 0, 0))
        cv2.imshow(str("SLIC: Display "), image_wk_display)
        print("[INFO]: Segmentation", numSegments, "Elapsed time", (timer() - start) * 1000, "[ms] - Image Shape:",
              image_wk.shape)

        for segment in segments:
            print(segment)

        command = "idle"


