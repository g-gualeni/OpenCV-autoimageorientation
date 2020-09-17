import collections
import imutils
from timeit import default_timer as timer

from skimage.color import rgb2gray

import list_images
import cv2

from skimage.segmentation import  felzenszwalb
from skimage.segmentation import quickshift
from skimage.segmentation import slic
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.filters import sobel

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

    if command == "read":
        image_file = image_files_d[0]
        image = cv2.imread(image_file)
        print("Next image please: " + image_files_d[0], image.shape)
        image_display = imutils.resize(image, width=500)
        cv2.imshow("SLIC: Original Image", image_display)
        command = "process"

    if command == "process":
        # Create the working image
        start = timer()
        image_wk = imutils.resize(image, width=500)
        image_wk_float = img_as_float(image_wk)
        print("[INFO]: Resize Elapsed time", (timer() - start) * 1000, "[ms]")

        # SLIC
        start = timer()
        numSegments = 100
        segments = slic(image_wk_float, n_segments=numSegments, sigma=1)
        image_wk_display = mark_boundaries(image_wk, segments, color=(1, 0, 0))
        print("[INFO]: SLIC", len(segments), "Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow(str("SLIC: Display "), image_wk_display)

        # Quick Shift
        start = timer()
        segments_qshift = quickshift(image_wk_float)
        image_wk_quickshift = mark_boundaries(image_wk, segments_qshift, color=(1, 0, 0))
        print("[INFO]: Quickshift", len(segments_qshift), "Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow(str("Quickshift: Display "), image_wk_quickshift)

        # Felzenszwalb
        start = timer()
        segments_felzenszwalb = felzenszwalb(image_wk_float, sigma=2)
        image_wk_felzenszwalb = mark_boundaries(image_wk, segments_felzenszwalb, color=(1, 0, 0))
        print("[INFO]: Felzenszwalb", len(segments_felzenszwalb), "Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow(str("Felzenszwalb: Display "), image_wk_felzenszwalb)

        # Watershed
        start = timer()
        gradient = sobel(rgb2gray(image_wk))
        segments_watershed = watershed(gradient, markers=100, compactness=0.001)
        image_wk_watershed = mark_boundaries(image_wk, segments_watershed, color=(1, 0, 0))
        print("[INFO]: watershed", len(segments_watershed), "Elapsed time", (timer() - start) * 1000, "[ms]")
        cv2.imshow(str("watershed: Display "), image_wk_watershed)

        command = "idle"

# Provare con SLICO di OpenCV dovrebbe gestire meglio le texture