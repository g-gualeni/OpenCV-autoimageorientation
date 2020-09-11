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
        image_wk = img_as_float(io.imread(image_files[0]))

        # loop over the number of segments
        for numSegments in (10, 100, 200, 300):
            # apply SLIC and extract (approximately) the supplied number
            # of segments
            segments = slic(image_wk, n_segments=numSegments, sigma=5)
            # show the output of SLIC
            fig = plt.figure("Superpixels -- %d segments" % numSegments)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image_wk, segments, color=(1, 0, 0)))
            plt.axis("off")
            print("[INFO]: Segmentation", numSegments, "Elapsed time", (timer() - start) * 1000, "[ms]")

        # show the plots
        plt.show()
        command = "idle"


