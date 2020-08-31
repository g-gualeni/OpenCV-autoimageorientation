import glob
import json
import os

import cv2
import imutils
import numpy as np


# List all files in a folder
def files(path, extension):
    path2 = path.strip()
    path2 = path2.replace('\\', '/')
    path2 = path2.rstrip('/') + '/'
    glob_arg = path2 + "*." + extension.strip()
    print(glob_arg)
    file_list = glob.glob(glob_arg)
    return file_list


# Read corners position from a JSON file named as the image file
def corners_read(image_path):
    path = image_path.strip() + ".json"

    with open(path) as json_file:
        data = json.load(json_file)
        corners_list = data['corners']
        image_name = data['imageName']
        if image_name != os.path.basename(image_path):
            raise ValueError(os.path.basename(image_path) +
                             " is not the expected imageName (" + image_name + ")")

        rect = np.zeros((4, 2), dtype="float32")
        rows, cols = rect.shape
        for idx, p in enumerate(data["corners"]):
            if idx < rows:
                rect[idx] = (p["x"], p["y"])

    return rect


# Rescale corner position for the display image
def corner_scale(image, image_display, points):
    # print(corner_scale.__name__, points)
    (display_w, display_h) = image_display.shape[:2]
    (image_w, image_h) = image.shape[:2]
    scale_w: float = display_w / image_w
    scale_h: float = display_h / image_h

    scaled_pts = points * (scale_w, scale_h)

    return scaled_pts


# Image Range
# Return the min and max value, excluding a fixed % of points
# this help remove noise
def image_range(image, channel, filter_percentage):
    histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
    img_w, img_h = image.shape[:2]
    image_area = img_w * img_h
    filter_th = image_area * filter_percentage / 100
    acc = 0
    range_min = 0
    for idx, val in enumerate(histogram, 0):
        if acc + val > filter_th:
            range_min = idx
            break
        acc = acc + val

    acc = 0
    for idx in range(255, -1, -1):
        val = histogram[idx]
        if acc + val > filter_th:
            range_max = idx
            break
        acc = acc + val

    return range_min, range_max


# Adjust image dynamic range
# not working correctly because the black and white pixel are flipped
# not sure it is useful
def image_stretching(image, range_min, range_max):
    # Thresholding above range
    th1, out_image1 = cv2.threshold(image, range_max, 255, cv2.THRESH_TRUNC)
    out_image2 = out_image1.astype("float") - range_min
    out_image3 = np.where(out_image2 < 0, 0, out_image2)
    amplification = 255 / (range_max - range_min)
    out_image4 = out_image3 * amplification
    out_image5 = out_image4.astype('uint8')
    return out_image5


# Generate a contour with the same size of the image
def contours_from_image(img, border_x, border_y):
    img_contour = np.zeros((4, 1, 2), dtype=int)
    img_h, img_w = img.shape[:2]
    print(img.shape)
    img_contour[0] = (0 + border_x, 0 + border_y)
    img_contour[1] = (img_w - border_x, 0 + border_y)
    img_contour[2] = (img_w - border_x, img_h - border_y)
    img_contour[3] = (0 + border_x, img_h - border_y)
    return img_contour


# Return empty contour if it is impossible to find a valid rect
def contours_from_edges(img_canny):
    contours = cv2.findContours(img_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_contour = []

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screen_contour = approx
            break

        cv2.drawContours(img_canny, [c], -1, (50, 50, 50), 2)
        cv2.imshow("Contour From Edges: Debug", img_canny)

    return screen_contour
