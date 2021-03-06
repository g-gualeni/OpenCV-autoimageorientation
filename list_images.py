import glob
import json
import ntpath
import os

import cv2
import imutils
import numpy as np
from pathlib import Path


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
# Usage:
#   range_min, range_max = list_images.image_range(image, 0, 3)
#
def image_range(image, channel=0, filter_percentage=3):
    histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
    img_w, img_h = image.shape[:2]
    area = img_w * img_h
    filter_th = area * filter_percentage / 100
    acc = 0
    range_min = 0
    for idx, val in enumerate(histogram, 0):
        if acc + val > filter_th:
            range_min = idx
            break
        acc = acc + val

    acc = 0
    range_max = 0
    for idx in range(255, -1, -1):
        val = histogram[idx]
        if acc + val > filter_th:
            range_max = idx
            break
        acc = acc + val

    return range_min, range_max


# Get image area in pixels
def image_area(image):
    img_h, img_w = image.shape[:2]
    return img_h * img_w


# Adjust image dynamic range
# not working correctly because the black and white pixel are flipped
# not sure it is useful
# Usage:
# img_wk_bw = image_stretching(image_wk_hls[:, :, 1], range_min, range_max)
def image_stretching(image, range_min, range_max):
    if range_min == range_max:
        res = "Range min and range max are equal and {val}".format(val=range_max)
        raise ValueError(res)

    # Thresholding above range
    th1, out_image1 = cv2.threshold(image, range_max, 255, cv2.THRESH_TRUNC)
    out_image2 = out_image1.astype("float") - range_min
    out_image3 = np.where(out_image2 < 0, 0, out_image2)
    amplification = 255 / (range_max - range_min)
    out_image4 = out_image3 * amplification
    out_image5 = out_image4.astype('uint8')
    return out_image5


# Generate a contour with the same size of the image
# or a little smaller
def contours_from_image(img, border_x, border_y):
    img_contour = np.zeros((4, 1, 2), dtype=int)
    img_h, img_w = img.shape[:2]
    img_contour[0] = (0 + border_x, 0 + border_y)
    img_contour[1] = (img_w - border_x, 0 + border_y)
    img_contour[2] = (img_w - border_x, img_h - border_y)
    img_contour[3] = (0 + border_x, img_h - border_y)
    return img_contour


# Return empty contour if it is impossible to find a valid rect
# Validate the contour based on area occupation percentage (0-100)
def contours_from_edges(img_canny, min_area_percentage):
    contours = cv2.findContours(img_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_contour = []

    for idx, c in enumerate(contours):
        # approximate the contour
        # print("Contour:", idx, "Area:", cv2.contourArea(c), len(c))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        fill_rate = 100 * cv2.contourArea(approx) / image_area(img_canny)
        # print(fill_rate)
        if (len(approx) == 4) and (fill_rate > min_area_percentage):
            screen_contour = approx
            break
        # Just for debug... keep it but commented
        # img_canny_display = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
        # c2 = cv2.convexHull(c)
        # cv2.drawContours(img_canny_display, [c2], -1, (255 - (10 * idx), 200, 10 * idx), -1)
        # cv2.imshow("Contour From Edges: Debug", img_canny_display)

    return screen_contour


# Use the convexhull from  each contour we see
# this is  less precise but can be more robust
# I use the min area rect of the biggest shape
# Validate the contour based on area occupation percentage (0-100)
def contours_from_edges_convex_hull(img_canny, min_area_percentage):
    contours = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    convex_hulls = list()
    for c in contours:
        convex_hulls.append(cv2.convexHull(c))

    # Get the one with biggest area
    sorted_hulls = sorted(convex_hulls, key=cv2.contourArea, reverse=True)[:1]
    rotated_rect = cv2.minAreaRect(sorted_hulls[0])
    screen_contour = np.int0(cv2.boxPoints(rotated_rect))
    fill_rate = 100 * cv2.contourArea(screen_contour) / image_area(img_canny)
    # print(fill_rate)
    if fill_rate > min_area_percentage:
        return screen_contour
    # If fail return an empty list
    return []


# Save an image adding prefix, suffix
# save to a specified destination folder
# get name from file path
def image_save(image, filename, destination_folder, prefix="", suffix=""):
    img_name = ntpath.basename(filename)
    img_title, img_ext = os.path.splitext(img_name)
    img_name_out = prefix + img_title + suffix + img_ext
    img_path = os.path.join(destination_folder, img_name_out)
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(img_path, image)
    # print(filename, prefix, suffix, destination_folder, img_name_out)


# Plot hugh lines
def lines_hough_plot(image, lines, bgr_color=(0,255, 0), thickness=2):
    for line in lines:
        rho, theta = line[0]
        aa = np.cos(theta)
        bb = np.sin(theta)
        x0 = aa * rho
        y0 = bb * rho
        # first point
        x1 = int(x0 + 1000*(-bb))
        y1 = int(y0 + 1000 * aa)
        # second point
        x2 = int(x0 - 1000 * (-bb))
        y2 = int(y0 - 1000 * aa)

        # draw the line
        cv2.line(image, (x1, y1), (x2, y2), bgr_color, thickness)
        cv2.line(image, (x1, y1), (x2, 100), bgr_color, thickness)

    cv2.imshow("tua zia", image)

    return image


# Plot hugh lines from hough P so lines is already a list of points in the image
def lines_hough_p_plot(image, lines, bgr_color=(0,255, 0), thickness=2):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), bgr_color, thickness)

    return image
