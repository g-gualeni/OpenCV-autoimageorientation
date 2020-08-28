import glob
import json
import os
import numpy as np


def files(path, extension):
    path2 = path.strip()
    path2 = path2.replace('\\', '/')
    path2 = path2.rstrip('/') + '/'
    glob_arg = path2 + "*." + extension.strip()
    print(glob_arg)
    file_list = glob.glob(glob_arg)
    return file_list


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


def corner_scale(image, image_display, points):
    # print(corner_scale.__name__, points)
    (display_w, display_h) = image_display.shape[:2]
    (image_w, image_h) = image.shape[:2]
    scale_w: float = display_w / image_w
    scale_h: float = display_h / image_h

    scaled_pts = points * (scale_w, scale_h)

    return scaled_pts
