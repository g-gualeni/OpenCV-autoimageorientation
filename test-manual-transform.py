import list_images
import cv2
import imutils
import transform


# Manual transformation using json file containing corners coordinate
image_files = list_images.files("test-data ", " png")
for image_file in image_files:
    image = cv2.imread(image_file)
    image_display = imutils.resize(image, width=600)

    pts = list_images.corners_read(image_file)
    for scaled_pt in list_images.corner_scale(image, image_display, pts):
        image_display = cv2.circle(image_display, (int(scaled_pt[0]), int(scaled_pt[1])), 6, color=(255, 0, 0),
                                   thickness=-1)
    cv2.imshow("Current", image_display)
    image_rect = transform.four_point_transform(image, pts)
    image_rect_display = imutils.resize(image_rect, width=600)
    cv2.imshow("Rectified", image_rect_display)
    cv2.waitKey(0)

