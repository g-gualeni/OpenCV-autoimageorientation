import list_images
import cv2
import imutils
import transform

image_files = list_images.files("test-data ", " png")
for image_file in image_files:
    image = cv2.imread(image_file)
    image_display = imutils.resize(image, width=600)

    (display_w, display_h) = image_display.shape[:2]
    (image_w, image_h) = image.shape[:2]

    scale_w: float = display_w / image_w
    scale_h: float = display_h / image_h

    pts = list_images.corners_read(image_file)
    scaled_pts = pts * (scale_w, scale_h)
    for scaled_pt in scaled_pts:
        image_display = cv2.circle(image_display, (int(scaled_pt[0]), int(scaled_pt[1])), 6, color=(255, 0, 0),
                                   thickness=6)

    cv2.imshow("Current", image_display)
    image_rect = transform.four_point_transform(image, pts)
    image_rect_display = imutils.resize(image_rect, width=600)
    cv2.imshow("Rectified", image_rect_display)
    cv2.waitKey(0)




