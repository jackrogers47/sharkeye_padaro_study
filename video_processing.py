import cv2

def ar_resize(width, height, imgsz_h):
    imgsz_w = imgsz_h * width / height
    return [imgsz_h, imgsz_w]
