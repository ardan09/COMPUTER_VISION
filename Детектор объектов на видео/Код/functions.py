import cv2 as cv
import numpy as np
import csv

def canny(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    blur = cv.GaussianBlur(img, (5,5), 0)

    res_img = cv.Canny(blur, 50, 100)  # отношение параметров лучше брать 1 к 2 или 1 к 3

    return res_img

def get_color_part(img):

    low_fire = np.array((30,20,50), np.uint8)
    high_fire = np.array((70,100,100), np.uint8)

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_fire = cv.inRange(img_hsv, low_fire, high_fire)

    img_fire = cv.bitwise_and(img_hsv, img_hsv, mask = mask_fire)
    img_fire = cv.cvtColor(img_fire, cv.COLOR_HSV2BGR)

    return img_fire

# записываем объекты в лог
def save_log(id_frame, id_object, x_coord , y_coord, width, height, obj_class, obj_proba, log_file_path):
    """Создает файл или открывает существующий => записывет 1 объект (контур)"""
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['id_frame', 'id_object', 'x_coord', 'y_coord', 'width', 'height', 'obj_class', 'obj_proba']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(dict(id_frame = id_frame, id_object = id_object,
                     x_coord = x_coord, y_coord = y_coord,
                     width = width, height = height,
                     obj_class = obj_class, obj_proba = obj_proba))
