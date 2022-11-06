#### Задание ####
# По видео определить является ли движущийся объект человеком
# 1. Взять видео с движущимся человеком и не человеком
# 2. Движущиеся объекты выделить прямоугольником,
#     сохранить прямоугольники в виде файлов jpg (crop),
#     для каждого прямоугольника определить класс (человек, не человек) - сетка,
#     в лог файле написать что на таком - то кадре был человек с кординатами x1,y1,x2,y2

# 22.12.2020 - Промежуточные итоги
# VGG16 делает неправильный предикт

import cv2 as cv
import numpy as np
import pandas as pd
import functions # мои функции

import keras
from keras import layers, optimizers
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

print('import success')

# Считывание видео
video = cv.VideoCapture('test_video.mp4')
length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# VGG16
model = VGG16(weights='imagenet',
              include_top=True, # подключаем классификатор
              input_shape=(224, 224, 3))

# Для подсчета кадров видео в цикле
k = 0

if not video.isOpened():
    print('ERROR OPEN VIDEO')

cv.waitKey(1) # видео - это переключающиеся картинки - задаем время переключения (1 мс)

# для записи конечного видео
writer = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('test_video_vgg16.mp4', writer, 24.0, (width, height))

# считываем видео
_, frame1 = video.read()
_, frame2 = video.read()

# Путь для записи в лог
log_file_path = "/Users/ardandorzhiev/Downloads/UCHEBA_FA/Машинное зрение 2020/zachetnoe_zadanie/log_data_vgg16.csv"

while (video.isOpened() and k < length):

    # Находим разницу между кадрами
    diff = cv.absdiff(frame1, frame2)

    # Чтобы легче было найти контуры переведем в серый и заблюрим
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Определим порог 20
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

    # Выполним дилатацию, чтобы "залить" все "дырки" из изображния с порогом
    dilated = cv.dilate(thresh, None, iterations=3)

    # Находим контуры
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Обходим контуры
    cnt_contours = 0

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)

        ### Если контур маленький, то ничего  не делаем, и переходим к следующему контуру
        if cv.contourArea(contour) < 500:
            continue

        # Рисуем рамку
        cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Вырезаем объект из рамки
        obj = frame1.copy()[y:y+h, x:x+w]

        # Определяем класс объекта. Используем нейросеть из коробки
        image = cv.resize(obj, (224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        predict = model.predict(image)

        label = decode_predictions(predict)
        label = label[0][0]

        class_name = str(label[1])
        proba = label[2].item() # .item() - из np.float32 -> float

        ## Если уверенность в классе больше 70 процентов, то сохраняем и выводим класс на экран
        if proba > 0.1:
            proba = int(round(proba * 100))

            # сохраняем объект, распознанный в кадре
            path_name = 'objects_vgg16/' + 'frame_' + str(k) + '_obj_'+ class_name + '_' + str(cnt_contours) + '.jpg'
            cv.imwrite(path_name, obj)

            # Название объекта в рамке
            class_name_prob = class_name + '_' + str(proba) + '%'

            cv.putText(frame1, class_name_prob, (x, y-5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            #Записываем в лог
            functions.save_log(id_frame = k, id_object = cnt_contours,
                               x_coord = x, y_coord = y,
                               width = w, height = h,
                               obj_class = class_name, obj_proba = proba,
                               log_file_path = log_file_path)

            # Обработали один контур
            cnt_contours += 1

    # Прочитали один кадр
    k += 1

    # Записываем итог
    out.write(frame1)

    # Выводим на экран итог
    cv.imshow("Video", frame1)
    frame1 = frame2
    ret, frame2 = video.read()

    if k > length - 2:
        video.release()
        out.release()
        cv.destroyAllWindows()

    if cv.waitKey(1) & 0xFF == ord('q'): # нажимаем q если надо остановить воспроизвдеение
        video.release()
        out.release()
        cv.destroyAllWindows()