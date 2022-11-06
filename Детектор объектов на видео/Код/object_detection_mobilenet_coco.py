#### Задание ####
# По видео определить является ли движущийся объект человеком
# 1. Взять видео с движущимся человеком и не человеком
# 2. Движущиеся объекты выделить прямоугольником,
#     сохранить прямоугольники в виде файлов jpg (crop),
#     для каждого прямоугольника определить класс (человек, не человек) - сетка,
#     в лог файле написать что на таком - то кадре был человек с кординатами x1,y1,x2,y2

import cv2 as cv
import numpy as np
import pandas as pd
import functions # мои функции
import matplotlib as plt

print('import success')

# Считывание видео
video = cv.VideoCapture('test_video.mp4')
length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Для подсчета кадров видео в цикле
k = 0

# Предобученная модель
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv.dnn_DetectionModel(frozen_model, config_file)

# Устанавливаем параметры модели
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) # 255/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) # mobilenet => [-1, 1]
model.setInputSwapRB(True) # convert img to

# Classes
classLabels = []
file_name = 'Labels_mobilenet.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

if not video.isOpened():
    print('ERROR OPEN VIDEO')

# Параметры рамки объектов
font_scale = 1
font = cv.FONT_HERSHEY_COMPLEX_SMALL

cv.waitKey(1) # видео - это переключающиеся картинки - задаем время переключения (1 мс)

# для записи конечного видео
writer = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('test_video_mobilenet.mp4', writer, 24.0, (width, height))

# Путь для записи в лог
log_file_path = "/Users/ardandorzhiev/Downloads/UCHEBA_FA/Машинное зрение 2020/zachetnoe_zadanie/log_data_mobilenet.csv"

while (video.isOpened() and k < length):

    ret, frame = video.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if (len(ClassIndex) != 0): # различила ли модель вообще какие-нибудь классы
        # обходим все объекты
        cnt_objs = 0
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<=80): # тк всего изначально 80 классов, проверяем!!!
                # Найдем границы прямогульника, который выделяет объекты
                x = boxes[0]
                y = boxes[1]
                w = boxes[2]
                h = boxes[3]

                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                # cv.rectangle(frame, boxes, (0,255,0), 2) - можно и boxes сразу вставлять - список [x, y, w, h]

                # Название объекта в рамке
                class_name = classLabels[ClassInd-1]
                proba = round(conf*100,0)
                class_name_prob = class_name + '_' + str(proba) + '%'

                cv.putText(frame, class_name_prob, (boxes[0], boxes[1]-5), font, font_scale, (0, 0, 255), 2)

                # сохрянем объект, распознанный в кадре
                # Вырезаем объект из рамки
                obj = frame.copy()[y:y + h, x:x + w]
                path_name = 'objects_mobilenet/' + 'frame_' + str(k) + '_obj_'+ class_name + '_' + str(cnt_objs) + '.jpg'
                cv.imwrite(path_name, obj)

                #Записываем в лог
                functions.save_log(id_frame = k, id_object = cnt_objs,
                                   x_coord = x, y_coord = y,
                                   width = w, height = h,
                                   obj_class = class_name, obj_proba = proba,
                                   log_file_path = log_file_path)

                # Обработали один объект
                cnt_objs += 1


    # Прочитали один кадр
    k += 1

    # Выводим на экран итог
    cv.imshow("Video", frame)

    # Записываем итог
    out.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'): # нажимаем q если надо остановить воспроизвдеение
        video.release()
        out.release()
        cv.destroyAllWindows()