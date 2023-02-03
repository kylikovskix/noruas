import argparse
import pickle
from threading import Thread
import cv2
import os
from flask import Flask, render_template, Response
from face_recognition_thread import FaceRecognitionThread
from face_recognition_thread import face_recognition_train
from hal import *
from paho.mqtt import client as mqtt_client
from config import *

app = Flask(__name__)

def frame_pub(mqtt_client, face_recognition_tread):
    while True:
        jpeg = get_jpeg(face_recognition_tread)
        if jpeg is not None:
            mqtt_client.publish(topic_frame_pub, jpeg.tobytes())
        time.sleep(1)


def unlock_sub(client, userdata, msg):
    if msg.payload.decode() == "PRESS" and msg.topic == topic_unlock_sub:
        door_unlock()


def mqtt_stream_start(face_recognition_tread):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            thread = Thread(target=frame_pub, args=(client, face_recognition_tread))
            thread.start()

            client.subscribe(topic_unlock_sub)
            client.on_message = unlock_sub
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect_async(broker, port)
    client.loop_start()
    print("mqtt client started")


def get_jpeg(face_recognition_tread):
    result = face_recognition_tread.get_result()
    img = result['frame']

    if img is not None:
        for (top, right, bottom, left), name, landmark in zip(result['faces'], result['names'], result['landmarks']):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            for k in landmark:
                for p in landmark[k]:
                    cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), -1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        ret, jpeg = cv2.imencode('.jpg', img, encode_param)
        return jpeg

    return None


@app.route('/')
def index():
    return render_template('index.html')


def gen(face_recognition_tread):
    while True:
        jpeg = get_jpeg(face_recognition_tread)
        if jpeg is not None:
            frame_data = jpeg.tobytes()
        else:
            frame_data = "None"

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(face_recognition_thread),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Обработчик нажатия кнопки
def my_press_btn_callback(chanel):
    print('button pressed')
    # включаем жёлтый светодиод и делаем паузу на 1 секунду
    yellow_indicator_on()
    time.sleep(1)
    # получаем список распознанных лиц
    names = face_recognition_thread.get_result()['names']
    # определяем наличие распознанх персон в списке запрещенных
    for name in black_list:
        if name in names:
            # в случае их наличия, гасим желтый индикатор и зажигаем красный на 1 секунду
            yellow_indicator_off()
            red_indicator_on()
            time.sleep(1)
            red_indicator_off()
            # в доступе отказано
            return

    # определяем наличие распознанх персон в списке разрешенных
    for name in access_list:
        # если есть, гасим жёлтый светодиод, зажигаем зеленый и включаем реле на 5 сек для открытия двери
        if name in names:
            yellow_indicator_off()
            door_unlock()
            return

    # в остальных случаях доступ запрещен
    yellow_indicator_off()
    red_indicator_on()
    time.sleep(1)
    red_indicator_off()


if __name__ == '__main__':
    try:
        # разбор агрументов переданных при запуске программы
        parser = argparse.ArgumentParser()
        parser.add_argument('--faces', type=str,
                            help='Каталог фотографий для тренировки модели распознавания лиц')
        parser.add_argument('--face_enc', type=str,
                            help='Файл данных обученной модели распознавания лиц')
        parser.add_argument('--force-train', type=bool,
                            help='Выполнить принудительное обучение модели перед её использованием')
        args = parser.parse_args()


        if args.face_enc is not None:
            face_enc_name = args.face_enc
        else:
            face_enc_name = 'face_enc'

        if args.faces is not None:
            faces_path = args.faces
        else:
            faces_path = 'faces'

        # при необходимости выполним тренировку модели
        if args.force_train or not os.path.exists(face_enc_name):
            face_recognition_train(faces_path, face_enc_name)

        # загрузми файл данных обученной модели
        data = pickle.loads(open(face_enc_name, "rb").read())
        # создадим и запустим в отдельном потоке исполнения
        # распознование лиц с камеры
        face_recognition_thread = FaceRecognitionThread(data)
        face_recognition_thread.start()

        # инициализация аппаратной части
        hal_init()
        # регистрация обработчика нажатия кнопки
        add_press_btn_callback(my_press_btn_callback)

        # запуск подсистемы сообщений на базе протокола mqtt
        mqtt_stream_start(face_recognition_thread)

        # запуск веб-интерфеса
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        hal_free()

    hal_free()
    face_recognition_thread.stop()
    face_recognition_thread.join()



