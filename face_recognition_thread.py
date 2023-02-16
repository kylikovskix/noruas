import threading
import face_recognition
import pickle
import cv2
import os
from imutils import paths
import config


def face_recognition_train(faces_dir, face_enc_name):
    print("Start math faces encodings...")
    imagePaths = list(paths.list_images(faces_dir))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        # извлекаем имя человека из названия папки
        name = imagePath.split(os.path.sep)[-2]
        # загружаем изображение и конвертируем его из BGR (OpenCV ordering)
        # в dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # используем библиотеку Face_recognition для обнаружения лиц
        boxes = face_recognition.face_locations(rgb, model='hog')
        # вычисляем эмбеддинги для каждого лица
        encodings = face_recognition.face_encodings(rgb, boxes, model=config.model)
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # сохраним эмбеддинги вместе с их именами в формате словаря
    data = {"encodings": knownEncodings, "names": knownNames}
    # для сохранения данных в файл используем библиотеку pickle
    f = open(face_enc_name, "wb")
    f.write(pickle.dumps(data))
    f.close()

class FaceRecognitionThread(threading.Thread):

    def __init__(self, face_encoders_data, video_stream=None):
        super().__init__(name="face-recogniton-thread")

        self.face_encoders_data = face_encoders_data
        self.stop_flag = False  # признак необходимости завершения потока
        self.frame = None       # текущий кадр
        self.names = None       # список найденных лиц
        self.faces = None       # контуры найденных лиц
        self.landmarks = None   # опорные точки найденных лиц

        # если видеопоток не задан, используем камеру
        if video_stream is None:
            print("Streaming started")
            self.video_stream = cv2.VideoCapture(config.cam_id)
            print(self.video_stream)
        else:
            self.video_stream = video_stream

    def run(self):
        while True:
            ret, img = self.video_stream.read()

            if img is None:
                continue

            # для увеличения скорости распознавания, уменьшим кадр
            small_frame = cv2.resize(img, (0, 0), fx=config.scale, fy=config.scale)
            # выполним преобразование в формат RGB
            rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
            # найдем контуры лиц
            faces = face_recognition.face_locations(rgb_small_frame)
            # получим координаты опорных точек и вычислим эмбединги

            landmarks = face_recognition.face_landmarks(rgb_small_frame, faces, model=config.model)
            encodings = face_recognition.face_encodings(rgb_small_frame, faces, model=config.model)

            names = []
            # Для каждого вычисленного эмбединга
            for encoding in encodings:
                # найдем совпадение с известными
                matches = face_recognition.compare_faces(self.face_encoders_data["encodings"],
                                                         encoding, tolerance=config.tolerance)
                name = "Unknown"
                if True in matches:
                    # если такие имеют место быть, запомним их индексы
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # и по каждому индексу
                    for i in matchedIdxs:
                        # получим имя, которое будем использовать как ключ словаря
                        name = self.face_encoders_data["names"][i]
                        # для увеличения количества совпадений
                        counts[name] = counts.get(name, 0) + 1
                    # имя с максимальным количеством совпадений будет являться результатом распознования
                    name = max(counts, key=counts.get)

                # добавим его в список имен
                names.append(name)

            # запомним текущий кадр и обнулим старую информацию
            self.frame = img
            self.faces = []
            self.names = []
            self.landmarks = []

            # добавим новую информаци, пересчитав координаты относительно исходного кадра
            inv_scale = int(1/config.scale)
            for (top, right, bottom, left), name, landmark in zip(faces, names, landmarks):
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale
                self.faces.append((top, right, bottom, left))
                self.names.append(name)

                for k in landmark:
                    points = []
                    for t in landmark[k]:
                        points.append((t[0] * inv_scale, t[1] * inv_scale))
                    landmark[k] = points
                self.landmarks.append(landmark)

            if self.stop_flag:
                break

        if self.video_stream is not None:
            self.video_stream.release()

    def stop(self):
        self.stop_flag = True

    def get_result(self):
        return {'frame': self.frame, 'names': self.names, 'faces': self.faces, 'landmarks': self.landmarks}
