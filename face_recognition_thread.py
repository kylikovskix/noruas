import threading
import face_recognition
import pickle
import cv2
import os
from imutils import paths


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
        #encodings = face_recognition.face_encodings(rgb, boxes)
        encodings = face_recognition.face_encodings(rgb, boxes, model="large")
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # сохраним эмбеддинги вместе с их именами в формате словаря
    data = {"encodings": knownEncodings, "names": knownNames}
    # для сохранения данных в файл используем метод pickle
    f = open(face_enc_name, "wb")
    f.write(pickle.dumps(data))
    f.close()

class FaceRecognitionThread(threading.Thread):

    def __init__(self, face_encoders_data, video_stream=None):
        super().__init__(name="face recogniton thread")

        self.face_encoders_data = face_encoders_data
        self.stop_flag = False
        self.frame = None
        self.names = None
        self.faces = None
        self.landmarks = None

        if video_stream is None:
            print("Streaming started")
            self.video_stream = cv2.VideoCapture(0)
            print(self.video_stream)
        else:
            self.video_stream = video_stream

    def run(self):
        while True:
            ret, img = self.video_stream.read()

            if img is None:
                continue

            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # the facial embeddings for face in input
            faces = face_recognition.face_locations(rgb_small_frame)

            landmarks = face_recognition.face_landmarks(rgb_small_frame, faces, model="large")
            encodings = face_recognition.face_encodings(rgb_small_frame, faces, model="large")
            names = []
            # loop over the facial embeddings incase
            # we have multiple embeddings for multiple fcaes
            for encoding in encodings:
                # Compare encodings with encodings in data["encodings"]
                # Matches contain array with boolean values and True for the embeddings it matches closely
                # and False for rest
                matches = face_recognition.compare_faces(self.face_encoders_data["encodings"], encoding)
                # set name =inknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    # Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        # Check the names at respective indexes we stored in matchedIdxs
                        name = self.face_encoders_data["names"][i]
                        # increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    # set name which has highest count
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            self.frame = img
            self.faces = []
            self.names = []
            self.landmarks = []
            for (top, right, bottom, left), name, landmark in zip(faces, names, landmarks):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                self.faces.append((top, right, bottom, left))
                self.names.append(name)

                for k in landmark:
                    points = []
                    for t in landmark[k]:
                        points.append((t[0] * 4, t[1] * 4))
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
