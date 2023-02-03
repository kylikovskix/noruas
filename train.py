import argparse
from face_recognition_thread import *

if __name__ == '__main__':
    # разбор агрументов переданных при запуске программы
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces', type=str,
                        help='Каталог фотографий для тренировки модели распознавания лиц')
    parser.add_argument('--face_enc', type=str,
                        help='Файл данных обученной модели распознавания лиц')
    args = parser.parse_args()

    if args.face_enc is not None:
        face_enc_name = args.face_enc
    else:
        face_enc_name = 'face_enc'

    if args.faces is not None:
        faces_path = args.faces
    else:
        faces_path = 'faces'
    # выполним тренировку модели
    face_recognition_train(faces_path, face_enc_name)


