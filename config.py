broker = '127.0.0.1'                            # mqqt broker ip address
port = 1883                                     # mqqt broker port
client_id = ''                                  # mqqt client id
username = ''                                   # mqqt login name
password = ''                                   # mqqt login password

topic_frame_pub = 'noruas/face_recognition'     # mqqt public - video translation
topic_unlock_sub = 'noruas/door/unlock'         # mqqt subbscribe - open door

access_list = ['brus', 'arnold', 'zemfira']
black_list = ['filip']

model = "large"                                 # "large" = 64 face points or "small" - 5 face  points
scale = 0.25                                    # коэфициент уменьшения кадра при распозновании для повышения скорости
tolerance = 0.6                                 # точность распознавания лиц
cam_id = 0                                      #  индекс видеовхода (задается системой)
show_landmarks = False                          # показывать опорные точки на изображении
