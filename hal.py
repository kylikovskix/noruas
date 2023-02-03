import OPi.GPIO as GPIO
import time

pinInBtn = 7
pinOutRed = 11
pinOutYellow = 13
pinOutGreen = 19
pinOutRelley = 21


def add_press_btn_callback(callback):
    GPIO.add_event_callback(pinInBtn, callback)


def hal_init():
    print('setup GPIO')
    # GPIO.setwarnings(False)
    GPIO.setboard(GPIO.H616)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pinInBtn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(pinOutRed, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(pinOutYellow, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(pinOutGreen, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(pinOutRelley, GPIO.OUT, initial=GPIO.HIGH)

    print('add event callback')
    GPIO.add_event_detect(pinInBtn, GPIO.FALLING, bouncetime=200)


def hal_free():
    print('cleanup GPIO')
    GPIO.cleanup()


def door_is_lock(is_lock=None):
    if is_lock is not None:
        if is_lock:
            GPIO.output(pinOutRelley, GPIO.HIGH)
        else:
            GPIO.output(pinOutRelley, GPIO.LOW)

    if GPIO.input(pinOutRelley) == GPIO.LOW:
        return True
    else:
        return False


def green_indicator_on():
    GPIO.output(pinOutGreen, GPIO.HIGH)


def green_indicator_off():
    GPIO.output(pinOutGreen, GPIO.LOW)


def yellow_indicator_on():
    GPIO.output(pinOutYellow, GPIO.HIGH)


def yellow_indicator_off():
    GPIO.output(pinOutYellow, GPIO.LOW)


def red_indicator_on():
    GPIO.output(pinOutRed, GPIO.HIGH)


def red_indicator_off():
    GPIO.output(pinOutRed, GPIO.LOW)


def door_unlock():
    green_indicator_on()
    door_is_lock(False)
    time.sleep(5)
    door_is_lock(True)
    green_indicator_off()
