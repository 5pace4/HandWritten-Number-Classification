import cv2

def resize_digit(digit, target_size=(28, 28)):
    resized_digit = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)
    return resized_digit
