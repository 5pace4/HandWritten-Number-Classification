import numpy as np
from custom_module.resize_digit import resize_digit

def find_digits(img):
    height, width = img.shape
    digit_columns = []
    is_digit = False
    start_col = 0

    for col in range(width):
        column_data = img[:, col]

        if np.any(column_data == 255):
            if not is_digit:
                start_col = col
                is_digit = True
        else:
            if is_digit:
                digit_columns.append((start_col-10, col+10))
                is_digit = False

    if is_digit:
        digit_columns.append((start_col, width))

    min_column_width = 5
    digit_columns = [(start, end) for start, end in digit_columns if (end - start) > min_column_width]
    digits = [resize_digit(img[:, start:end]) for start, end in digit_columns]

    return digits
