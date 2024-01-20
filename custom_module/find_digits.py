import numpy as np
from custom_module.resize_digit import resize_digit
from PIL import Image

def find_digits(img):
    # Get the height and width of the image
    width, height = img.size
    
    digit_columns = []
    is_digit = False
    start_col = 0

    for col in range(width):
        # Extract column data
        column_data = np.array(img.crop((col, 0, col + 1, height)))[:, 0]

        if np.any(column_data == 255):
            if not is_digit:
                start_col = col
                is_digit = True
        else:
            if is_digit:
                digit_columns.append((start_col - 10, col + 10))
                is_digit = False

    if is_digit:
        digit_columns.append((start_col, width))

    min_column_width = 5
    digit_columns = [(start, end) for start, end in digit_columns if (end - start) > min_column_width]
    digits = [resize_digit(np.array(img.crop((start, 0, end, height)))) for start, end in digit_columns]

    return digits
