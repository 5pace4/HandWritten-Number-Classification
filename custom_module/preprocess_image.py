from PIL import Image
import numpy as np

def preprocess_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    thresholded_array = (img_array > 127) * 255
    thresholded_img = Image.fromarray(thresholded_array.astype('uint8'))
    return thresholded_img
