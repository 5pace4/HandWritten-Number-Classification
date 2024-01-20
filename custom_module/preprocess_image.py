import cv2
import numpy as np
from PIL import Image

def preprocess_image(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Thresholding using Otsu's method
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert to PIL Image
    thresh_pil = Image.fromarray(thresh)
    
    return thresh_pil
