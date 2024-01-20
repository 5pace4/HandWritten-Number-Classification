from PIL import Image

def resize_digit(digit, target_size=(28, 28)):
    img_pil = Image.fromarray(digit)
    resized_digit = img_pil.resize(target_size, resample=Image.LANCZOS)
    return resized_digit
