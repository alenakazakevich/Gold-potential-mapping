from PIL import Image
import numpy as np
import os

def image_data(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    files = [name for name in files if '.tiff' in name]
    images = [np.array(Image.open(folder_path + '/' + file)) for file in files]
    return np.array(images)/255

