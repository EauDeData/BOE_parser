import os
import cv2
import pdf2image
import numpy as np
import cv2
import zipfile
import warnings

def read_img(path, how = str):
    img = pdf2image.convert_from_path(path.strip())
    return {how(i): np.array(img[i]) for i in range(len(img))}

def save_numpy(pages, original_path, replace_expr = '("images", "numpy")'):
    new_path = original_path.replace(*eval(replace_expr)).replace('.pdf', '.npz')
    if os.path.exists(new_path): return True
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    np.savez_compressed(new_path, **pages)

