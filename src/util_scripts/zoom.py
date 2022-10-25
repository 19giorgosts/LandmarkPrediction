import cv2
import os
import numpy as np

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

path="E:/LandmarkBasedRegistration/results/new_deformable2.0/"

for item in os.listdir(path):
    if 'jpg' in item:
        img = cv2.imread(path+item)
        img = zoom(img, 4)  # zoom into image for better visualization
        cv2.imwrite(os.path.join(path +'/n/'+ item), (img).astype(np.uint8))
