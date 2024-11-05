import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import os 
import zipfile

path = 'datasets\yalefaces.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r')
zip_object.extractall('./')
zip_object.close()

print(os.listdir('yalefaces/train'))

imagem_teste = 'yalefaces/train/subject01.leftlight.gif'
imagem = Image.open(imagem_teste).convert('L')
type(imagem)

imagem_np = np.array(imagem, 'uint8')
cv2_plt_imshow(imagem_np)
print(imagem_np.shape)