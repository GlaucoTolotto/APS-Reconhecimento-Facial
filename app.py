from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import os 
import zipfile
import seaborn

path = 'datasets\yalefaces.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r')
zip_object.extractall('./')
zip_object.close()

# Convertendo imagem para cinza
imagem_teste = 'yalefaces/train/subject01.leftlight.gif'
imagem = Image.open(imagem_teste).convert('L')
type(imagem)

imagem_np = np.array(imagem, 'uint8')
cv2_plt_imshow(imagem_np)

network = cv2.dnn.readNetFromCaffe('weights/deploy.prototxt.txt', 'weights/res10_300x300_ssd_iter_140000.caffemodel')

def detecta_face(network, path_imagem, conf_min = 0.7):
  imagem = Image.open(path_imagem).convert('L')
  imagem = np.array(imagem, 'uint8')
  imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
  (h, w) = imagem.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (100, 100)), 1.0, (100,100), (104.0, 117.0, 123.0))
  network.setInput(blob)
  deteccoes = network.forward()

  face = None
  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]
    if confianca > conf_min:
      bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
      (start_x, start_y, end_x, end_y) = bbox.astype('int')
      roi = imagem[start_y:end_y, start_x:end_x]
      roi = cv2.resize(roi, (60,80))
      cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
      face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  return face, imagem
# teste_imagem = 'yalefaces/train/subject01.sad.gif'
# face, imagem = detecta_face(network, teste_imagem)
# cv2_plt_imshow(imagem)
# plt.show()

def get_image_data():
  paths = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
  faces = []
  ids = []
  for path in paths:
    face, imagem = detecta_face(network, path)
    cv2_plt_imshow(imagem)
    cv2_plt_imshow(face)

    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    ids.append(id)
    faces.append(face)
    cv2_plt_imshow(face)
  return np.array(ids), faces

ids, faces = get_image_data()

eigen_classifier = cv2.face.EigenFaceRecognizer.create()
eigen_classifier.train(faces, ids)
eigen_classifier.write('eigen_classifier.yml')

eigen_classifier = cv2.face.EigenFaceRecognizer.create()
eigen_classifier.read('eigen_classifier.yml')

imagem_teste = 'yalefaces/test/subject03.glasses.gif'

face, imagem = detecta_face(network, imagem_teste)
cv2_plt_imshow(face)


def teste_reconhecimento(imagem_teste, classificador, show_conf = False):
  face, imagem_np = detecta_face(network, imagem_teste)
  previsao, conf = classificador.predict(face)
  saida_esperada = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
  cv2.putText(imagem_np, 'Pred: ' + str(previsao), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
  cv2.putText(imagem_np, 'Exp: ' + str(saida_esperada), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0))
  if show_conf:
    print(conf)

  return imagem_np, previsao

imagem_teste = 'yalefaces/test/subject11.happy.gif'
imagem_np, previsao = teste_reconhecimento(imagem_teste, eigen_classifier, False)
cv2_plt_imshow(imagem_np)

def avalia_algoritmo(paths, classificador):
  previsoes = []
  saidas_esperadas = []
  for path in paths:
    face, imagem = detecta_face(network, path)
    previsao, conf = classificador.predict(face)
    saida_esperada = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    previsoes.append(previsao)
    saidas_esperadas.append(saida_esperada)
  return np.array(previsoes), np.array(saidas_esperadas)

paths_teste = [os.path.join('yalefaces/test', f) for f in os.listdir('yalefaces/test')]
previsoes, saidas_esperadas = avalia_algoritmo(paths_teste, eigen_classifier)

cm = confusion_matrix(saidas_esperadas, previsoes)

seaborn.heatmap(cm, annot=True);
plt.show()



