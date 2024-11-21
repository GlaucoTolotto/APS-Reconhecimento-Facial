import threading
import cv2
from deepface import DeepFace


captura = cv2.VideoCapture(0, cv2.CAP_DSHOW) # variavel que seleciona camera e exibe sua imagem na tela

captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)              #variavel que determina largura da imagem a ser exibida
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)             #variavel que determina a altura da imagem a ser exibida


contador = 0                #contador que determina o tempo entre uma verificação e outra

face_permitida = False      #variavel que determina se aquela face está na lista de faces que possuem acesso

referencia = cv2.imread(["ministro_do_meio_ambiente.jpg","acesso_geral.jpg","acesso_setor.jpg"])        #variavel que armazena a referencia das faces permitidas no cofre



def verificar_face(frame):
    global face_permitida
    try:
        if DeepFace.verify(frame, referencia.copy())["verificado"]:
            face_permitida = True
        else:
            face_permitida = False
    except ValueError:
        pass




while True:
    ret, frame = captura.read()


    if ret:
        if contador % 30 == 0:
            try:
                threading.Thread(target=verificar_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        contador += 1

        if face_permitida:
            cv2.putText(frame, "Verificado", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        else:
            cv2.putText(frame, "Nao Verificado", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
        cv2.imshow("verificador",frame)

    
    tecla = cv2.waitKey(1)
    if tecla == ord("e"):
        break


cv2.destroyAllWindows()
