import cv2
import pickle

ESCALA = 0.7          # achica la imagen (usa el MISMO número en contar.py)
SEGUNDO_VACIO = 293   # segundo donde la tienda está vacía

video = cv2.VideoCapture('estabilizado.mp4')
video.set(cv2.CAP_PROP_POS_MSEC, SEGUNDO_VACIO * 1000)
ok, img = video.read()
video.release()

if not ok:
    print('No se pudo leer el video. Revisa el nombre o el SEGUNDO_VACIO.')
    exit()

img = cv2.resize(img, None, fx=ESCALA, fy=ESCALA)
cv2.imwrite('fotograma.png', img)

espacios = []

while True:
    espacio = cv2.selectROI('Marca una zona (ENTER = guardar, ESC = terminar)', img, False)
    cv2.destroyAllWindows()
    if espacio[2] == 0 or espacio[3] == 0:
        break
    espacios.append(espacio)
    x, y, w, h = espacio
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, str(len(espacios)), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

with open('espacios.pkl', 'wb') as file:
    pickle.dump(espacios, file)

print(f'{len(espacios)} zonas guardadas')
