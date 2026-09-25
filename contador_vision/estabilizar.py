import cv2
import numpy as np

SEGUNDO_VACIO = 293     # el mismo de contar.py

video = cv2.VideoCapture('video.mp4')
fps = video.get(cv2.CAP_PROP_FPS) or 30
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# referencia = fotograma con la tienda vacía
video.set(cv2.CAP_PROP_POS_MSEC, SEGUNDO_VACIO * 1000)
ok, ref = video.read()
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

alto, ancho = ref.shape[:2]
salida = cv2.VideoWriter('estabilizado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto))

refGris = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
puntosRef = cv2.goodFeaturesToTrack(refGris, maxCorners=400, qualityLevel=0.01, minDistance=20)

n = 0
while True:
    ok, img = video.read()
    if not ok:
        break
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    puntos, estado, err = cv2.calcOpticalFlowPyrLK(refGris, gris, puntosRef, None,
                                                   winSize=(21, 21), maxLevel=4)
    buenosRef = puntosRef[estado == 1]
    buenosAct = puntos[estado == 1]

    M, _ = cv2.estimateAffinePartial2D(buenosAct, buenosRef)
    if M is not None:
        img = cv2.warpAffine(img, M, (ancho, alto))

    salida.write(img)
    n += 1
    if n % 300 == 0:
        print(f'{n}/{total} fotogramas')

video.release()
salida.release()
print('Listo: estabilizado.mp4')
