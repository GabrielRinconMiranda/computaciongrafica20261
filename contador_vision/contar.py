import cv2
import pickle
import numpy as np

# ================== PARÁMETROS PARA AJUSTAR ==================
VIDEO = 'estabilizado.mp4'
ESCALA = 0.7

BRILLO = 40           # aumento de brillo para la tienda oscura

CORTE_SUPERIOR = 0.45 # elimina el 45% superior (cielo y edificio alto)
CORTE_INFERIOR = 0.85 # elimina pasto inferior (usa 1.0 si quieres hasta abajo)

SEGUNDO_VACIO = 293
UMBRAL = 0.10
DIFERENCIA = 30
SEG_MIN = 8
SEG_LIBRE = 1
# =============================================================

with open('espacios.pkl', 'rb') as file:
    zonas = pickle.load(file)

video = cv2.VideoCapture(VIDEO)
fps = video.get(cv2.CAP_PROP_FPS) or 30

# =============================================================
# 1. FONDO: tienda vacía
# =============================================================

video.set(cv2.CAP_PROP_POS_MSEC, SEGUNDO_VACIO * 1000)

muestras = []

for _ in range(15):
    check, f = video.read()

    if check:
        f = cv2.resize(f, None, fx=ESCALA, fy=ESCALA)
        f = cv2.convertScaleAbs(f, beta=BRILLO)

        alto, ancho = f.shape[:2]
        y_inicio = int(alto * CORTE_SUPERIOR)
        y_fin = int(alto * CORTE_INFERIOR)
        f = f[y_inicio:y_fin, :]

        gris = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        muestras.append(gris)

fondo = np.median(muestras, axis=0).astype(np.uint8)
fondo = cv2.GaussianBlur(fondo, (5, 5), 0)

cv2.imwrite('fondo.png', fondo)

# volver al inicio
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Ajustar coordenadas Y de las zonas al recorte vertical
zonas_recortadas = [(x, y - y_inicio, w, h) for x, y, w, h in zonas]

kernel = np.ones((5, 5), np.uint8)

frames_ocupado = [0] * len(zonas)
frames_libre = [0] * len(zonas)
contado = [False] * len(zonas)
contador = [0] * len(zonas)

n_frame = 0

# =============================================================
# 2. PROCESAMIENTO DEL VIDEO
# =============================================================

while True:

    check, img = video.read()

    if not check:
        break

    img = cv2.resize(img, None, fx=ESCALA, fy=ESCALA)
    img = cv2.convertScaleAbs(img, beta=BRILLO)
    img = img[y_inicio:y_fin, :]

    n_frame += 1

    # =========================================================
    # comparar contra el fondo
    # =========================================================

    imgBN = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0
    )

    dif = cv2.absdiff(imgBN, fondo)

    _, imgTH = cv2.threshold(
        dif,
        DIFERENCIA,
        255,
        cv2.THRESH_BINARY
    )

    imgMedian = cv2.medianBlur(imgTH, 5)

    imgDil = cv2.dilate(
        imgMedian,
        kernel,
        iterations=2
    )

    # =========================================================
    # revisar cada zona
    # =========================================================

    for i, (x, y, w, h) in enumerate(zonas_recortadas):

        zona_binaria = imgDil[y:y+h, x:x+w]

        count = cv2.countNonZero(zona_binaria)

        porcentaje = count / (w * h)

        if porcentaje > UMBRAL:

            frames_ocupado[i] += 1
            frames_libre[i] = 0

            if frames_ocupado[i] >= SEG_MIN * fps and not contado[i]:

                contador[i] += 1
                contado[i] = True

                print(
                    f'Cliente en zona {i+1} '
                    f'al segundo {n_frame/fps:.0f} '
                    f'-> total {contador[i]}'
                )

        else:

            frames_libre[i] += 1

            if frames_libre[i] >= SEG_LIBRE * fps:
                frames_ocupado[i] = 0
                contado[i] = False

        # =====================================================
        # colores
        # =====================================================

        if contado[i]:
            color = (0, 0, 255)

        elif frames_ocupado[i] > 0:
            color = (0, 255, 255)

        else:
            color = (0, 255, 0)

        # rectángulo
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            img,
            f'Zona {i+1}: {contador[i]} | {porcentaje*100:.0f}%',
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # =========================================================
    # contador total
    # =========================================================

    cv2.putText(
        img,
        f'CLIENTES: {sum(contador)}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    # =========================================================
    # mostrar
    # =========================================================

    cv2.imshow('video', img)
    cv2.imshow('video Dilatada', cv2.bitwise_not(imgDil))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(
    'Clientes por zona:',
    contador,
    '| TOTAL:',
    sum(contador)
)