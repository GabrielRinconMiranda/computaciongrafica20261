import cv2
import pickle
from ultralytics import YOLO

# ================== PARÁMETROS PARA AJUSTAR ==================
VIDEO = 'estabilizado.mp4'
SEG_MIN =7        # segundos parado en la zona para contarlo como cliente
SALTO = 4          # analiza 1 de cada 2 fotogramas (más rápido)
ESCALA = 0.7       # el MISMO número que en marcar.py y contar.py
TAMANO = 960      # resolución de análisis: más alto detecta mejor a gente lejana
# =============================================================

modelo = YOLO('yolo11n.pt')      # se descarga solo la primera vez

with open('espacios.pkl', 'rb') as f:
    zonas = pickle.load(f)

zonas = [(int(x / ESCALA), int(y / ESCALA), int(w / ESCALA), int(h / ESCALA)) for x, y, w, h in zonas]

video = cv2.VideoCapture(VIDEO)
fps = video.get(cv2.CAP_PROP_FPS) or 30

tiempo_en_zona = {}              # ID de persona -> fotogramas que lleva en una zona
contados = set()                 # IDs ya contados como clientes
por_zona = [0] * len(zonas)

ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow('YOLO', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('YOLO', int(700 * ancho / alto), 700)

n = 0
while True:
    ok, img = video.read()
    if not ok:
        break
    n += 1
    if n % SALTO != 0:
        continue

    # detectar y seguir solo personas (clase 0 = persona)
    res = modelo.track(img, persist=True, classes=[0], imgsz=TAMANO, verbose=False)[0]

    if res.boxes.id is not None:
        cajas = res.boxes.xyxy.int().tolist()
        ids = res.boxes.id.int().tolist()
        for (x1, y1, x2, y2), id_persona in zip(cajas, ids):
            cx, cy = (x1 + x2) // 2, y2          # punto de los pies

            # ¿en qué zona están sus pies?
            zona = None
            for i, (x, y, w, h) in enumerate(zonas):
                if x <= cx <= x + w and y <= cy <= y + h:
                    zona = i
                    break

            if zona is not None:
                tiempo_en_zona[id_persona] = tiempo_en_zona.get(id_persona, 0) + SALTO
                if tiempo_en_zona[id_persona] >= SEG_MIN * fps and id_persona not in contados:
                    contados.add(id_persona)
                    por_zona[zona] += 1
                    print(f'Cliente ID {id_persona} en zona {zona+1} al segundo {n/fps:.0f} -> total {len(contados)}')

            color = (0, 0, 255) if id_persona in contados else (255, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'ID {id_persona}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)

    for i, (x, y, w, h) in enumerate(zonas):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{i+1}:{por_zona[i]}', (x + 3, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(img, f'CLIENTES YOLO: {len(contados)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow('YOLO', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print('Clientes por zona:', por_zona, '| TOTAL:', len(contados))
