from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Esquinas exteriores de los ojos en la malla de Mediapipe (referencia para normalizar)
OJO_A = 33
OJO_B = 263


# Crea el Face Landmarker: modo "imagen" para fotos sueltas, "video" para la webcam.
# num_caras > 1 permite quedarse con la del jugador cuando hay varias.
def crear_landmarker(ruta_modelo, modo="imagen", num_caras=1):
    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No encuentro el modelo {ruta_modelo.resolve()}")
    modos = {"imagen": VisionRunningMode.IMAGE, "video": VisionRunningMode.VIDEO}
    opciones = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ruta_modelo)),
        running_mode=modos[modo],
        num_faces=num_caras,
        min_face_detection_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(opciones)


# Centra los landmarks entre los ojos, los gira para que los ojos queden horizontales
# y los escala para que la distancia entre ojos valga 1. Entrada: array (N, 3) en píxeles.
def normalizar_landmarks(puntos):
    p = np.asarray(puntos, dtype=np.float32).copy()
    a, b = p[OJO_A, :2], p[OJO_B, :2]
    centro = (a + b) / 2.0
    dx, dy = b - a
    distancia = float(np.hypot(dx, dy))
    if distancia < 1e-6:
        raise ValueError("Distancia entre ojos nula: landmarks no válidos")

    # Centrar
    p[:, 0] -= centro[0]
    p[:, 1] -= centro[1]
    p[:, 2] -= p[:, 2].mean()

    # Girar
    angulo = np.arctan2(dy, dx)
    c, s = np.cos(-angulo), np.sin(-angulo)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = c * x - s * y
    p[:, 1] = s * x + c * y

    # Escalar
    p /= distancia
    return p


# Índice de la cara más grande (la más cercana a la cámara), o None si no hay caras
def indice_cara_mayor(resultado):
    if not resultado.face_landmarks:
        return None
    areas = []
    for cara in resultado.face_landmarks:
        xs = [p.x for p in cara]
        ys = [p.y for p in cara]
        areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
    return int(np.argmax(areas))


# Rectángulo (x1, y1, x2, y2) en píxeles que rodea una cara
def caja_cara(resultado, indice, ancho, alto):
    cara = resultado.face_landmarks[indice]
    xs = [p.x * ancho for p in cara]
    ys = [p.y * alto for p in cara]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# Landmarks normalizados (N, 3) de una cara del resultado de Mediapipe, o None si no hay caras.
# Si no se indica la cara, se usa la más grande.
def resultado_a_arrays(resultado, ancho, alto, indice=None):
    if not resultado.face_landmarks:
        return None
    if indice is None:
        indice = indice_cara_mayor(resultado)
    cara = resultado.face_landmarks[indice]
    # x, y vienen en [0, 1] y z en una escala parecida a x: pasamos a píxeles
    puntos = np.array([[lm.x * ancho, lm.y * alto, lm.z * ancho] for lm in cara],
                      dtype=np.float32)
    return normalizar_landmarks(puntos)


# Lee una imagen en BGR (funciona con rutas de Windows con tildes o espacios)
def leer_imagen(ruta):
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


# Landmarks normalizados de una imagen BGR (landmarker en modo imagen)
def extraer_de_imagen(landmarker, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resultado = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    alto, ancho = img_bgr.shape[:2]
    return resultado_a_arrays(resultado, ancho, alto)


# Extrae los landmarks de todas las imágenes de df_usar y los guarda en un .npz.
# Si el .npz ya existe (y forzar=False) solo lo carga. Las fotos sin cara se guardan en un CSV aparte.
def extraer_dataset(df_usar, ruta_modelo, salida, forzar=False):
    salida = Path(salida)
    if salida.exists() and not forzar:
        datos = dict(np.load(salida, allow_pickle=False))
        print(f"Extracción ya hecha. Cargando {salida} ({len(datos['y'])} caras)")
        return datos

    X_lm, filas_ok, fallos = [], [], []
    with crear_landmarker(ruta_modelo, modo="imagen") as landmarker:
        total = len(df_usar)
        for i, fila in enumerate(df_usar.itertuples(index=False), 1):
            img = leer_imagen(fila.ruta)
            lm = None if img is None else extraer_de_imagen(landmarker, img)
            if lm is None:
                fallos.append({"archivo": fila.archivo, "emocion": fila.emocion, "angulo": fila.angulo,
                               "motivo": "ilegible" if img is None else "cara_no_detectada"})
            else:
                X_lm.append(lm)
                filas_ok.append(fila)
            if i % 250 == 0 or i == total:
                print(f"  procesadas {i}/{total} · fallos: {len(fallos)}")

    if not filas_ok:
        raise RuntimeError("No se detectó ninguna cara. Revisa el modelo y las rutas.")

    datos = {
        "X_lm": np.stack(X_lm),
        "y": np.array([f.emocion for f in filas_ok]),
        "persona": np.array([f.persona for f in filas_ok]),
        "sesion": np.array([f.sesion for f in filas_ok]),
        "angulo": np.array([f.angulo for f in filas_ok]),
    }
    np.savez_compressed(salida, **datos)

    ruta_fallos = salida.with_name(salida.stem + "_fallos.csv")
    pd.DataFrame(fallos, columns=["archivo", "emocion", "angulo", "motivo"]).to_csv(ruta_fallos, index=False)

    # Resumen
    print(f"\nCaras extraídas: {len(filas_ok)} de {len(df_usar)} · sin cara: {len(fallos)}")
    if fallos:
        f = pd.DataFrame(fallos)
        print("\nFallos por ángulo:\n" + f["angulo"].value_counts().to_string())
        print("\nFallos por emoción:\n" + f["emocion"].value_counts().to_string())
    print("\nCaras por emoción:\n" + pd.Series(datos["y"]).value_counts().sort_index().to_string())
    return datos