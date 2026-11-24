
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Esquinas exteriores de los ojos en la malla de Mediapipe.
# Se usan para enderezar la cara y como referencia de tamaño.
OJO_A = 33    # esquina exterior del ojo que aparece a la izquierda de la imagen
OJO_B = 263   # esquina exterior del ojo que aparece a la derecha de la imagen


def crear_landmarker(ruta_modelo="face_landmarker.task", modo="imagen", num_caras=1):
    """Crea el Face Landmarker.

    modo="imagen": para fotos sueltas (dataset).
    modo="video":  para la webcam en el filtro (fotograma a fotograma, con detect_for_video).
    num_caras: cuántas caras puede detectar a la vez (en el filtro conviene más de 1
               para poder quedarse con la del jugador, la más cercana).
    """
    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No encuentro el modelo {ruta_modelo.resolve()}. "
            "Copia face_landmarker.task a la carpeta del notebook."
        )
    modos = {"imagen": VisionRunningMode.IMAGE, "video": VisionRunningMode.VIDEO}
    opciones = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ruta_modelo)),
        running_mode=modos[modo],
        num_faces=num_caras,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(opciones)


# ---------------------------------------------------------------------------
# Normalización (la misma para el dataset y para el filtro)
# ---------------------------------------------------------------------------

def normalizar_landmarks(puntos):
    """Normaliza los landmarks para que no dependan de la posición, el tamaño
    ni la inclinación lateral de la cabeza.

    puntos: array (N, 3) en PÍXELES (x, y, z), con z en la misma escala que x.
    Devuelve un array (N, 3):
      1) centrado en el punto medio entre los ojos,
      2) girado para que la línea de los ojos quede horizontal,
      3) escalado para que la distancia entre ojos valga 1.
    """
    p = np.asarray(puntos, dtype=np.float32).copy()
    a, b = p[OJO_A, :2], p[OJO_B, :2]
    centro = (a + b) / 2.0
    dx, dy = b - a
    distancia = float(np.hypot(dx, dy))
    if distancia < 1e-6:
        raise ValueError("Distancia entre ojos nula: landmarks no válidos")

    # 1) centrar
    p[:, 0] -= centro[0]
    p[:, 1] -= centro[1]
    p[:, 2] -= p[:, 2].mean()

    # 2) girar en el plano de la imagen para que los ojos queden horizontales
    angulo = np.arctan2(dy, dx)
    c, s = np.cos(-angulo), np.sin(-angulo)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = c * x - s * y
    p[:, 1] = s * x + c * y

    # 3) escalar
    p /= distancia
    return p


def indice_cara_mayor(resultado):
    """Índice de la cara más grande de la imagen (la más cercana a la cámara), o None."""
    if not resultado.face_landmarks:
        return None
    areas = []
    for cara in resultado.face_landmarks:
        xs = [p.x for p in cara]
        ys = [p.y for p in cara]
        areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
    return int(np.argmax(areas))


def caja_cara(resultado, indice, ancho, alto):
    """Rectángulo (x1, y1, x2, y2) en píxeles que rodea la cara indicada."""
    cara = resultado.face_landmarks[indice]
    xs = [p.x * ancho for p in cara]
    ys = [p.y * alto for p in cara]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def resultado_a_arrays(resultado, ancho, alto, indice=None):
    """Convierte la salida de Mediapipe en arrays de numpy.

    indice: qué cara usar. Si es None, se usa la más grande (la más cercana a la cámara).
    Devuelve (landmarks_normalizados (N,3), blendshapes (52,), nombres_blendshapes)
    o None si no se ha detectado ninguna cara.
    """
    if not resultado.face_landmarks:
        return None
    if indice is None:
        indice = indice_cara_mayor(resultado)
    cara = resultado.face_landmarks[indice]
    # Mediapipe da x, y en [0,1] y z en una escala parecida a x: pasamos a píxeles
    puntos = np.array([[lm.x * ancho, lm.y * alto, lm.z * ancho] for lm in cara],
                      dtype=np.float32)
    lm_norm = normalizar_landmarks(puntos)

    categorias = resultado.face_blendshapes[indice]
    blend = np.array([c.score for c in categorias], dtype=np.float32)
    nombres = [c.category_name for c in categorias]
    return lm_norm, blend, nombres


def leer_imagen(ruta):
    """Lee una imagen en BGR (seguro con rutas de Windows con tildes o espacios)."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def extraer_de_imagen(landmarker, img_bgr):
    """Landmarks normalizados y blendshapes de una imagen BGR (modo imagen)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = landmarker.detect(mp_img)
    alto, ancho = img_bgr.shape[:2]
    return resultado_a_arrays(resultado, ancho, alto)


# ---------------------------------------------------------------------------
# Dataset completo
# ---------------------------------------------------------------------------

def extraer_dataset(df_usar, ruta_modelo="face_landmarker.task",
                    salida="kdef_caracteristicas.npz", forzar=False):
    """Extrae landmarks y blendshapes de todas las imágenes de df_usar.

    Si el .npz ya existe y forzar=False, lo carga sin repetir la extracción.
    Devuelve un diccionario con:
      X_lm (M, N, 3), X_bs (M, 52), y (M,), persona (M,), sesion (M,),
      angulo (M,), archivo (M,), nombres_bs (52,)
    y guarda aparte un CSV con las imágenes en las que no se detectó la cara.
    """
    salida = Path(salida)
    if salida.exists() and not forzar:
        print(f"La extracción ya estaba hecha. Cargando {salida.resolve()}")
        datos = dict(np.load(salida, allow_pickle=False))
        print(f"Caras: {len(datos['y'])} · landmarks por cara: {datos['X_lm'].shape[1]} · "
              f"blendshapes: {datos['X_bs'].shape[1]}")
        return datos

    X_lm, X_bs, nombres_bs, filas_ok, fallos = [], [], None, [], []

    with crear_landmarker(ruta_modelo, modo="imagen") as landmarker:
        total = len(df_usar)
        for i, fila in enumerate(df_usar.itertuples(index=False), 1):
            img = leer_imagen(fila.ruta)
            res = None if img is None else extraer_de_imagen(landmarker, img)
            if res is None:
                fallos.append({"archivo": fila.archivo, "emocion": fila.emocion,
                               "angulo": fila.angulo,
                               "motivo": "ilegible" if img is None else "cara_no_detectada"})
            else:
                lm, bs, nombres = res
                X_lm.append(lm)
                X_bs.append(bs)
                nombres_bs = nombres_bs or nombres
                filas_ok.append(fila)
            if i % 250 == 0 or i == total:
                print(f"  procesadas {i}/{total} · fallos: {len(fallos)}")

    if not filas_ok:
        raise RuntimeError("No se detectó ninguna cara. Revisa el modelo y las rutas.")

    datos = {
        "X_lm": np.stack(X_lm),
        "X_bs": np.stack(X_bs),
        "y": np.array([f.emocion for f in filas_ok]),
        "persona": np.array([f.persona for f in filas_ok]),
        "sesion": np.array([f.sesion for f in filas_ok]),
        "angulo": np.array([f.angulo for f in filas_ok]),
        "archivo": np.array([f.archivo for f in filas_ok]),
        "nombres_bs": np.array(nombres_bs),
    }
    np.savez_compressed(salida, **datos)

    ruta_fallos = salida.with_name(salida.stem + "_fallos.csv")
    pd.DataFrame(fallos, columns=["archivo", "emocion", "angulo", "motivo"]).to_csv(
        ruta_fallos, index=False, encoding="utf-8")

    # ---------------------------------------------------------------- resumen
    print("\n========== RESUMEN EXTRACCIÓN ==========")
    print(f"Caras extraídas:        {len(filas_ok)} de {len(df_usar)}")
    print(f"Sin cara detectada:     {len(fallos)}")
    if fallos:
        f = pd.DataFrame(fallos)
        print("\nFallos por ángulo:")
        print(f["angulo"].value_counts().to_string())
        print("\nFallos por emoción:")
        print(f["emocion"].value_counts().to_string())
    print(f"\nLandmarks por cara: {datos['X_lm'].shape[1]} · blendshapes: {datos['X_bs'].shape[1]}")
    print("\nCaras por emoción:")
    print(pd.Series(datos["y"]).value_counts().sort_index().to_string())
    print(f"\nGuardado en: {salida.resolve()}")
    print(f"Fallos en:   {ruta_fallos.resolve()}")
    return datos