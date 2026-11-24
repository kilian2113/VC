import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ANGULOS_USADOS = {"S", "HL", "HR"}

EMOCIONES = {
    "AF": "miedo",
    "AN": "enfado",
    "DI": "asco",
    "HA": "alegria",
    "NE": "neutral",
    "SA": "tristeza",
    "SU": "sorpresa",
}

ANGULOS = {
    "FL": "perfil_izq",
    "HL": "semiperfil_izq",
    "S": "frente",
    "HR": "semiperfil_der",
    "FR": "perfil_der",
}

# Formato del nombre según la documentación de KDEF, p. ej. AF01ANFL o BF32HAS
#   sesión (A/B) · sexo (F/M) · número (01-35) · emoción (2 letras) · ángulo (1-2 letras)
PATRON = re.compile(r"^([AB])([FM])(\d{2})(AF|AN|DI|HA|NE|SA|SU)(FL|FR|HL|HR|S)$")

EXTENSIONES = {".jpg", ".jpeg"}


def leer_imagen(ruta: Path):
    """Lee una imagen de forma segura en Windows (rutas con tildes o espacios)."""
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception:
        return None


def analizar(ruta: Path) -> dict:
    fila = {
        "ruta": str(ruta),
        "archivo": ruta.name,
        "carpeta": ruta.parent.name,
        "sesion": None,
        "sexo": None,
        "num": None,
        "persona": None,  # sexo + número: AF01 y BF01 son la misma persona
        "codigo_emocion": None,
        "emocion": None,
        "codigo_angulo": None,
        "angulo": None,
        "ancho": None,
        "alto": None,
        "usar": False,
        "motivo_descarte": "",
    }

    m = PATRON.match(ruta.stem.upper())
    if not m:
        fila["motivo_descarte"] = "nombre_no_valido"
        return fila

    sesion, sexo, num, emo, ang = m.groups()
    fila.update(
        sesion=sesion,
        sexo=sexo,
        num=num,
        persona=f"{sexo}{num}",
        codigo_emocion=emo,
        emocion=EMOCIONES[emo],
        codigo_angulo=ang,
        angulo=ANGULOS[ang],
    )

    # La carpeta debería coincidir con los 4 primeros caracteres (p. ej. AF01)
    if fila["carpeta"].upper() != f"{sesion}{sexo}{num}":
        fila["motivo_descarte"] = "no_coincide_con_carpeta"
        return fila

    img = leer_imagen(ruta)
    if img is None:
        fila["motivo_descarte"] = "imagen_ilegible"
        return fila
    fila["alto"], fila["ancho"] = img.shape[:2]

    if ang not in ANGULOS_USADOS:
        fila["motivo_descarte"] = "perfil_completo"
        return fila

    fila["usar"] = True
    return fila


def preparar_kdef(ruta_kdef, salida="kdef_index.csv", forzar=False):
    """Indexa y filtra KDEF y devuelve la tabla (DataFrame).

    Si el CSV de salida ya existe y forzar=False, no repite el proceso:
    simplemente lo carga. Con forzar=True lo vuelve a generar.
    """
    salida = Path(salida)
    if salida.exists() and not forzar:
        print(f"La limpieza ya estaba hecha. Cargando {salida.resolve()}")
        df = pd.read_csv(salida, dtype={"num": str})
        print(f"Imágenes que se usan: {int(df['usar'].sum())} de {len(df)}")
        return df

    raiz = Path(ruta_kdef)
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {raiz}")
    if "map" in raiz.name.lower():
        raise ValueError("Parece que has indicado KDEFmap. Indica la carpeta KDEF.")

    rutas = sorted(p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES)
    otros = sorted(p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() not in EXTENSIONES)
    print(f"Imágenes encontradas: {len(rutas)}")
    if otros:
        print(f"Otros archivos ignorados ({len(otros)}): {[p.name for p in otros[:10]]}")

    filas = []
    for i, ruta in enumerate(rutas, 1):
        filas.append(analizar(ruta))
        if i % 1000 == 0:
            print(f"  procesadas {i}/{len(rutas)}")

    df = pd.DataFrame(filas)

    # Nombres repetidos (la misma foto en dos sitios)
    dup = df["archivo"].str.upper().duplicated(keep="first")
    df.loc[dup & df["usar"], "motivo_descarte"] = "duplicada"
    df.loc[dup, "usar"] = False

    df.to_csv(salida, index=False, encoding="utf-8")
    mostrar_resumen(df)
    print(f"\nTabla guardada en: {salida.resolve()}")
    return df


def mostrar_resumen(df):
    usadas = df[df["usar"]]
    print("\n========== RESUMEN ==========")
    print(f"Total de imágenes:       {len(df)}")
    print(f"Imágenes que se usan:    {len(usadas)}  (esperadas: 2940)")
    print(f"Imágenes descartadas:    {len(df) - len(usadas)}")
    print("\nMotivos de descarte:")
    for motivo, n in Counter(df.loc[~df["usar"], "motivo_descarte"]).most_common():
        print(f"  {motivo:25s} {n}")

    print(f"\nPersonas distintas: {usadas['persona'].nunique()}  "
          f"(mujeres: {usadas.loc[usadas['sexo']=='F','persona'].nunique()}, "
          f"hombres: {usadas.loc[usadas['sexo']=='M','persona'].nunique()})")

    print("\nImágenes por emoción (deberían ser iguales):")
    print(usadas["emocion"].value_counts().sort_index().to_string())

    print("\nImágenes por ángulo:")
    print(usadas["angulo"].value_counts().sort_index().to_string())

    # 2 sesiones x 7 emociones x 3 ángulos = 42 fotos por persona
    por_persona = usadas.groupby("persona").size()
    incompletas = por_persona[por_persona != 42]
    if len(incompletas):
        print("\nPersonas con un número de fotos distinto de 42:")
        print(incompletas.to_string())
    else:
        print("\nTodas las personas tienen sus 42 fotos.")

    raros = df[df["motivo_descarte"].isin(["nombre_no_valido", "no_coincide_con_carpeta",
                                           "imagen_ilegible", "duplicada"])]
    if len(raros):
        print("\nArchivos con problemas (revísalos):")
        print(raros[["carpeta", "archivo", "motivo_descarte"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa y filtra el dataset KDEF.")
    parser.add_argument("--kdef", required=True, help="Ruta a la carpeta KDEF (no KDEFmap)")
    parser.add_argument("--salida", default="kdef_index.csv", help="CSV de salida")
    parser.add_argument("--forzar", action="store_true", help="Repetir aunque ya exista el CSV")
    a = parser.parse_args()
    preparar_kdef(a.kdef, a.salida, a.forzar)