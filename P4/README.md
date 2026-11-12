# Práctica 4 · Detección y reconocimiento

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Detection-8A2BE2?style=flat-square)
![RT-DETR](https://img.shields.io/badge/RT--DETR-Detection-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat-square)

---

En esta práctica se aborda la detección y reconocimiento de matrículas, así como la detección de personas y vehículos mediante técnicas de visión por computador y aprendizaje profundo.

En primer lugar, se utiliza el modelo YOLO para la detección de personas y vehículos. Posteriormente, se entrena un modelo específico de YOLO para la detección de matrículas, cuyos resultados se emplean como base para desarrollar y evaluar diferentes métodos de reconocimiento de caracteres. En particular, se realiza una comparación entre los sistemas tradicionales de OCR (Optical Character Recognition) y los modelos VLM (Vision Language Models) para analizar su capacidad de lectura de matrículas.

Además, se desarrolla un sistema de lectura automática de matrículas que integra la detección mediante YOLO con el reconocimiento de texto mediante OCR.

Como ampliación del trabajo, se realizan dos tareas adicionales:

- Detección con RT-DETR: se entrena un modelo basado en esta arquitectura para la detección de matrículas y se comparan sus resultados con los obtenidos mediante YOLO.
- Anonimización de personas y matrículas: se implementa un sistema que aplica filtros sobre las regiones detectadas, permitiendo ocultar estos elementos en imágenes y vídeos.

---

## Índice

1. [Detección de personas y vehículos con YOLO](#1-detección-de-personas-y-vehículos-con-yolo)
2. [Detección de matrículas](#2-detección-de-matrículas)
3. [Reconocimiento de matriculas](#3-reconocimiento-de-matriculas)
4. [Resultados](#4-resultados)
5. [Anonimización de personas y matrículas](#5-anonimización-de-personas-y-matrículas)
6. [Especificaciones del equipo](#6-especificaciones-del-equipo)

---

## 1. Detección de personas y vehículos con YOLO

En esta primera parte se hace uso del modelo **YOLO11n** de Ultralytics para la detección y seguimiento de personas y vehículos. El objetivo es identificar estos elementos en cada fotograma del [vídeo de ejemplo proporcionado para la práctica](https://ulpgc-my.sharepoint.com/personal/mcastrillon_iusiani_ulpgc_es/_layouts/15/stream.aspx?id=%2Fpersonal%2Fmcastrillon%5Fiusiani%5Fulpgc%5Fes%2FDocuments%2FRecordings%2FC0142%2EMP4&ga=1&share=cQp0bJ6%2DGLkPQq4XHpEzwA%5FGEgUCbGUicnP93So3tDpcv8c50Q&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E3e3bece3%2Dd245%2D4d98%2D911d%2D54c2b09bd5e2) y mantener su seguimiento a lo largo de este.

Para realizar la detección se utiliza YOLO11n, una variante ligera de la arquitectura YOLO11 que permite detectar objetos de forma eficiente. El modelo identifica las clases correspondientes a personas y vehículos, obteniendo para cada detección su clase, nivel de confianza y caja delimitadora (bounding box).

Para el seguimiento de los objetos detectados se utiliza BoT-SORT, un algoritmo de seguimiento multiobjeto (Multi-Object Tracking). Su utilización resulta especialmente relevante en las últimas partes del vídeo, donde aparecen vehículos estacionados y otros vehículos circulan por delante de ellos. Esta situación dificulta el seguimiento y puede provocar la pérdida de identificadores (IDs) cuando se producen oclusiones.

Para mejorar el comportamiento del tracker ante estas situaciones, se utiliza un archivo de configuración personalizado, [`custom_botsort.yaml`](./custom_botsort.yaml), en el que se ajustan sus parámetros. Los principales ajustes realizados son:

- **Persistencia de los identificadores (track_buffer=300)**: aumenta el número de fotogramas durante los que se conserva un seguimiento perdido antes de eliminarlo. Esto permite recuperar identificadores cuando un vehículo queda oculto temporalmente.
- **Asociación de detecciones (match_thresh=0.85)**: establece el umbral utilizado durante el proceso de asociación entre detecciones y seguimientos existentes.
- **Compensación del movimiento de cámara (gmc_method=sparseOptFlow)**: utiliza flujo óptico disperso para estimar y compensar los movimientos de la cámara, mejorando la asociación de objetos entre fotogramas.
- **Reidentificación por apariencia (with_reid=True)**: habilita el uso de características visuales para ayudar a identificar objetos que han desaparecido temporalmente de la imagen.
- **Umbral de similitud de apariencia (appearance_thresh=0.25)**: establece el nivel de similitud requerido para considerar que dos detecciones corresponden al mismo objeto.
- **Umbral de proximidad (proximity_thresh=0.5)**: controla la proximidad espacial necesaria para considerar la información de apariencia durante la asociación.
Además, se utiliza **model=auto** para aprovechar las características extraídas por el propio modelo YOLO durante la reidentificación, evitando cargar un modelo externo adicional.

---

## 2. Detección de matrículas

### Conjunto de datos
https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI







---

## 4. Resultados

| Objeto | Total |
|:---|:---|
| Bicicleta | 2 |
| Bus | 2 |
| Coche | 64 |
| Moto | 6 |
| Perona | 20 |

---

## 6. Especificaciones del equipo

| Componente | Especificación |
|:---|:---|
| Sistema operativo | Windows 10 |
| CPU | Intel64 Family 6 Model 183 Stepping 1 (10 núcleos) |
| Memoria RAM | 34.1 GB |
| GPU | NVIDIA GeForce RTX 3050 (6.4 GB VRAM) |
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |

---

<div align="center">

**Práctica 4 · Visión por Computador**

*Universidad de Las Palmas de Gran Canaria · 2026/2027*

</div>

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/AliciaM05">
        <img src="https://github.com/AliciaM05.png" width="100px;" alt="Alicia María Rodríguez Trujillo"/>
        <br />
        <sub><b>Alicia María Rodríguez Trujillo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/kilian2113">
        <img src="https://github.com/kilian2113.png" width="100px;" alt="Kilian Santana Delgado"/>
        <br />
        <sub><b>Kilian Santana Delgado</b></sub>
      </a>
    </td>
  </tr>
</table>

<div align="center">
