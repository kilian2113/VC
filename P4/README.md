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
3. [Reconocimiento de matrículas](#3-reconocimiento-de-matrículas)
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

En esta sección se aborda el entrenamiento de modelos específicos para la detección de matrículas. Para ello, se utilizan dos arquitecturas diferentes, **YOLO** y **RT-DETR**, con el objetivo de comparar su rendimiento en esta tarea.

### Conjunto de datos
El conjunto de datos, al cual se puede accede desde este [enlace](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI) contiene **2418 imágenes** recopiladas una a una con matrículas etiquetadas manualmente en formato **YOLO**. Para el entrenamiento, se han utilizado 2387 imágenes ya que cuando se comenzó con el entranmiento no estaban todas las imágenes disponibles, este dataset se dividió en:

| Partición | Imágenes | Matrículas | Porcentaje |
|---|---|---|---|
| Entrenamiento | 1670 | 3244 | 70% |
| Validación | 358 | 721 | 15% |
| Test | 359 | 700 | 15% |
| **Total** | **2387** | **4665** | 100% |

**YOLO11n** se entrena directamente con anotaciones YOLO. RT-DETR (implementación de Baidu) necesita anotaciones en **formato COCO** para poder entrenar. La conversión se hace con el script [`yolo_to_coco.py`](./yolo_to_coco.py).

### Entrenamiento

Para el entrenamiento de ambos modelos se han usado la misma configuracion:
- EPOCHS = 120
- BATCH = 8
- WORKERS = 4

Los resultados de los entrenamientos se pueden ver en [`results.csv`](./runs/matriculas_yolo11n_1280_min001/results.csv) para **YOLO11n** y [`log.txt`](./runs_rtdetr/rtdetrv2_r18_matriculas/log.txt) para **RT-DETR r18**
 
### Evaluación

<div align=center>
<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/aa6ffdb7-2c18-4e5f-a42a-9e84ba41fcfb" />


| Modelo | mAP@0.5 | F1 máx. (conf.) | Precisión | Exhaustividad | TP / FP / FN | ms/imagen |
|---|---|---|---|---|---|---|
| YOLO11n (1280) | 0.838 | 0.84 (0.37) | 0.87 | 0.81 | 570 / 88 / 130 | 19.1 |
| RT-DETR r18 (640) | 0.858 | 0.86 (0.55) | 0.91 | 0.82 | 576 / 58 / 124 | 33.6 |

</div>


**Conclusiones:** **RT-DETR r18** obtiene mejores resultados de detección (mAP@0.5 de 0.858 frente a 0.838 y F1 máximo de 0.86 frente a 0.84) y es más robusto a la elección del umbral de confianza.

---

## 3. Reconocimiento de matrículas

En esta sección se lleva acabo la lectura de matrículas haciendo uso de OCR y VLM. Para ello se realiza la comparación tres modelos: **EasyOCR**, **PaddleOCR** y **SmolVLM** evaluando las mismas matrículas recortadas del dataset (a partir de las cajas anotadas). El código y detalles de la comparación se encuentra en el archivo [`OCR_vs_VLM`](./OCR_vs_VLM.ipynb).

### Resultados

<img width="1489" height="440" alt="image" src="https://github.com/user-attachments/assets/c068f25f-92e4-4754-ad05-288c86a379e6" />


| Modelo | Tasa de acierto | Precisión de caracteres | Sin Lectura | Tiempo medio (ms) | Tiempo mediana (ms) | Matrículas por segundo |
|---|---|---|---|---|---|---|
| EasyOCR | 0.553 | 0.711 | 0.240 | 20.504 | 15.415 | 48.772 |
| PaddleOCR | 0.898 | 0.931 | 0.055 | 17.778 | 14.730 | 56.249 |
| SmolVLM | 0.727 | 0.850 | 0.119 | 810.030 | 571.460 | 1.235 |

Métricas:

- **Tasa de acierto:** proporción de matrículas leídas completamente bien. Una lectura con un solo carácter mal cuenta como fallo.
- **Precisión de caracteres:** proporción media de caracteres leídos correctamente. Se calcula como 1 - CER, donde el CER (tasa de error por carácter) es el número de caracteres que hay que cambiar, quitar o añadir para convertir la lectura en la matrícula real, dividido entre la longitud de la matrícula real. A diferencia de tasa de acierto, tiene en cuenta las lecturas que fallan solo en algún carácter.
- **Sin Lectura:** proporción de matrículas en las que el modelo no devolvió ninguna matrícula con formato válido (no leyó nada o lo que leyó no encaja con el formato de matrícula).
- **Tiempo medio (ms):** tiempo medio de lectura por matrícula, en milisegundos.
- **Tiempo mediana (ms):** mediana del tiempo de lectura por matrícula. Es menos sensible que la media a los casos que tardan mucho más de lo normal.
- **Matrículas por segundo:** matrículas que el modelo puede leer por segundo (1000 / tiempo_medio_ms).

### Análisis

**PaddleOCR es el mejor en todas las columnas:** tiene la mayor tasa de acierto (0.898) y precisión de caracteres (0.931), la menor sin lectura (0.055) y el menor tiempo (14.730 ms de mediana). Estos resultado concuerdan con el modelo, ya que se trata de un OCR especializado en leer texto en imágenes, exactamente la tarea propuesta.

EasyOCR es casi igual de rápido que PaddleOCR, pero mucho menos preciso con menor tasa de acierto (0.553). Gran parte de sus fallos vienen de matrículas sin lectura (0.240): en una de cada cuatro matrículas no devuelve una matrícula válida. Si le falta o le sobra un solo carácter, el texto ya no encaja con el formato de matrícula y se descarta. En cuanto a SmolVLM, queda en medio en precisión con una tasa de acierto de 0.727, pero su precisión de caracteres (0.850) es bastante mayor. Esto indica que muchos de sus fallos son de pocos caracteres. Su gran problema es el tiempo: 571.460 ms de mediana frente a unos 15 ms de los OCR. Tiene sentido, porque un VLM es un modelo de lenguaje mucho más grande que genera la respuesta token a token. Además, su tiempo medio (810.030 ms) es mucho mayor que su mediana, así que algunas lecturas tardan bastante más que el resto.

En conclusión **PaddleOCR** es la mejor opción para en el vídeo de la práctica, debido a su gran precisión y rápidez. SmolVLM tiene la ventaja de no necesitar un modelo específico, pero con 1.235 matrículas por segundo no es viable para procesar vídeo y mucho menos si lo quisiéramos utilizar en una detección en tiempo real. Para la tarea concreta como leer matrículas, un OCR especializado es más eficiente.

---

## 4. Resultados
Cada ejecución sobre el vídeo genera una nueva fila en un **CSV**.
`fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matrícula_en_su_caso, confianza, mx1,my1,mx2,my2, texto_matricula`
| Objeto | Total |
|:---|:---|
| Bicicleta | 2 |
| Bus | 2 |
| Coche | 64 |
| Moto | 6 |
| Perona | 20 |

---

## 5. Anonimización de personas y matrículas

Como parte de la tarea extra, se ha implementado un sistema de anonimización de personas y matrículas mediante un efecto de pixelado intenso. A partir de las detecciones realizadas por el modelo YOLO, se obtiene la región de interés (ROI) correspondiente a la caja delimitadora (bounding box) de cada elemento detectado. Posteriormente, se reduce su resolución mediante cv2.resize() y se vuelve a escalar a su tamaño original utilizando la interpolación cv2.INTER_NEAREST, generando así el efecto de pixelado. Finalmente, la región anonimizada se integra de nuevo en el fotograma original. El código de esta tarea se puede encontrar en el apartado *Extra: Anonimizar personas y vehículos* del [`VC_P4.ipynb`](./VC_P4.ipynb).

<p align="center">
  <a href="https://drive.google.com/file/d/1KXbaEPX6dSREo9uKQ1JAsiv1tJC540z1/view?usp=sharing">
    <img width="775" height="433" alt="image" src="https://github.com/user-attachments/assets/f6c04e5d-d6dc-4767-a1b6-9e979b87afc4" />
  </a>
  <br>
  <em>Haz clic en la imagen para ver el vídeo de demostración.</em>
</p>

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
