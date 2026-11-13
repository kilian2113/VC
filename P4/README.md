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
El conjunto de datos, al cual se puede acceder desde este [enlace](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI) contiene **2418 imágenes** recopiladas una a una con matrículas etiquetadas manualmente en formato **YOLO**. Para el entrenamiento, se han utilizado 2387 imágenes ya que cuando se comenzó con el entrenamiento no estaban todas las imágenes disponibles, este dataset se dividió en:

| Partición | Imágenes | Matrículas | Porcentaje |
|---|---|---|---|
| Entrenamiento | 1670 | 3244 | 70% |
| Validación | 358 | 712 | 15% |
| Test | 359 | 700 | 15% |
| **Total** | **2387** | **4656** | 100% |

**YOLO11n** se entrena directamente con anotaciones YOLO. RT-DETR (implementación de Baidu) necesita anotaciones en **formato COCO** para poder entrenar. La conversión se hace con el script [`yolo_a_coco.py`](./yolo_a_coco.py).

### Entrenamiento

El entrenamiento y la evaluación de YOLO están en [`Entrenamiento_YOLO.ipynb`](./Entrenamiento_YOLO.ipynb).

| Parámetro | YOLO11n | Decisión y efecto |
|---|---|---|
| Modelo base | `yolo11n.pt` (preentrenado en COCO) | Se hace *fine-tuning*: el modelo ya sabe detectar objetos y solo aprende a detectar matrículas. |
| Épocas | 120 | Máximo de épocas. |
| *Early stopping* | 20 épocas (`PATIENCE`) | Si el mAP en validación no mejora en 20 épocas, se detiene el entrenamiento. |
| Resolución (`IMGSZ`) | 1280 (y 640 para comparar) | Las matrículas ocupan una parte muy pequeña de la imagen. A más resolución, más píxeles tiene cada matrícula. |
| Lote (`BATCH`) | 8 | Fijo para no superar los 6 GB de VRAM de la GPU a 1280. |
| Semilla | 42 | La división train/val/test es siempre la misma. |
| Pesos finales | `best.pt` | Se guarda la época con mejor resultado en validación, no la última. |

Otras decisiones sobre los datos:
- **Se descartan las matrículas con un ancho menor al 1 % del de la imagen** (`MIN_ANCHO_REL = 0.01`): son ilegibles y enseñarían al modelo a buscar matrículas que ni siquiera se pueden leer.
- **Las imágenes se guardan con el lado mayor reducido a 1280 px**, la resolución de entrenamiento. Así no se cargan imágenes más grandes de lo que el modelo va a usar.

Se entrenaron dos modelos YOLO11n con la misma configuración y solo distinta resolución: **1280** (el que se usa en el vídeo) y **640** (la misma resolución que RT-DETR, para que la comparación entre los dos sea justa).

#### Análisis del entrenamiento de YOLO

<div align="center">
<img src="runs/curvas_entrenamiento_yolo.png" width="800" alt="Curvas de entrenamiento de YOLO11n a 640 y 1280">
</div>

Cada fila de la figura muestra una variable: a la izquierda se calcula sobre las imágenes de entrenamiento (train) y a la derecha sobre las de validación (val), al final de cada época. Las cuatro últimas gráficas son las métricas en validación. La pérdida de clasificación de train está en escala logarítmica, porque su valor en la primera época es mucho mayor que el resto.

**Qué mide cada gráfica:**

- **Pérdida de localización (box_loss):** cuánto se diferencian las cajas predichas de las reales, a partir de su solape (IoU). Cuanto más baja, mejor ajustadas están las cajas.
- **Pérdida de clasificación (cls_loss):** cuánto se equivoca el modelo al decidir si en una zona hay una matrícula o es fondo, es decir, al asignar la confianza.
- **Pérdida DFL (Distribution Focal Loss):** afina la posición exacta de cada borde de la caja. El modelo predice para cada borde una distribución de posibles posiciones, y esta pérdida es baja cuando esa distribución se concentra en la posición real.
- **Precisión (val):** de todas las detecciones del modelo, qué proporción son matrículas reales. Ultralytics la calcula con el umbral de confianza que maximiza F1.
- **Exhaustividad (val):** de todas las matrículas reales, qué proporción detecta el modelo. Se calcula con el mismo umbral que la precisión.
- **mAP@0.5 (val):** área bajo la curva precisión-exhaustividad, contando una detección como correcta si su IoU con la caja real es al menos 0.5. Resume el detector con todos los umbrales de confianza a la vez.
- **mAP@0.5:0.95 (val):** media del mAP con umbrales de IoU de 0.5 a 0.95. Exige cajas cada vez más ajustadas, así que además de si encuentra la matrícula, mide lo bien colocada que está la caja.

En train, las pérdidas se calculan sobre imágenes con aumento de datos, y en val sobre imágenes sin modificar, que el modelo no usa para aprender. Comparar las dos permite detectar el **sobreajuste**: si la pérdida de train sigue bajando mientras la de val se estanca o sube, el modelo está aprendiendo detalles de las imágenes de entrenamiento que no le sirven para imágenes nuevas.

**Análisis**

*Pérdidas de entrenamiento.* Las tres bajan en los dos modelos hasta la última época. La de clasificación es muy alta en la primera época y cae de golpe en la segunda: la cabeza de clasificación se crea nueva al pasar de las 80 clases de COCO (con las que se preentrenó `yolo11n.pt`) a una sola, y aprende muy rápido a distinguir la matrícula del fondo. En las 10 últimas épocas, la pérdida de localización baja más deprisa, porque Ultralytics desactiva el aumento de datos por mosaico (`close_mosaic: 10`) y las imágenes de entrenamiento se vuelven más sencillas.

*Pérdidas de validación y sobreajuste.* La de clasificación baja hasta el final en los dos modelos. Las de localización y DFL, en cambio, se estabilizan hacia la época 60. En el modelo de 640 se mantienen planas hasta el final. En el de 1280 suben ligeramente (de en torno a 1,3 a 1,35 y de 1,08 a 1,10) mientras las de train siguen bajando, lo que indica un **sobreajuste leve** en la localización de las cajas. No llega a afectar a la detección, porque ninguna métrica de validación empeora y se guarda la mejor época en validación (`best.pt`).

*Métricas de validación.* Tras una caída brusca del modelo de 1280 en las primeras épocas (precisión cerca de 0,4), la precisión de los dos modelos evoluciona casi igual y termina en torno a 0,88-0,90. La diferencia está en la exhaustividad: la del modelo de 1280 queda por encima durante todo el entrenamiento (en torno a 0,80 frente a 0,72 al final), y lo mismo ocurre con el mAP@0.5 (0,85 frente a 0,76) y el mAP@0.5:0.95 (0,50 frente a 0,41). En las últimas épocas, la exhaustividad y los dos mAP están prácticamente estabilizados.

*Comparación entre resoluciones.* Las pérdidas no sirven para comparar los dos modelos, porque cada uno trabaja con imágenes y cajas de distinto tamaño: el de 640 tiene menor pérdida DFL y de clasificación de val, pero peores métricas. Las métricas muestran que las dos resoluciones cometen una proporción parecida de falsas detecciones, y que la diferencia está en la exhaustividad: a 640, las matrículas pequeñas tienen tan pocos píxeles que el modelo no llega a detectarlas.

**Resultado de cada entrenamiento** (mejor época según mAP@0.5:0.95 en validación):

| Modelo | Épocas entrenadas | Mejor época | mAP@0.5 (val) | mAP@0.5:0.95 (val) |
|---|---|---|---|---|
| YOLO11n 640 | 120 | 108 | 0.76476 | 0.41982 | 
| YOLO11n 1280 | 120 | 100 | 0.85974 | 0.50583 |

Los dos modelos completaron las 120 épocas, con su mejor resultado en validación cerca del final (época 108 en el de 640 y época 100 en el de 1280). El *early stopping* no llegó a activarse, porque necesita 20 épocas seguidas sin mejora y el entrenamiento terminó antes de que se cumplieran.

### Evaluación en test

#### YOLO11n a 1280

Métricas calculadas con Ultralytics en validación y test (umbral de confianza de uso: 0.35):

| Métrica | Validación | Test |
|---|---|---|
| Imágenes | 358 | 359 |
| Matrículas reales | 712 | 700 |
| Precisión | 0.882 | 0.873 |
| Exhaustividad (recall) | 0.803 | 0.806 |
| F1 | 0.841 | 0.838 |
| mAP@0.5 | 0.853 | 0.837 |
| mAP@0.5:0.95 | 0.510 | 0.493 |
| TP (conf = 0.35) | 594 | 584 |
| FP (conf = 0.35) | 81 | 82 |
| FN (conf = 0.35) | 118 | 116 |
| Precisión (conf = 0.35) | 0.880 | 0.877 |
| Exhaustividad (conf = 0.35) | 0.834 | 0.834 |
| F1 (conf = 0.35) | 0.857 | 0.855 |
| Inferencia (ms/img) | 14.398 | 14.461 |

TP (verdaderos positivos) son las matrículas detectadas correctamente, FP (falsos positivos) las detecciones sin matrícula y FN (falsos negativos) las matrículas no detectadas. Las filas con "conf = 0.35" usan el umbral de uso del modelo. El resto las calcula Ultralytics con el umbral que maximiza F1, salvo los mAP, que no dependen del umbral.

Los valores de validación difieren ligeramente de los de la tabla de entrenamiento (por ejemplo, mAP@0.5 de 0.853 frente a 0.860), porque durante el entrenamiento Ultralytics evalúa en media precisión (FP16) y aquí en precisión completa.

Los resultados en validación y test son muy parecidos: test queda ligeramente por debajo en mAP@0.5 (0.837 frente a 0.853) y mAP@0.5:0.95 (0.493 frente a 0.510), y con el umbral de 0.35 la exhaustividad es idéntica (0.834). Es lo esperado, porque la mejor época se eligió con validación, mientras que las imágenes de test no han intervenido en el entrenamiento. Como la diferencia es pequeña, el modelo generaliza a imágenes nuevas, y los resultados de test representan su comportamiento en la práctica.

<div align="center">
<img src="runs/matriculas_yolo11n_1280_min001_test_conf0.35/confusion_matrix.png" width="450" alt="Matriz de confusión en test">
</div>

La matriz de confusión muestra los resultados en test con el umbral de confianza de 0.35. Las filas indican lo que predijo el modelo (*Predicted*) y las columnas lo que había realmente en la imagen (*True*):

- **584 (plate / plate):** matrículas que el modelo detectó correctamente (verdaderos positivos, TP).
- **116 (background / plate):** matrículas reales que el modelo no detectó y tomó como fondo (falsos negativos, FN).
- **82 (plate / background):** detecciones del modelo en zonas donde no había ninguna matrícula (falsos positivos, FP).
- **Casilla vacía (background / background):** en detección no se cuentan los aciertos de fondo, porque el fondo es todo lo que no es una matrícula y no se puede contar como objetos.

De las 700 matrículas de test, el modelo detecta 584 (exhaustividad de 0.834). De sus 666 detecciones, 584 son correctas (precisión de 0.877). Hay más matrículas sin detectar (116) que detecciones falsas (82), igual que en el entrenamiento, donde la precisión quedaba por encima de la exhaustividad: el error principal del modelo es no detectar matrículas.

<div align="center">
<img src="runs/matriculas_yolo11n_1280_min001_test/BoxPR_curve.png" width="450" alt="Curva precisión-exhaustividad en test">
</div>

**Curva precisión-exhaustividad.** Cada punto corresponde a un umbral de confianza: al bajarlo, el modelo detecta más matrículas (más exhaustividad) pero comete más errores (menos precisión). En nuestro modelo, la precisión se mantiene en torno a 0,9 hasta una exhaustividad de 0,8, es decir, las detecciones con confianza media y alta son casi siempre correctas. A partir de ahí la curva cae de golpe: para encontrar más matrículas habría que aceptar detecciones de confianza muy baja, en su mayoría falsas. La curva termina en una exhaustividad de en torno a 0,93, así que en torno al 7 % de las matrículas no se detectan con ningún umbral. El área bajo la curva es el mAP@0.5 en test: 0.837.

<div align="center">
<img src="runs/matriculas_yolo11n_1280_min001_test/BoxF1_curve.png" width="450" alt="Curva F1-confianza en test">
</div>

**Curva F1-confianza.** Muestra el F1 (equilibrio entre precisión y exhaustividad) para cada umbral. El máximo es **0.84 con un umbral de 0.373**, y la curva es bastante plana a su alrededor (F1 por encima de 0,8 entre umbrales de 0,2 y 0,55, aproximadamente), así que el resultado no depende mucho del umbral exacto. Con umbrales muy bajos el F1 baja por las detecciones falsas, y con umbrales por encima de 0,6 cae rápidamente porque el modelo descarta la mayoría de las matrículas. El umbral de uso, 0.35, está en la zona plana y muy cerca del máximo, lo que justifica haberlo elegido.

**Exhaustividad según el tamaño de la matrícula** (ancho en píxeles a la resolución de entrada del modelo, en test, con conf = 0.35):

| Tamaño | Exhaustividad | Matrículas |
|---|---|---|
| <20 px | 0.520 | 148 |
| 20-40 px | 0.860 | 171 |
| 40-80 px | 0.864 | 169 |
| >80 px | 0.953 | 212 |

La exhaustividad depende del tamaño: el modelo solo detecta la mitad de las matrículas de menos de 20 px (0.520), frente a 0.860 a partir de 20 px y 0.953 por encima de 80 px. Las matrículas pequeñas son la principal causa de los FN de la matriz de confusión. Esto también explica la diferencia con el modelo de 640: los tamaños están medidos a 1280, así que a 640 las matrículas de 20 a 40 px pasarían a medir menos de 20 px, el grupo en el que el modelo más falla.

#### Efecto de la resolución: YOLO11n a 640 frente a 1280

Métricas en test calculadas con el mismo código para los dos modelos (cada uno con su umbral de confianza óptimo):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | mejor F1 | conf óptima | precisión | exhaustividad | TP | FP | FN | ms/imagen |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n (640) | 0.734 | 0.403 | 0.794 | 0.308 | 0.854 | 0.743 | 520 | 89 | 180 | 9.672 |
| YOLO11n (1280) | 0.838 | 0.496 | 0.839 | 0.366 | 0.866 | 0.814 | 570 | 88 | 130 | 18.443 |

Pasar de 640 a 1280 sube el mAP@0.5 de 0.734 a 0.838. La mejora viene de la exhaustividad (0.743 → 0.814, 50 matrículas más detectadas), mientras que las falsas detecciones son prácticamente las mismas (89 frente a 88 FP). Coincide con lo visto en el entrenamiento y en la tabla por tamaños: más resolución sirve para encontrar las matrículas pequeñas, no para evitar falsos positivos. El coste es el tiempo, que casi se duplica (de 9.672 a 18.443 ms por imagen).

En el vídeo se usa el modelo de **1280**, porque detecta más matrículas con la misma precisión. El tiempo extra sigue siendo bajo para procesar vídeo.


### RT-DETR (tarea extra)

RT-DETR es un detector basado en *transformers*. A diferencia de YOLO, predice directamente un conjunto fijo de cajas y no necesita eliminar detecciones duplicadas al final (NMS). Se ha usado la implementación oficial del [repositorio de RT-DETR](https://github.com/lyuwenyu/RT-DETR).

#### Modelo elegido

Se ha usado **RT-DETRv2 en su tamaño S**, cuyo *backbone* es una ResNet-18 (r18vd), partiendo de los pesos preentrenados en COCO que descarga [`load_rtdetr.py`](./load_rtdetr.py). Se eligió la versión v2 por ser la más reciente de las que incluye el repositorio, y el tamaño S por ser el más pequeño. Es el más cercano a YOLO11n y el que mejor se adapta a nuestra GPU de 6 GB. Aun así, tiene 21.854 millones de parámetros frente a los 2.605 millones de YOLO11n, unas 8 veces más, algo que hay que tener en cuenta al comparar los dos modelos.

#### Configuración del entrenamiento

La configuración está en [`configs_matriculas/rtdetrv2_r18vd_matriculas.yml`](./configs_matriculas/rtdetrv2_r18vd_matriculas.yml). Hereda la configuración oficial `rtdetrv2_r18vd_120e_coco.yml` y estas son las decisiones principales:

- **Mismas condiciones que YOLO:** se parte de los pesos de COCO (*fine-tuning*), con las mismas imágenes y la misma división train/val/test, 120 épocas y 4 *workers*. El lote se reduce de 16 (valor oficial) a 8, para que quepa en los 6 GB de VRAM y sea el mismo que en YOLO. Las anotaciones se leen de los JSON en formato COCO generados por `yolo_a_coco.py`.
- **Una sola clase** (`num_classes: 1`, `remap_mscoco_category: False`): el modelo solo detecta matrículas, y la clase se usa tal cual en lugar de convertirla a las clases de COCO.
- **Calentamiento de 200 iteraciones** en lugar de 2000: con nuestro dataset, una época tiene unas 209 iteraciones (1670 imágenes / lote 8), así que el valor oficial supondría unas 10 épocas solo de calentamiento.
- **Sin descargar los pesos de ImageNet del *backbone*** (`PResNet.pretrained: False`), porque ya vienen incluidos en los pesos de COCO.
- **Precisión mixta** (`--use-amp`), para reducir la memoria y el tiempo de entrenamiento.

El repositorio no tiene *early stopping*, así que entrena siempre las 120 épocas. Los pesos entrenados (`best.pth`) no están en el repositorio porque superan el límite de 100 MB de GitHub. Se pueden descargar desde este [enlace](https://drive.google.com/file/d/1jTYK-iaJYLwn_d8NYBSUyiTf_g1WBSIx/view?usp=sharing) y deben colocarse en `runs_rtdetr/rtdetrv2_r18_matriculas/best.pth` para ejecutar la comparación sin volver a entrenar.

#### Análisis del entrenamiento de RT-DETR

<div align="center">
<img src="runs_rtdetr/rtdetrv2_r18_matriculas/curvas_entrenamiento.png" width="800" alt="Curvas de entrenamiento de RT-DETR">
</div>

El repositorio de RT-DETR no guarda la misma información que Ultralytics. En cada época solo registra las **pérdidas de entrenamiento** y las **métricas COCO en validación** ([`log.txt`](./runs_rtdetr/rtdetrv2_r18_matriculas/log.txt)). No calcula las pérdidas en validación ni la precisión, así que esas gráficas no se pueden generar. Obtenerlas habría exigido modificar el código del repositorio y volver a entrenar. Por eso se muestran las 6 gráficas equivalentes a las de YOLO:

- **Pérdida de localización GIoU (train):** mide el solape entre la caja predicha y la real. Es la equivalente a box_loss de YOLO.
- **Pérdida de clasificación VFL (train):** mide si el modelo asigna bien la confianza, teniendo en cuenta también lo bien ajustada que está la caja. Es la equivalente a cls_loss.
- **Pérdida de regresión L1 (train):** diferencia entre las coordenadas de la caja predicha y la real. Ajusta la posición de la caja, igual que DFL en YOLO.
- **Exhaustividad media AR@100 (val):** proporción de matrículas encontradas, como media de los umbrales de IoU de 0.5 a 0.95 y con un máximo de 100 detecciones por imagen. Es parecida a la exhaustividad de YOLO, pero no igual, porque la de YOLO usa solo IoU 0.5.
- **mAP@0.5 y mAP@0.5:0.95 (val):** las mismas métricas que en YOLO.

Como las funciones de pérdida son distintas a las de YOLO, sus valores no se pueden comparar entre los dos modelos, solo su evolución.

**Análisis**

*Pérdidas de entrenamiento.* La GIoU y la L1 bajan de golpe en las primeras 5 épocas y después siguen bajando lentamente hasta el final. La VFL, en cambio, sube al principio y luego baja lentamente con más oscilaciones. Esto es posible porque su objetivo depende de la calidad de cada caja: cuanto mejor se ajustan las cajas, más confianza se le exige al modelo. Las tres pérdidas tienen dos cambios bruscos: un salto en torno a la época 80, cuando el entrenamiento se interrumpió y se reanudó desde el último checkpoint, y una bajada en las 3 últimas épocas, cuando se desactivan los aumentos de datos más fuertes (`policy.epoch: 117`).

*Métricas de validación.* Suben muy rápido en las primeras épocas y después lentamente, sin caer en ningún momento ni verse afectadas por la reanudación. Como el repositorio no calcula pérdidas en validación, el sobreajuste no se puede analizar como en YOLO, pero las métricas de validación no muestran ningún signo de que afecte a la detección.

| Modelo | Épocas entrenadas | Mejor época | mAP@0.5 (val) | mAP@0.5:0.95 (val) |
|---|---|---|---|---|
| RT-DETR r18 640 | 120 | 120 | 0.865 | 0.509 |

La mejor época es la última (120): las métricas de validación siguen subiendo hasta el final, también en las 3 últimas épocas. A diferencia de YOLO, cuyo mejor resultado se alcanzó antes (épocas 100 y 108), RT-DETR no llegó a estabilizarse del todo, así que es posible que con más épocas hubiera mejorado algo más.

### Comparación entre YOLO y RT-DETR

#### Entrenamiento

<div align="center">
<img src="runs_rtdetr/comparativa_entrenamiento_yolo_rtdetr.png" width="800" alt="mAP en validación durante el entrenamiento de YOLO y RT-DETR">
</div>

Solo se comparan el mAP@0.5 y el mAP@0.5:0.95 en validación, las únicas métricas que calculan los dos modelos. Aun así, cada uno las calcula con su propio código, así que los valores pueden diferir ligeramente.

RT-DETR es el que más rápido aprende: hacia la época 10 ya alcanza un mAP@0.5 en torno a 0,74, porque su calentamiento dura menos de una época y parte de un detector de COCO más grande. Después, RT-DETR y YOLO a 1280 evolucionan de forma parecida y terminan prácticamente igual (en torno a 0,86 y 0,85 de mAP@0.5, y 0,51 y 0,50 de mAP@0.5:0.95). En cambio, YOLO a 640 queda por debajo durante todo el entrenamiento (en torno a 0,76 y 0,41), así que a la misma resolución RT-DETR obtiene un resultado mejor.

La curva de RT-DETR es suave y la de YOLO oscila. Los dos validan con una media móvil de los pesos (EMA), pero RT-DETR los actualiza en cada lote y con una tasa de aprendizaje baja, así que el modelo apenas cambia entre épocas. YOLO acumula gradientes hasta un lote efectivo de 64 imágenes y usa aumentos de datos más agresivos, así que cada época cambia más el modelo. Estas oscilaciones se mantienen en un margen estrecho y con tendencia creciente, así que no indican ningún problema.

#### Test

<div align="center">
<img src="runs_rtdetr/comparativa_test_yolo_rtdetr.png" width="800" alt="Curvas precisión-exhaustividad y F1-confianza en test">
</div>

Las métricas se calculan con el mismo código para los tres modelos ([`RT-DETR.ipynb`](./RT-DETR.ipynb)), con cada modelo en su umbral de confianza óptimo (el máximo de F1):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | mejor F1 | conf óptima | precisión | exhaustividad | TP | FP | FN | ms/imagen | parámetros (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n (640) | 0.734 | 0.403 | 0.794 | 0.308 | 0.854 | 0.743 | 520 | 89 | 180 | 9.672 | 2.605 |
| YOLO11n (1280) | 0.838 | 0.496 | 0.839 | 0.366 | 0.866 | 0.814 | 570 | 88 | 130 | 18.443 | 2.605 |
| RT-DETR r18 (640) | 0.858 | 0.490 | 0.864 | 0.553 | 0.909 | 0.823 | 576 | 58 | 124 | 33.186 | 21.854 |

El tiempo incluye todo el proceso de cada imagen (preparación, red y filtrado de detecciones), procesando las imágenes de una en una. Por eso es mayor que el tiempo de inferencia que da Ultralytics en la evaluación de YOLO, que solo mide la red y procesa las imágenes en lotes.

**Análisis**

El efecto de la resolución en YOLO ya se analizó en su evaluación, así que aquí nos centramos en RT-DETR.

En **exhaustividad**, RT-DETR a 640 detecta casi las mismas matrículas que YOLO a 1280 (576 y 570 TP) y bastantes más que YOLO a 640 (520). Como las que YOLO pierde a 640 son sobre todo las pequeñas, esto indica que RT-DETR las detecta mejor sin necesitar más resolución. En **precisión**, RT-DETR es el mejor de los tres: 58 detecciones falsas frente a 88-89 de los dos YOLO (precisión de 0.909 frente a 0.866 y 0.854). YOLO genera muchas cajas candidatas y elimina las duplicadas al final (NMS), y algunas detecciones dudosas sobreviven a ese filtro, mientras que RT-DETR se entrena para predecir una sola caja por objeto. Por eso RT-DETR tiene el mayor mAP@0.5 (0.858). En cambio, en mAP@0.5:0.95, que exige cajas más ajustadas, YOLO a 1280 queda ligeramente por encima (0.496 frente a 0.490): con el doble de píxeles por matrícula, sus cajas se ajustan igual de bien o algo mejor.

*Curvas.* En la curva precisión-exhaustividad, RT-DETR mantiene más precisión que los dos YOLO cuando la exhaustividad es alta, y llega a la misma exhaustividad máxima que YOLO a 1280. La curva F1-confianza de RT-DETR tiene una forma distinta. Con umbrales bajos su F1 es menor, porque siempre devuelve un número fijo de predicciones por imagen, la mayoría de confianza muy baja y sin matrícula. Con umbrales altos se mantiene en torno a 0,86 en un rango amplio (aproximadamente de 0,4 a 0,75), porque asigna una confianza alta a sus detecciones correctas. Por eso su umbral óptimo (0.553) es muy distinto del de YOLO, y a cambio es menos sensible a la elección exacta del umbral.

*Coste.* RT-DETR tiene unas 8 veces más parámetros (21.854 M frente a 2.605 M) y tarda 33.186 ms por imagen, casi el doble que YOLO a 1280 (18.443 ms).

**Conclusión**

A la misma resolución, RT-DETR es el mejor detector. Frente a YOLO a 1280, su ventaja se reduce a 30 detecciones falsas menos, con la misma exhaustividad y un mAP@0.5:0.95 prácticamente igual. Teniendo en cuenta su coste, YOLO11n a 1280 ofrece un resultado casi igual siendo mucho más rápido y ligero. RT-DETR sería preferible si las detecciones falsas fueran especialmente costosas y el tiempo no fuera una limitación.

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

El vídeo se procesa fotograma a fotograma en [`VC_P4.ipynb`](./VC_P4.ipynb):

1. **Personas y vehículos:** YOLO11n (COCO) los detecta y BoT-SORT les asigna un identificador que se mantiene entre fotogramas (apartado 1).
2. **Matrículas:** nuestro detector YOLO11n a 1280 (apartado 2) busca las matrículas en el fotograma, y cada una se asigna al vehículo más pequeño que la contiene.
3. **Lectura:** PaddleOCR lee el texto de cada matrícula (apartado 3). Como una misma matrícula se lee en muchos fotogramas, el texto de cada vehículo se decide por votación entre todas sus lecturas.


<p align="center">
  <a href="https://drive.google.com/file/d/1CBm8d97C9uk3vcQ3UD7TkJbtchpKZw0z/view?usp=sharing">
    <img width="816" height="457" alt="image" src="https://github.com/user-attachments/assets/294c3e1c-a33d-4e76-963c-9c4a0713c796" />
  </a>
  <br>
  <em>Haz clic en la imagen para ver el vídeo de demostración.</em>
</p>

Los resultados se guardan en [`detecciones.csv`](./detecciones.csv), con una fila por cada objeto detectado en cada fotograma y estas columnas:

fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matrícula_en_su_caso, confianza, mx1,my1,mx2,my2, texto_matricula

El número de objetos de cada clase se obtiene contando los identificadores distintos que asigna el tracker:

| Objeto | Total |
|:---|:---|
| Persona | 20 |
| Coche | 64 |
| Moto | 6 |
| Autobús | 2 |
| Camión | 3 |
| Bicicleta | 2 |

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
