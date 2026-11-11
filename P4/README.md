# Práctica 4


## Estructura del repositorio
| Archivo | Contenido |
|---|---|
| `VC_P4_YOLO.ipynb` | Preparación del dataset (división train/val/test), entrenamiento del detector de matrículas YOLO11n, curvas de entrenamiento y evaluación en validación y test |
| `YOLO.ipynb` | Procesado del vídeo: detección y seguimiento de personas y vehículos, detección de matrículas, lectura con PaddleOCR, conteo por clase, vídeo de salida y CSV |
| `Comparacion.ipynb` | Comparativa de OCR (EasyOCR, PaddleOCR y SmolVLM) sobre las matrículas del dataset: tasa de acierto, precisión por carácter y tiempos |
| `VC_P4_RT-DETR.ipynb` | Instalación de RT-DETR, curvas de su entrenamiento y comparación con YOLO en test |

### Código y configuración

| Archivo | Contenido |
|---|---|
| `load_rtdetr.py` | Descarga los pesos oficiales de RT-DETR (COCO) y carga el modelo |
| `yolo_a_coco.py` | Convierte las etiquetas del dataset de formato YOLO a COCO para entrenar RT-DETR (misma división train/val/test) |
| `configs_matriculas/rtdetrv2_r18vd_matriculas.yml` | Configuración del entrenamiento de RT-DETR (1 clase, rutas del dataset, épocas, lote, warmup) |
| `custom_bytetrack.yaml` | Configuración del seguimiento en el vídeo (BoT-SORT con re-identificación por aspecto), ajustada para que los coches aparcados no cambien de ID cuando algo pasa por delante |

### Resultados

| Archivo / carpeta | Contenido |
|---|---|
| `detecciones.csv` | Resultado del vídeo: una fila por objeto y fotograma, con clase, confianza, ID de seguimiento, caja, matrícula y texto leído |
| `runs/matriculas_yolo11n_1280_min001/` | Entrenamiento de YOLO: pesos (`weights/best.pt`), métricas por época (`results.csv`, `results.png`) y gráficas |
| `runs/matriculas_yolo11n_1280_min001_test/` | Evaluación de YOLO en test: curvas precisión-exhaustividad y F1-confianza |
| `runs/matriculas_yolo11n_1280_min001_test_conf0.35/` | Evaluación de YOLO en test con el umbral de uso (0.35): matriz de confusión |
| `runs_rtdetr/rtdetrv2_r18_matriculas/` | Entrenamiento de RT-DETR: métricas por época (`log.txt`), curvas de TensorBoard (`summary/`) y `curvas_entrenamiento.png` |
| `runs_rtdetr/comparativa_test_yolo_rtdetr.png` | Comparación YOLO vs RT-DETR en test: curvas precisión-exhaustividad y F1-confianza |
| `ocr_eval/resultados_ocr.csv` | Lectura de cada OCR en cada matrícula de la comparativa (texto, acierto, error por carácter y tiempo) |
| `ocr_eval/comparativa_metricas.png`, `ocr_eval/comparativa_tiempos.png` | Gráficas de la comparativa de OCR |

### Disponibles por enlace (no incluidos en el repositorio)

| Recurso | Motivo | Enlace |
|---|---|---|
| Vídeo resultado (`resultado.mp4`) | Supera el límite de tamaño de GitHub | *(enlace)* |
| Dataset (imágenes + anotaciones JSON) | Contiene matrículas reales; acceso restringido | *(enlace)* |
| Pesos de RT-DETR (`best.pth`) | Supera el límite de tamaño de GitHub | *(enlace)* |

### Generados al ejecutar (no incluidos)

| Carpeta | Cómo se obtiene |
|---|---|
| `dataset_yolo/` | Celda 3 de `VC_P4_YOLO.ipynb`, a partir del dataset original |
| `dataset_coco/` | `python yolo_a_coco.py` |
| `RT-DETR/` | Se clona automáticamente desde `VC_P4_RT-DETR.ipynb` |
| `weights/` | Se descarga con `load_rtdetr.py` |
| `ocr_eval/recortes/`, `ocr_eval/gt.csv` | Celdas iniciales de `Comparacion.ipynb` |

#### *Bibliografía*
- 

**Autores**: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)