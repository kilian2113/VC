# Práctica 4
# Práctica 4 · Detector de matrículas con YOLO

Notebook: `VC_P4_YOLO.ipynb`

Detector de matrículas entrenado con YOLO11n (fine-tuning desde los pesos de COCO) a una
resolución de 1280 px. El modelo entrenado está incluido en el repositorio, por lo que
**no hace falta volver a entrenar** para ver los resultados ni para usar el detector.

## Resultados (conjunto de test)

| Métrica | Valor |
|---|---|
| Precisión | 0.873 |
| Exhaustividad (recall) | 0.806 |
| F1 | 0.838 |
| mAP@0.5 | 0.837 |
| mAP@0.5:0.95 | 0.493 |

## Dataset (no incluido en el repositorio)

Las imágenes no se suben porque contienen matrículas legibles (datos personales). Hay que
colocar el dataset original dentro de la carpeta `P4` con esta estructura:

```
P4/
├── Imagenes VC/
│   ├── images/        # fotos .jpg / .jpeg
│   └── annotations/   # un .json por imagen, con el mismo nombre
├── runs/              # incluido en el repositorio
└── VC_P4_YOLO.ipynb
```

La celda 3 genera automáticamente la carpeta `dataset_yolo/` a partir de `Imagenes VC/`:
- convierte los JSON a etiquetas `.txt` de YOLO,
- divide las imágenes en train/val/test (70/15/15) con la semilla 42, por lo que la
  división es idéntica en cualquier ordenador,
- descarta las matrículas de menos del 1 % del ancho de la imagen (ilegibles),
- reduce las imágenes a 1280 px de lado mayor,
- escribe el `data.yaml` con las rutas del ordenador actual.

Si `dataset_yolo/` ya existe, la celda 3 no lo regenera. Para regenerarlo, borrar esa carpeta.

## Ver los resultados sin entrenar

1. Comprobar en la celda 1 que `ENTRENAR = False`.
2. Ejecutar todas las celdas en orden.

La celda 5 carga el modelo ya entrenado desde
`runs/matriculas_yolo11n_1280_min001/weights/best.pt`. El resto de celdas muestra las
curvas de entrenamiento, la tabla de métricas en val y test, las gráficas de evaluación
y ejemplos de predicciones.

> Sin el dataset solo funcionan las celdas 1, 2, 5 y 6 (cargar el modelo y ver las
> curvas de entrenamiento). Para las métricas y las predicciones (celdas 7-10) hace
> falta el dataset del apartado 2.

## Volver a entrenar

1. En la celda 1, poner `ENTRENAR = True` y cambiar `RUN_NAME` para no mezclarlo con el
   entrenamiento existente.
2. Ejecutar todas las celdas.

En una RTX 3050 tarda unas **2,5 horas** (120 épocas como máximo, con early stopping).
Si aparece `CUDA out of memory`, bajar `BATCH` (por ejemplo a 4) o `IMGSZ` (por ejemplo a 960).
En Windows, si el entrenamiento se queda colgado al empezar, poner `WORKERS = 0`.

## Contenido de `runs/`

| Carpeta | Contenido |
|---|---|
| `matriculas_yolo11n_1280_min001/` | Modelo (`weights/best.pt`), métricas por época (`results.csv`, `results.png`), configuración (`args.yaml`) y gráficas |
| `..._val`, `..._test` | Curvas PR y F1 con el umbral por defecto (cálculo del mAP) |
| `..._val_conf0.35`, `..._test_conf0.35` | Matriz de confusión con el umbral de uso (0.35) |

## Usar el detector en otro código

```python
from ultralytics import YOLO

detector_matriculas = YOLO("runs/matriculas_yolo11n_1280_min001/weights/best.pt")
resultados = detector_matriculas.predict("imagen.jpg", conf=0.35, imgsz=1280)
for caja in resultados[0].boxes:
    print(caja.xyxy[0].tolist(), float(caja.conf))
```

#### *Bibliografía*
- 

**Autores**: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)