# Práctica 5 · Biometría: Detección y caracterización de caras

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat-square)
![Mediapipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-5C3EE8?style=flat-square&logo=google&logoColor=white)

---

## Índice

1. [Tarea 1. Simón dice emociones](#tarea-1-simón-dice-emociones)
   - [Entrenamiento](#entrenamiento)
   - [Filtro](#filtro)
2. [Tarea 2. ¿Qué Windows eres?](#tarea-2-qué-windows-eres)
3. [Creación del entorno](#creación-del-entorno)
4. [Referencias y recursos externos](#referencias-y-recursos-externos)

---

## Tarea 1. Simón dice emociones

### Entrenamiento

*completar*

### Filtro

*completar*

---

## Tarea 2. ¿Qué Windows eres?

Esta tarea consiste en desarrollar un filtro interactivo basado en el reconocimiento facial, que superpone diferentes imágenes representativas de las versiones de Windows sobre el rostro detectado en tiempo real. Cuando se detecta una cara, se inicia una ruleta que muestra aleatoriamente las distintas versiones de Windows hasta seleccionar una de ellas. Una vez finalizada la selección, la imagen correspondiente se coloca sobre el rostro y se reproduce el sonido característico de dicha versión.

Para implementar el reconocimiento facial se utiliza MediaPipe, concretamente su herramienta **Face Mesh**, que permite detectar numerosos puntos de referencia (landmarks) distribuidos por el rostro. A partir de estos puntos se calcula la posición y las dimensiones de la cara, delimitándola mediante un rectángulo que permite determinar su ubicación dentro de la imagen capturada por la cámara.

Una vez localizada la cara, se emplea OpenCV para procesar los fotogramas de la cámara y superponer sobre la frente de la persona la imagen de la versión de Windows seleccionada. Para ello, se calcula el tamaño adecuado de la imagen en función de las dimensiones del rostro, manteniendo sus proporciones originales y ajustando su posición para conseguir una superposición correcta. Además, se tiene en cuenta el canal alfa de las imágenes, lo que permite conservar la transparencia de los archivos PNG y obtener una integración visual más limpia.

Para evitar que la ruleta se reinicie continuamente mientras el rostro permanece frente a la cámara, se implementa un control sobre la detección facial y las posibles ausencias del rostro. De esta manera, la selección se inicia al detectar una cara y el resultado se mantiene hasta que esta deja de detectarse durante un número determinado de fotogramas.

La selección de la versión de Windows se realiza de forma aleatoria durante un intervalo de tiempo establecido mediante la variable `DURACION_RULETA`. Una vez transcurrido este intervalo, se selecciona aleatoriamente una de las versiones disponibles y se muestra su imagen sobre el rostro. Cada versión tiene asociado un archivo de audio con su sonido característico, cuya reproducción se gestiona mediante *Pygame*. La asociación entre imágenes y sonidos se realiza utilizando el mismo nombre de archivo para ambos recursos.

Adicionalmente se han añadido dos opciones de teclado:

  - **Tecla 'r'**: reinicia la ruleta y permite realizar una nueva selección.
  - **Tecla 'f'**: activa la visualización de los landmarks detectados en el rostro mediante la función `dibujar_landmarks()`

Para el desarrollo del filtro también se utilizan otros módulos de Python, como **Random**, **OS**, **Glob** y **Time**, que permiten gestionar, respectivamente, la selección aleatoria, el acceso y la organización de archivos, la búsqueda de recursos y el control de los tiempos de ejecución.


Los recursos necesarios para el funcionamiento del filtro se encuentran organizados en dos carpetas: [`fotos`](./fotos), que contiene las imágenes de las diferentes versiones de Windows, y [`audios`](./audios), donde se almacenan sus respectivos sonidos en formato WAV. Para garantizar la correcta asociación entre ambos recursos, cada imagen debe tener un archivo de audio con el mismo nombre.

Por último, todo el código correspondiente al desarrollo e implementación del filtro se encuentra en el archivo [`VC_P5.ipynb`](./VC_P5.ipynb).

---

## Creación del entorno

Para ejecutar la práctica, es necesario crear un entorno virtual de Conda con Python 3.11 e instalar las dependencias utilizadas.

```bash
conda create -n enviroment python=3.11 -y
conda activate enviroment
python -m pip install pandas
python -m pip install torch
python -m pip install mediapipe==0.10.21
python -m pip install opencv-python pygame ipykernel
python -m ipykernel install --user --name enviroment --display-name "enviroment"
```

---

## Referencias y recursos externos

Para la realización de esta práctica se han utilizado:
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker 
- https://www.kdef.se/
- https://www.youtube.com/watch?v=Bwm4tZ4fjyk&t=6s

---


<div align="center">

**Práctica 5 · Visión por Computador**

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
