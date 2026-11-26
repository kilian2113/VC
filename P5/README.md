# Práctica 5 · Biometría: Detección y caracterización de caras

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat-square)

---


---

## Índice

1. [Tarea 1](#1-detección-de-personas-y-vehículos-con-yolo)
2. [Tarea 2. ¿Qué Windows eres?](#Tarea-2.-¿Qué-Windows-eres?)

---

## Tarea 1

---

## Tarea 2. ¿Qué Windows eres?

Esta tarea consiste en el desarrollo un filtro interactivo basado en el reconocimiento facial, cuyo objetivo es colocar diferentes imágenes representativas de las versiones de Windows sobre el rostro de una persona en tiempo real. Al detectar una cara, se inicia una ruleta que va mostrando aleatoriamente las distintas versiones de Windows hasta seleccionar una de ellas al azar. Una vez realizada la selección, la imagen correspondiente queda colocada sobre el rostro y se reproduce su sonido característico de dicha versión de Windows.

Para el reconocimiento facial se hace uso de MediaPipe, concretamente utilizando su herramienta Face Mesh. Esta tecnología permite detectar una gran cantidad de puntos de referencia (landmarks) distribuidos por el rostro. A partir de estos puntos se calcula la posición y las dimensiones de la cara, obteniendo un rectángulo que permite conocer su ubicación dentro de la imagen capturada por la cámara.

Una vez localizada la cara, se utiliza OpenCV para procesar la imagen de la cámara y colocar sobre la frente de la persona la imagen de la versión de Windows seleccionada. Para ello, se calcula el tamaño que debe tener la imagen en función de las dimensiones del rostro, manteniendo sus proporciones y adaptando su posición para que quede correctamente situada. También se tiene en cuenta el canal alfa de las imágenes, permitiendo conservar la transparencia de los archivos PNG y conseguir una superposición más limpia.

Para evitar que la ruleta se reinicie constantemente mientras la cara permanece delante de la cámara, se controla la detección facial y las posibles ausencias del rostro. De esta forma, la selección comienza cuando se detecta una cara y se mantiene el resultado hasta que la persona deja de aparecer durante varios fotogramas.

La selección de la versión de Windows se realiza de forma aleatoria durante un intervalo de tiempo determinado por la variable `DURACION_RULETA`. Cuando termina este intervalo, se establece una versión ganadora y se muestra su imagen sobre el rostro. Cada versión tiene asociado un archivo de audio que reproduce su sonido característico mediante *Pygame*. La relación entre imágenes y sonidos se realiza utilizando el mismo nombre de archivo.

Para desarrollar la práctica también se han empleado módulos como Random, OS, Glob y Time para gestionar la selección aleatoria, los archivos y los tiempos de funcionamiento.

Los recursos del filtro se encuentran organizados en dos carpetas: [`fotos`](./fotos), donde se almacenan las imágenes de las diferentes versiones de Windows, y [`audios`](./audios), donde se guardan sus respectivos sonidos en formato WAV. Para asociar correctamente cada imagen con su sonido, ambos archivos deben compartir el mismo nombre.

---

## Referencias y recursos externos

Para la realización de esta práctica se han utilizado:
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker 
- https://www.kdef.se/

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
