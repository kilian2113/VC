# Práctica 2

## **TAREA 1**
### **Enunciado:** Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil. Resalta con alguna primitiva gráfica en la imagen de Canny las filas que cumplen dicha condición.

Para llevar a cabo esta tarea, hemos comenzado leyendo la imagen original, convirtiéndola a escala de grises y aplicando el algoritmo de detección de bordes de Canny para obtener una imagen binaria.
Con el objetivo de mantener un código limpio y reutilizable para la siguiente tarea, hemos dividido la lógica en varias funciones auxiliares:

max_row(): Realiza el conteo de píxeles blancos por fila con la función reduce() de OpenCV, que suma los valores horizontalmente. Tras normalizar estos datos, calcula el valor máximo, establece el umbral requerido del 90% y devuelve las filas, el umbral y los índices de las filas que cumplen esta condición.

color_img_row(): Convierte la imagen resultante de Canny a RGB. Después, modifica  los valores de la matriz para colorear las filas indicadas.

plot_counts(): Representa el porcentaje de píxeles blancos por fila e incluye una línea horizontal discontinua que ilustra de forma clara la posición del umbral calculado.

show_full_result(): Crea una única ventana dividida en dos secciones: en la parte izquierda muestra la imagen procesada devuelta por color_img_row(), y en la parte derecha integra la gráfica llamando internamente a la función plot_counts().

En el bloque principal del programa, llamamos a estas funciones de forma secuencial.


## **TAREA 2**
### **Enunciado:** Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. Visualiza los resultados obtenidos para la imagen (o una de tu elección) con Canny y Sobel tras umbralizar ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

Para esta segunda tarea, el primer paso ha consistido en procesar la imagen original en escala de grises. Para evitar que el ruido afecte a la detección de bordes, aplicamos un suavizado inicial mediante un filtro Gaussiano. A continuación, calculamos los gradientes horizontales y verticales utilizando el operador de Sobel, combinamos ambos resultados y convertimos la matriz resultante a un formato de 8 bits sin signo. Finalmente, aplicamos un umbral binario para aislar los bordes más representativos.

Al igual que en la tarea anterior, hemos estructurado el código creando nuevas funciones auxiliares:

max_col(): De forma análoga a la función creada para las filas, esta función calcula matemáticamente el máximo de píxeles blancos por columna. Utiliza reduce() de OpenCV para sumar verticalmente, normaliza los resultados dividiendo por el alto de la imagen y determina qué columnas superan el límite del 90% respecto al valor máximo detectado.

color_img_col(): Pasa la imagen umbralizada a un formato de tres canales y emplea la vectorización de NumPy para colorear simultáneamente todas las columnas que superan el umbral.

En el bloque principal del programa, mostramos una comparativa inicial entre la imagen de Sobel antes y después de aplicar la umbralización. Tras esto, reutilizamos las funciones max_row() y color_img_row() de la Tarea 1 junto con las nuevas funciones de columnas para obtener las imágenes y gráficas.

Finalmente, hacemos uso de show_full_result() para generar la visualización final. Esta nos permite desplegar dos ventanas independientes: una dedicada al análisis horizontal (filas) y otra al vertical (columnas), mostrando en cada una la imagen con sus respectivas marcas y la gráfica con la línea del umbral.


## **TAREA 3**
### **Enunciado:** Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.



#### *Bilbliografía*
- https://shimat.github.io/opencvsharp_docs/html/7bb05237-7ff6-0e19-bfeb-36ea352b3051.htm 

Autores: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)