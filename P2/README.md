# Práctica 2

## **TAREA 1**
### **Enunciado:** Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil. Resalta con alguna primitiva gráfica en la imagen de Canny las filas que cumplen dicha condición.

Para llevar a cabo esta tarea, hemos comenzado leyendo la imagen original, convirtiéndola a escala de grises y aplicando el algoritmo de detección de bordes de Canny para obtener una imagen binaria.
Con el objetivo de mantener un código limpio y reutilizable para la siguiente tarea, hemos dividido la lógica en varias funciones auxiliares:

**max_row()**: Realiza el conteo de píxeles blancos por fila con la función **reduce()** de *OpenCV*, que suma los valores horizontalmente. Tras normalizar estos datos, calcula el valor máximo, establece el umbral requerido del 90% y devuelve las filas, el umbral y los índices de las filas que cumplen esta condición.

**color_img_row()**: Convierte la imagen resultante de Canny a RGB. Después, modifica  los valores de la matriz para colorear las filas indicadas.

**plot_counts()**: Representa el porcentaje de píxeles blancos por fila e incluye una línea horizontal discontinua que ilustra de forma clara la posición del umbral calculado.

**show_full_result()**: Crea una única ventana dividida en dos secciones: en la parte izquierda muestra la imagen procesada devuelta por **color_img_row()**, y en la parte derecha integra la gráfica llamando internamente a la función **plot_counts()**.

En el bloque principal del programa, llamamos a estas funciones de forma secuencial.

<img width="516" height="455" alt="resultado canny" src="https://github.com/user-attachments/assets/f1a53aeb-ee72-4b3d-8730-83d6a83d53ce" />



## **TAREA 2**
### **Enunciado:** Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. Visualiza los resultados obtenidos para la imagen (o una de tu elección) con Canny y Sobel tras umbralizar ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

Para esta segunda tarea, el primer paso ha consistido en procesar la imagen original en escala de grises. Para evitar que el ruido afecte a la detección de bordes, aplicamos un suavizado inicial mediante un filtro Gaussiano. A continuación, calculamos los gradientes horizontales y verticales utilizando el operador de Sobel, combinamos ambos resultados y convertimos la matriz resultante a un formato de 8 bits sin signo. Finalmente, aplicamos un umbral binario para aislar los bordes más representativos.

Al igual que en la tarea anterior, hemos estructurado el código creando nuevas funciones auxiliares:

**max_col()**: De forma análoga a la función creada para las filas, esta función calcula matemáticamente el máximo de píxeles blancos por columna. Utiliza **reduce()** de OpenCV para sumar verticalmente, normaliza los resultados dividiendo por el alto de la imagen y determina qué columnas superan el límite del 90% respecto al valor máximo detectado.

**color_img_col()**: Pasa la imagen umbralizada a un formato de tres canales y emplea la vectorización de NumPy para colorear simultáneamente todas las columnas que superan el umbral.

En el bloque principal del programa, mostramos una comparativa inicial entre la imagen de Sobel antes y después de aplicar la umbralización. Tras esto, reutilizamos las funciones **max_row()** y **color_img_row()** de la Tarea 1 junto con las nuevas funciones de columnas para obtener las imágenes y gráficas.

Finalmente, hacemos uso de **show_full_result()** para generar la visualización final. Esta nos permite desplegar dos ventanas independientes: una dedicada al análisis horizontal (filas) y otra al vertical (columnas), mostrando en cada una la imagen con sus respectivas marcas y la gráfica con la línea del umbral.

<img width="340" height="187" alt="2" src="https://github.com/user-attachments/assets/2ba512f1-2e90-408d-bdc1-a17b36eec3e1" /> 
<br><br>
<img width="516" height="455" alt="4" src="https://github.com/user-attachments/assets/c28a142d-37e1-4af9-9a2e-60ebd9ae7bb7" />
<img width="529" height="455" alt="3" src="https://github.com/user-attachments/assets/85fba55d-0573-42f2-bd46-3f7cbaf435a7" />


#### **Comparación entre Canny y Sobel**
Al comparar visualmente las imágenes binarias base, se observa que el método de Canny genera bordes muy finos y definidos, habitualmente de un solo píxel de grosor. Esto se debe a que su algoritmo incluye internamente una etapa matemática llamada "supresión de no máximos", diseñada específicamente para adelgazar los contornos. Por el contrario, la imagen procesada con el operador de Sobel presenta áreas blancas y bordes mucho más marcados y gruesos. Sobel calcula la magnitud del gradiente (los cambios de intensidad), y al aplicarle un umbralizado binario manual con un valor de 100, cualquier píxel cercano al borde que supere ese valor se convierte en blanco. Esto provoca que múltiples píxeles adyacentes se activen, haciendo más gruesa la línea.   

Esta diferencia inicial en el grosor de los bordes determina el resto de los resultados. Al realizar el conteo de píxeles blancos por fila, aunque ambas gráficas presentan una forma y distribución visualmente casi idéntica, los valores matemáticos difieren. Como el umbral se calcula dinámicamente como el 90% del valor máximo de cada gráfica, este se ajusta a 0.39 para Canny y a 0.38 para Sobel.

La combinación de tener bordes más gruesos en la imagen original y un valor de umbral distinto provoca que el número de filas que superan el límite sea significativamente mayor en Sobel (19 filas) que en Canny (7 filas).

Finalmente, esto explica la diferencia en la visualización de los resultados. En la imagen de Canny, al haber menos filas que superan el umbral, las líneas rojas se dibujan de manera más aislada. En la imagen de Sobel, al cumplirse la condición en múltiples filas consecutivas, las líneas se dibujan unas junto a otras, creando el efecto visual de bandas horizontales rojas mucho más gruesas.

| <img width="450" alt="output" src="https://github.com/user-attachments/assets/efa20ded-802a-4f70-8154-7c5db7b96a85" /> | <img width="450" alt="4" src="https://github.com/user-attachments/assets/73a9b50f-0035-4dc7-ad38-b566d22b41aa" /> |


## **TAREA 3**
### **Enunciado:** Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

En esta tarea libre hemos realizado un contador de cruces. Este consiste en dividir la pantalla en dos lados, *izquierda* y *derecha*, e ir detectando el movimiento de un lado de la pantalla al otro para aumentar el contador. Cada 10 cruces que se realicen sonará un pequeño pitido a modo de recompensa.

Para la detección del movimiento, se comparan los píxeles de dos fotogramas consecutivos. Para ello, primero se pasa el fotograma capturado a escala de grises, que posteriormente se suaviza mediante un filtro Gaussiano para reducir el ruido de la imagen (**GaussianBlur()**). Una vez tenemos dos fotogramas, realizamos la diferencia entre ellos mediante la función **absdiff()** de *OpenCV*.

Con la diferencia entre los dos fotogramas captados, podemos empezar a detectar el movimiento. En primer lugar, creamos una imagen binaria con ayuda de **threshold()** para luego localizar los píxeles blancos, es decir, aquellos donde ha habido movimiento, mediante la función **findNonZero()**.

Calculamos la media de las coordenadas (**x-y**) de dichos puntos para poder aproximar el centro del movimiento, que pintamos con un círculo rojo. También definimos un mínimo de 1500 píxeles con cambios entre dos fotogramas para considerar que se ha producido un movimiento real.

Para detectar el cambio de lado, tomamos como referencia la posición del centro del movimiento en el eje *x*, estableciendo un margen de 50 píxeles respecto al eje central que separa ambos lados. Por último, mediante el uso de variables guardamos el lado actual y el anterior, bloqueando el conteo hasta que haya un nuevo movimiento que se encuentre en el lado que se acaba de registrar. De esta forma, evitamos contabilizar varias veces el mismo cruce.

Para el sonido que se reproduce cada diez cruces se ha utilizado la función **Beep()** de la librería **winsound**, en Windows.


https://github.com/user-attachments/assets/0716f7d7-24cd-4d62-88f4-0d9c67256a4f


## **TAREA EXTRA**
Como extensión a la tarea de detección de movimiento, hemos hecho una pizarra virtual. Para aislar el movimiento en pantalla, partimos de la base de la Tarea 3: cálculo de la diferencia entre fotogramas consecutivos, suavizado y generación de una máscara binaria.

Con respecto a la detección del centro de movimiento, en lugar de localizar individualmente todos los píxeles blancos y calcular su media, hemos optimizado el proceso empleando la función **moments()** de OpenCV. Esta función calcula los momentos espaciales de la imagen binaria, lo que nos permite extraer directamente el área total del movimiento (mediante el parámetro *m00*) y las coordenadas de su centroide (*cx* y *cy*).

Para generar el trazo en el aire, almacenamos dinámicamente estas coordenadas en una lista llamada *path*. En cada iteración del fotograma, un bucle recorre este historial y utiliza la función **line()** para conectar cada punto con su posición inmediatamente anterior, dibujando así el recorrido del movimiento. La punta del "pincel" se indica visualmente con un círculo rojo empleando la función **circle()**.

Finalmente, para evitar que los trazos pinten la pantalla de forma permanente y se sature la imagen, se ha establecido un límite de almacenamiento con *MAX_POINT*. Una vez alcanzado este límite, elimina la coordenada más antigua de la lista con la instrucción **pop(0)**. Esto genera un efecto visual de estela que se va borrando de forma progresiva conforme el usuario se mueve.


https://github.com/user-attachments/assets/012922c8-3497-4d92-a872-eb0cb1db94b2




#### *Bilbliografía*
- https://shimat.github.io/opencvsharp_docs/html/7bb05237-7ff6-0e19-bfeb-36ea352b3051.htm
- https://chatgpt.com/share/6ac57af8-7bcc-83ed-889f-f82caf43c5f5
- https://sbirchfield.github.io/cvintro/lessons/lesson05_moments.html 

**Autores**: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)
