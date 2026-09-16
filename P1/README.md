# Práctica 1

## **TAREA 1**
### **Enunciado:** Sin herramientas de IA, crea una imagen, p.e. de 800x800 píxeles, con la textura del tablero de ajedrez. Una vez resuelto de forma manual, resuelve la misma tarea usando un asistente de IA de tu elección (Claude, ChatGPT, Copilot, etc.). Compara ambas versiones en el informe de la práctica

Para esta tarea hemos creado una imagen de resolución 800 × 800 píxeles. Para generar el tablero recorremos la imagen mediante dos bucles 
anidados, utilizando saltos de 100 píxeles en las filas y de 200 píxeles en las columnas para evitar realizar iteraciones innecesarias.

Dentro de los bucles utilizamos condiciones para determinar qué cuadrados deben ser blancos y cuáles negros. De esta forma, 
dependiendo de si la fila es par o impar, se establece el color correspondiente a cada cudrado. Cuando se cumple la condición, se modifica el bloque de 100 × 100 píxeles 
correspondiente, estableciendo sus valores a 255 (blanco).

<img width="426" height="418" alt="image" src="https://github.com/user-attachments/assets/303304b3-b7c8-4fa0-b10c-deb8a4319ad3" />



## **TAREA 2**
### **Enunciado:** Crear una imagen estilo Mondrian (un ejemplo https://www3.gobiernodecanarias.org/medusa/ecoescuela/sa/2017/04/17/descubriendo-a-mondrian/) con las funciones de dibujo de OpenCV. No hagas uso de herramientas de IA, parte del ejemplo anterior.
Para crear la imagen con estilo Mondrian comenzamos creando una imagen negra de 3 canales, que posteriormente transformamos en blanca estableciendo el valor de los tres canales a 255.

Sobre este fondo blanco comenzamos a realizar rectángulos de diferentes tamaños y colores utilizando la función *rectangle()* de OpenCV. También utilizamos *line()* para trazar líneas de diferentes grosores y separar las distintas zonas de la composición.

En la versión hecha con IA se utiliza un planteamiento diferente, recorriendo directamente las 8 filas y 8 columnas del tablero. El tamaño de cada casilla se calcula automáticamente a partir de las dimensiones de la imagen y el número de casillas. Para alternar los colores se comprueba si la suma de la fila y la columna es par o impar, y finalmente se utiliza *rectangle()* de OpenCV para dibujar cada casilla.

<img width="328" height="389" alt="image" src="https://github.com/user-attachments/assets/dbab869d-97f8-475b-a412-f95a0eda78ba" />


## **TAREA 3**
### **Enunciado:** Pintar círculos en las posiciones del píxel más claro y oscuro de cada fotograma captado por la cámara. ¿Funciona de forma fluida o a saltos? En el segundo caso, ¿podrías acelerarlo?Si haces uso de herramientas de IA, incluye la conversación.

Para esta tarea hemos desarrollado dos versiones. En la primera, recorremos la imagen mediante dos bucles anidados y, para cada píxel, calculamos la suma de los tres canales. A partir de este valor, vamos registrando el píxel con el valor máximo y mínimo, junto con sus coordenadas.

Esta primera implementación resulta algo pesada e ineficiente, provocando pequeños saltos en la imagen debido a que se recorren todos los píxeles del frame y procesan sus tres canales. Por este motivo, desarrollamos una segunda versión utilizando funciones propias de *OpenCV*.

En esta segunda implementación, convertimos el frame capturado a escala de grises de un solo canal mediante la función *cvtColor()*, utilizando el parámetro **COLOR_BGR2GRAY**. A continuación, aplicamos sobre el nuevo frame la función *minMaxLoc()*, que nos da directamente los valores mínimo y máximo de los píxeles, así como sus coordenadas. Finalmente, al igual que en la primera versión, dibujamos un círculo sobre las coordenadas de ambos píxeles.

Esta segunda implementación presenta un funcionamiento más fluido, principalmente porque no es necesario recorrer manualmente todo el frame ni procesar cada uno de sus canales. En su lugar, utilizamos funciones de *OpenCV* que están optimizadas para este tipo de operaciones.

## **TAREA 4**
### **Enunciado:** Llevar a cabo una propuesta propia de pop art. Incluye fuentes consultadas. Si haces uso de herramientas de IA, incluye la conversación.
Para realizar esta tarea hemos creado una composición de estilo Pop Art formada por cuatro versiones diferentes del vídeo de la cámara web. Primero obtenemos las dimensiones de la cámara y reducimos la resolución a la mitad para que la composición ocupe menos espacio en pantalla. Después creamos una imagen conjunta dividida en cuatro planos.

En el plano de arriba a la izquierda dividimos la imagen en cuatro cuadrados y aplicamos diferentes combinaciones de los canales RGB, utilizando tanto los valores originales como sus valores invertidos.

En el plano de arriba a la derecha combinamos los canales de color de la imagen original y utilizamos la función *flip()* de OpenCV para invertir horizontalmente uno de los canales, creando un reflejo.

En el plano de abajo a la izquierda realizamos otra combinación de los canales de color, invirtiendo el canal azul. Además, utilizamos *circle()* de OpenCV para dibujar una serie de círculos alrededor de los bordes de la imagen, creando un marco decorativo.

Por último, en el plano de abajo a la derecha recorremos todos los píxeles de la imagen y calculamos la diferencia entre sus coordenadas horizontal y vertical (`x-y`). Dependiendo del resultado, cada píxel pertenece a una de las tres bandas diagonales definidas mediante diferentes valores límite. A cada banda diagonal se le aplica una combinación de colores diferente.


#### *Bilbliografía*
- https://docs.opencv.org/3.4.20/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707
- https://docs.opencv.org/3.4.20/d4/dc6/tutorial_py_template_matching.html
- https://chatgpt.com/share/6aaa60e4-7ac8-83eb-b346-8fe68bd3df8c 

**Autores:** Alicia María Rodríguez Trujillo y Kilian Santana Delgado
