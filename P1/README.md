# Práctica 1

## **TAREA 1**
### **Enunciado:** Sin herramientas de IA, crea una imagen, p.e. de 800x800 píxeles, con la textura del tablero de ajedrez. Una vez resuelto de forma manual, resuelve la misma tarea usando un asistente de IA de tu elección (Claude, ChatGPT, Copilot, etc.). Compara ambas versiones en el informe de la práctica

Para esta tarea hemos creado una imagen de resolución 800 × 800 píxeles. Para generar el tablero recorremos la imagen mediante dos bucles 
anidados, utilizando saltos de 100 píxeles para evitar realizar iteraciones innecesarias.

Dentro de los bucles utilizamos condiciones para determinar qué cuadrados deben ser blancos y cuáles negros. De esta forma, 
dependiendo de si la fila y la columna son pares o impares, se establece el color correspondiente. Cuando se cumple la condición, se modifica el bloque de 100 × 100 píxeles 
correspondiente, estableciendo sus valores a 255 (blanco).

<img width="426" height="418" alt="image" src="https://github.com/user-attachments/assets/9794451e-82be-4a80-ae96-3f2a2303bf9c" />


## **TAREA 2**
### **Enunciado:** Crear una imagen estilo Mondrian (un ejemplo https://www3.gobiernodecanarias.org/medusa/ecoescuela/sa/2017/04/17/descubriendo-a-mondrian/) con las funciones de dibujo de OpenCV. No hagas uso de herramientas de IA, parte del ejemplo anterior.
Para crear la imagen con estilo Mondrian comenzamos creando una imagen negra de 3 canales, que posteriormente transformamos en blanca estableciendo 
el valor de los tres canales a 255.

Sobre este fondo blanco comenzamos a realizar rectángulos de diferentes tamaños y colores utilizando la función *rectangle()* de OpenCV. También utilizamos *line()* 
para trazar líneas de diferentes grosores y separar las distintas zonas de la composición.

<img width="364" height="417" alt="image" src="https://github.com/user-attachments/assets/40f89871-9516-4012-bbc5-1c3403f2d12d" />

## **TAREA 3**
### **Enunciado:** Pintar círculos en las posiciones del píxel más claro y oscuro de cada fotograma captado por la cámara. ¿Funciona de forma fluida o a saltos? En el segundo caso, ¿podrías acelerarlo?Si haces uso de herramientas de IA, incluye la conversación.

## **TAREA 4**
### **Enunciado:** Llevar a cabo una propuesta propia de pop art. Incluye fuentes consultadas. Si haces uso de herramientas de IA, incluye la conversación.

#### *Bilbliografía*
- https://docs.opencv.org/3.4.20/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707
- https://docs.opencv.org/3.4.20/d4/dc6/tutorial_py_template_matching.html

**Autores:** Alicia María Rodríguez Trujillo y Kilian Santana
