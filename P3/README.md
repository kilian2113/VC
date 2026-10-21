# Práctica 3
## **TAREA 1**
### **Enunciado:** Los ejemplos ilustrativos anteriores permiten saber el número de monedas presentes en la imagen ideal proporcionada. ¿Cómo determinar la cantidad de dinero presente en ella? A partir únicamente de las dimensiones, una sugerencia impicaría en primer término identificar de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor fijo en la imagen (por ejemplo de 1€) para permitir establecer una relación entre píxeles y milímetros. Tras obtener esa información y las dimensiones en milímetros de las distintas monedas, es factible calcular la cantidad de dinero en la imagen ideal.

### Una vez resuelto el reto con la imagen ideal proporcionada, diseña un demostrador en vivo que a partir de la entrada de vídeo de la cámara, determine el dinero presente en la escena. Si bien no hay restricciones sobre utilizar medidas geométricas o de color, o cualquier clasificador, no deben hacer uso de modelos de aprendizaje profundo. ¿Funciona correctamente? ¿Se observan problemas? Posibles extras serían la presencia de objetos que no son monedas y/o haya solape entre las monedas.

### IMPORTANTE: No se admite el uso de modelos de aprendizaje profundo

### Nota: Para establecer la correspondencia entre píxeles y milímetros, comentar que la moneda de un euro tiene un diámetro de 23.25 mm. la de 50 céntimos de 24.35, la de 20 céntimos de 22.25, etc.




## **TAREA 2**
### **Enunciado:** La tarea consiste en extraer características (geométricas y/o visuales) de las tres imágenes completas de partida, y *aprender* patrones que permitan identificar las partículas en nuevas imágenes. Para ello se proporciona como imagen de test *MPs_test.jpg* y sus correpondientes anotaciones *MPs_test_bbs.csv* con la que deben obtener las métricas para su propuesta de clasificación de microplásticos, además de la matriz de confusión. La matriz de confusión permitirá mostrar para cada clase el número de muestras que se clasifican correctamente de dicha clase, y el número de muestras que se clasifican incorrectamente como perteneciente a una de las otras dos clases. De nuevo, no deben hacer uso de modelos de aprendizaje profundo ni para segmentar, ni para clasificar.

### En el trabajo [SMACC: A System for Microplastics Automatic Counting and Classification](https://doi.org/10.1109/ACCESS.2020.2970498), las características geométricas utilizadas fueron:

### - Área en píxeles
### - Perímetro en píxeles
### - Compacidad (relación entre el cuadrado del perímetro y el área de la partícula)
### - Relación del área de la partícula con la del contenedor
### - Relación del ancho y el alto del contenedor
### - Relación entre los ejes de la elipse ajustada
### - Definido el centroide, relación entre las distancias menor y mayor al contorno

### Si no se quedan satisfechos con la segmentación obtenida, es el mundo real, también en el README comento técnicas recientes de segmentación, que podrían despertar su curiosidad.

### IMPORTANTE: No se admite el uso de modelos de aprendizaje profundo

Para llevar a cabo esta tarea, hemos partido de las tres imágenes de entrenamiento, cada una con partículas de una sola clase (fragmentos, pellets y alquitrán), y de la imagen de prueba *MPs_test.jpg* con sus anotaciones en *MPs_test_bbs.csv*. La idea general es segmentar las partículas de cada imagen, describir cada una mediante un conjunto de características geométricas y de color, entrenar un clasificador con las partículas de las imágenes de entrenamiento y, finalmente, clasificar las partículas de la imagen de prueba y compararlas con las anotaciones.

Hemos observado dos problemas que condicionan el método:
- **Iluminación no uniforme**: las fotografías de entrenamiento tienen sombras y zonas más oscuras en los bordes, y la imagen de prueba, obtenida con escáner, tiene franjas horizontales y un borde inferior gris. Por ello, un umbral fijo o el método de Otsu aplicados directamente sobre la imagen en grises no separaban bien las partículas: con Otsu se perdía casi la mitad de las partículas claras.
- **Distinta resolución**: las imágenes de entrenamiento (2576x1932) tienen mucha más resolución que la de prueba (790x1186), por lo que una misma partícula ocupa muchos más píxeles en las primeras. Sin corregirlo, el área y el perímetro no son comparables entre imágenes.

Para mantener un código limpio, hemos separado el programa en tres bloques: variables, funciones y programa principal. Las funciones auxiliares son las siguientes:

**estimar_fondo()**: Estima cómo es el fondo de la imagen aplicando un filtro de mediana con una ventana mucho mayor que las partículas, de forma que estas desaparecen y solo queda la iluminación del fondo. El filtro se aplica dos veces, para eliminar los restos que puedan quedar donde hay partículas grandes o muy juntas. Para que sea más rápido, se trabaja sobre una versión reducida de la imagen, que después se vuelve a ampliar a su tamaño original.

**segmentar()**: Resta a la imagen el fondo estimado y mide, en el espacio de color *Lab*, cuánto se diferencia cada píxel del fondo. Además, elimina las franjas del escáner restando la mediana de cada fila y de cada columna, e ignora las variaciones pequeñas de brillo para que las sombras de las partículas no se segmenten. Se le da más peso al color que al brillo, ya que el fondo es gris y casi cualquier partícula se distingue por su color. Tras umbralizar, aplica una apertura para eliminar ruido y un cierre para cerrar huecos, y devuelve los contornos externos que superan un área mínima.

**caracteristicas()**: Calcula 13 características para cada partícula a partir de su contorno. Las siete primeras son las características geométricas del trabajo SMACC: área, perímetro, compacidad, relación entre el área de la partícula y la de su contenedor, relación entre el ancho y el alto del contenedor, relación entre los ejes de la elipse ajustada, y relación entre la distancia mínima y máxima del centroide al contorno. Sin embargo, al probarlas vimos que solo con ellas el acierto apenas llegaba al 55 %. Estas características distinguen bien los pellets, que son redondos y regulares, pero no separan bien los fragmentos del alquitrán, ya que ambos tienen formas irregulares. Por ello hemos añadido seis características más:

- **solidez**: relación entre el área de la partícula y la de su envolvente convexa, que es el contorno que tendría si "rellenáramos" sus entrantes. Los fragmentos son trozos rotos, con bordes quebrados y entrantes, por lo que su solidez es menor (0.91 de media) que la de los pellets y el alquitrán (0.97-0.98), que son más compactos.
- **dif_L**: diferencia media de brillo entre la partícula y el fondo. Es la característica que mejor identifica el alquitrán, ya que es mucho más oscuro que el resto (-77 de media frente a -31 en fragmentos y -46 en pellets).
- **dif_a** y **dif_b**: diferencia media de color entre la partícula y el fondo en los dos ejes de color del espacio *Lab* (*a*, de verde a rojo, y *b*, de azul a amarillo). Describen el tono de la partícula: los pellets suelen ser amarillentos (valor de *b* alto), los fragmentos aparecen en muchos colores (blancos, azules, verdes...) y el alquitrán no tiene tono (valores cercanos a 0).
- **croma**: intensidad del color, calculada a partir de *a* y *b*. Separa las partículas con color de las que son prácticamente grises o negras: el alquitrán tiene una croma de alrededor de 1, frente a valores de 9-11 en fragmentos y pellets.
- **std_L**: variación del brillo dentro de la partícula, que describe su textura. El alquitrán tiene una superficie rugosa con pequeños brillos, por lo que su variación es mucho mayor (alrededor de 40) que la de fragmentos y pellets (entre 14 y 17), que tienen un color más uniforme.

Las diferencias de brillo y color se miden respecto al fondo y no en valor absoluto, para que dependan lo menos posible de la cámara o del escáner con el que se ha tomado la imagen.

**procesar_imagen()**: Lee una imagen, la reescala si se le indica, la segmenta con **segmentar()** y calcula las características de cada partícula con **caracteristicas()**. Opcionalmente, descarta los contornos que tocan el borde de la imagen, ya que en las fotografías de entrenamiento la esquina inferior derecha es oscura y se detectaba como si fuera una partícula.

**iou()**: Calcula el índice de solapamiento IoU (intersección sobre unión) entre dos cajas, es decir, el área de su intersección dividida entre el área de su unión. Toma valores entre 0, si no se tocan, y 1, si son idénticas.

En el bloque principal del programa, comenzamos corrigiendo la diferencia de resolución entre imágenes. Para ello, calculamos el área mediana de todas las partículas detectadas en las imágenes de entrenamiento y en la imagen de prueba, y reducimos las imágenes de entrenamiento en la proporción necesaria para que ambas medianas coincidan. Es importante señalar que para este ajuste no utilizamos las anotaciones de la imagen de prueba, sino únicamente las partículas que detecta la segmentación. Se trata de una aproximación que funciona si las imágenes contienen partículas de tamaños parecidos; lo ideal sería conocer los milímetros por píxel de cada dispositivo, pero no disponíamos de esa información.

Una vez reescaladas, procesamos las tres imágenes de entrenamiento con **procesar_imagen()**. Como cada imagen contiene partículas de una sola clase, todas las partículas segmentadas en ella reciben directamente esa etiqueta, lo que nos permite construir el conjunto de entrenamiento. Para comprobar que la segmentación es correcta, mostramos los contornos obtenidos sobre cada imagen, y para analizar las características, una tabla con la media y la desviación típica de cada una por clase.

<img width="1790" height="477" alt="image" src="https://github.com/user-attachments/assets/7777c4c2-4594-49da-aa72-dc9be349e009" />
<img width="272" height="553" alt="image" src="https://github.com/user-attachments/assets/1cb1fee7-5f03-4715-ba34-c540de9b4dd9" />

A continuación, entrenamos el clasificador. Hemos utilizado el clasificador de los k vecinos más próximos (k-NN) de la biblioteca *scikit-learn*, con k=5: cada partícula nueva recibe la clase mayoritaria de las cinco partículas de entrenamiento más parecidas. A diferencia de otros métodos, como las redes neuronales, el k-NN no ajusta ningún parámetro durante el entrenamiento, sino que almacena las partículas de entrenamiento y, al clasificar, calcula la distancia a todas ellas. Por eso funciona bien con pocos datos, como en nuestro caso (unas 180 partículas). Además, no supone ninguna forma concreta en la frontera entre clases, es fácil de interpretar, ya que cada decisión se explica por sus vecinos, y solo requiere elegir el número de vecinos. Hemos tomado k=5 como término medio: con menos, el resultado depende demasiado de una sola partícula, que puede estar mal segmentada, y con más se mezclan partículas de otras clases. Como las características tienen escalas muy distintas (el área se mide en cientos de píxeles y las relaciones entre 0 y 1), las normalizamos previamente con **StandardScaler()** para que todas tengan el mismo peso al calcular la distancia. También probamos una máquina de vectores de soporte y un bosque aleatorio, que obtuvieron un acierto algo menor.

Después, pasamos a la imagen de prueba. Leemos las anotaciones del archivo CSV, segmentamos la imagen y emparejamos cada partícula anotada con la partícula detectada que más se solape con ella, utilizando **iou()**. Cada detección solo puede asociarse a una anotación, y el emparejamiento solo se acepta si el solapamiento es de al menos 0.3. Hemos elegido un umbral bajo porque las partículas miden pocos píxeles y las cajas anotadas son algo holgadas, de forma que con umbrales más altos quedaban sin emparejar partículas que estaban bien detectadas. Con este umbral se emparejan 110 de las 120 partículas anotadas.

Clasificamos las partículas emparejadas y comparamos la clase predicha con la anotada. Calculamos la exactitud, la precisión, la exhaustividad y el valor F1 con las funciones accuracy_score(), precision_score(), recall_score() y f1_score(), y la matriz de confusión con confusion_matrix(). Obtenemos una exactitud del 81.8 %, una precisión del 83.3 %, una exhaustividad del 81.8 % y un valor F1 del 81.1 %.

Además, dibujamos sobre la imagen de prueba el contorno de cada partícula con el color de la clase predicha y, en las mal clasificadas, escribimos encima su clase real; las partículas anotadas que no se detectaron se marcan con una caja gris.

<img width="662" height="562" alt="image" src="https://github.com/user-attachments/assets/1349ccfc-e57f-44a5-b4a7-37b6a5fe323a" />
<img width="635" height="965" alt="image" src="https://github.com/user-attachments/assets/aecc7ab7-828d-4ccf-ab21-76ed2539f12c" />


En cuanto a los resultados, los errores más frecuentes son pellets clasificados como fragmentos y alquitrán clasificado como fragmento, debidos principalmente a que las imágenes de entrenamiento y de prueba se han tomado con dispositivos e iluminación muy diferentes, lo que hace que el color de una misma clase no sea idéntico en ambas. Además, las 10 partículas no detectadas corresponden sobre todo a partículas que se tocan entre sí, y que la segmentación une en una sola región, y a fragmentos blancos casi transparentes, que apenas se diferencian del fondo.


#### *Bibliografía*
- https://www.bcl.lu/en/Banknotes-and-Coins/billets_pieces/car_pieces/car_tec/index.html
- https://www.youtube.com/watch?v=FtKNeAbzl1o
- https://www.youtube.com/watch?v=H-GmodwGBDw
- https://doi.org/10.1109/ACCESS.2020.2970498
- https://claude.ai/share/06a9976a-8281-48ca-aeed-025f1e5250e5 

**Autores**: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)
