# Práctica 3
## **TAREA 1**
### **Enunciado:** Los ejemplos ilustrativos anteriores permiten saber el número de monedas presentes en la imagen ideal proporcionada. ¿Cómo determinar la cantidad de dinero presente en ella? A partir únicamente de las dimensiones, una sugerencia impicaría en primer término identificar de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor fijo en la imagen (por ejemplo de 1€) para permitir establecer una relación entre píxeles y milímetros. Tras obtener esa información y las dimensiones en milímetros de las distintas monedas, es factible calcular la cantidad de dinero en la imagen ideal.

### Una vez resuelto el reto con la imagen ideal proporcionada, diseña un demostrador en vivo que a partir de la entrada de vídeo de la cámara, determine el dinero presente en la escena. Si bien no hay restricciones sobre utilizar medidas geométricas o de color, o cualquier clasificador, no deben hacer uso de modelos de aprendizaje profundo. ¿Funciona correctamente? ¿Se observan problemas? Posibles extras serían la presencia de objetos que no son monedas y/o haya solape entre las monedas.

### IMPORTANTE: No se admite el uso de modelos de aprendizaje profundo

### Nota: Para establecer la correspondencia entre píxeles y milímetros, comentar que la moneda de un euro tiene un diámetro de 23.25 mm. la de 50 céntimos de 24.35, la de 20 céntimos de 22.25, etc.

Para llevar a cabo esta tarea, la hemos dividido en dos partes: primero calculamos el dinero presente en la imagen ideal *Monedas.jpg* y, después, desarrollamos un demostrador en vivo que lo calcula a partir de la imagen de la cámara. En ambos casos la idea general es la misma: detectar las monedas, medir su diámetro en píxeles, convertirlo a milímetros mediante una referencia de tamaño conocido y asignar a cada moneda el valor cuyo diámetro real sea más parecido.

Para ello, hemos definido un diccionario con el diámetro oficial en milímetros de cada moneda de euro:

| Moneda | 1 cént. | 2 cént. | 5 cént. | 10 cént. | 20 cént. | 50 cént. | 1 € | 2 € |
|---|---|---|---|---|---|---|---|---|
| Diámetro (mm) | 16.25 | 18.75 | 21.25 | 19.75 | 22.25 | 24.25 | 23.25 | 25.75 |

### Imagen ideal

En la imagen ideal hemos probado dos formas de detectar las monedas.

La primera utiliza la transformada de Hough para círculos con **HoughCircles()**. Antes, convertimos la imagen a escala de grises y aplicamos un filtro de mediana con **medianBlur()** para eliminar el ruido y los detalles finos del relieve de las monedas, que podrían generar círculos falsos. A **HoughCircles()** le indicamos una distancia mínima entre centros, para no detectar dos veces la misma moneda, y un radio mínimo y máximo acordes con el tamaño de las monedas en la imagen. Esta función devuelve directamente el centro y el radio de cada círculo, por lo que el diámetro en píxeles es el doble del radio.

<img width="252" height="504" alt="image" src="https://github.com/user-attachments/assets/db441349-5c67-4c50-b9b0-34551f166158" />

La segunda umbraliza la imagen en grises con el método de Otsu, obtiene los contornos externos con **findContours()** y descarta los de área muy pequeña, que no pueden ser monedas. Como el contorno no da directamente el diámetro, lo calculamos a partir del área, como el diámetro del círculo que tendría esa misma área.

<img width="252" height="504" alt="image" src="https://github.com/user-attachments/assets/cee7104a-9e1a-477b-81a8-9942a21dae17" />

Para convertir píxeles a milímetros, en lugar de seleccionar una moneda haciendo clic, tomamos como referencia la moneda de mayor diámetro detectada, que suponemos que es la de 2 €, ya que es la más grande de todas. Dividiendo su diámetro real (25.75 mm) entre su diámetro en píxeles obtenemos los milímetros que mide cada píxel. A continuación, para cada moneda convertimos su diámetro a milímetros y le asignamos el valor cuyo diámetro real está más próximo, sumando ese valor al total. Por último, dibujamos cada moneda sobre la imagen con su valor e imprimimos el número de monedas encontradas y el dinero total.

Este método funciona porque la imagen ideal tiene un fondo uniforme, las monedas no se tocan y hay una moneda de 2 €. Si no la hubiera, la calibración tomaría como referencia otra moneda y todos los valores saldrían desplazados.

### Demostrador en vivo

En el vídeo de la cámara no podemos suponer que siempre haya una moneda de 2 €, y además la distancia y la inclinación de la cámara cambian continuamente. Por ello, hemos usado como referencia un folio A4, cuyas medidas conocemos (210x297 mm), y colocamos las monedas sobre él. El folio nos permite, por un lado, saber cuántos píxeles mide un milímetro y, por otro, corregir la perspectiva, de forma que las monedas se vean como círculos aunque la cámara esté inclinada. Además, para distinguir monedas de diámetro parecido, añadimos su color.

Las funciones auxiliares son las siguientes:

**order_points()**: Recibe las cuatro esquinas del folio y las ordena siempre de la misma forma: superior izquierda, superior derecha, inferior derecha e inferior izquierda. Para ello, usa la suma y la diferencia de sus coordenadas: la esquina superior izquierda es la de menor suma y la inferior derecha la de mayor suma, mientras que la superior derecha es la de menor diferencia y la inferior izquierda la de mayor.

**detec_paper()**: Localiza el folio en el fotograma. Convierte la imagen a escala de grises, la suaviza con un filtro gaussiano y la umbraliza con el método de Otsu, de forma que el folio, que es blanco, queda separado del fondo. Después limpia la máscara con una apertura y un cierre, busca los contornos externos y los recorre de mayor a menor área, descartando los demasiado pequeños o los que ocupan casi toda la imagen. Cada contorno se simplifica con **approxPolyDP()**, y el primero que queda reducido a un polígono convexo de cuatro lados se considera el folio. Devuelve sus cuatro esquinas ordenadas con **order_points()**.

**warp_paper()**: Corrige la perspectiva del folio para verlo como si la cámara estuviera justo encima. Calcula el ancho y el alto del folio en píxeles para saber si está en horizontal o en vertical, y crea una imagen nueva con las proporciones de un A4 y una densidad fija de 4 píxeles por milímetro. Con **getPerspectiveTransform()** obtiene la matriz que lleva las esquinas del folio a las esquinas de esa imagen y la aplica con **warpPerspective()**. Devuelve el folio rectificado y la matriz inversa, que sirve para devolver después los resultados al fotograma original.

**color_group()**: Clasifica una moneda según su color en tres grupos: cobre (1, 2 y 5 céntimos), oro (10, 20 y 50 céntimos) y bimetálica (1 y 2 €). Para ello, trabaja en el espacio de color *HSV*, que separa el tono (H) y la saturación (S) del brillo, y compara la zona central de la moneda con un anillo cercano al borde. Si el tono o la saturación del centro y del anillo son muy distintos, la moneda es bimetálica; si no, se decide entre cobre y oro según el tono del anillo. Usamos la mediana en lugar de la media para que los reflejos de la moneda no alteren el resultado.

En el bloque principal del programa, leemos continuamente los fotogramas de la cámara y buscamos el folio en cada uno con **detec_paper()**. Si lo encontramos, dibujamos su contorno y lo rectificamos con **warp_paper()**. Como el folio rectificado tiene siempre 4 píxeles por milímetro, la conversión de píxeles a milímetros es directa y no depende de la distancia a la cámara.

Sobre el folio rectificado, convertido a escala de grises y suavizado con un filtro de mediana, buscamos las monedas con **HoughCircles()**. Limitamos los radios a los que corresponden a diámetros entre 15 y 27 mm, que abarcan todas las monedas de euro, y descartamos los círculos que se salen de la imagen. Para cada moneda calculamos su diámetro en milímetros, obtenemos su grupo de color con **color_group()** y, dentro de ese grupo, le asignamos el valor cuyo diámetro real está más próximo. Combinar color y tamaño evita confusiones entre monedas de diámetro parecido, como la de 1 € (23.25 mm) y la de 20 céntimos (22.25 mm). Si la diferencia entre el diámetro medido y el de la moneda asignada supera 1 mm, consideramos que el objeto no es una moneda conocida y lo marcamos con un signo de interrogación, sin sumarlo al total.

Por último, usamos la matriz inversa para trasladar el contorno y el centro de cada moneda desde el folio rectificado al fotograma original, y dibujamos sobre él cada moneda con su valor, junto con el número de monedas detectadas y el dinero total. Además, mostramos el folio rectificado en una ventana aparte. El programa termina al pulsar la tecla ESC.



https://github.com/user-attachments/assets/910bee92-7810-4e79-9f56-0ddc42f3521b



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
- **dif_l**: diferencia media de brillo entre la partícula y el fondo. Es la característica que mejor identifica el alquitrán, ya que es mucho más oscuro que el resto (-77 de media frente a -31 en fragmentos y -46 en pellets).
- **dif_a** y **dif_b**: diferencia media de color entre la partícula y el fondo en los dos ejes de color del espacio *Lab* (*a*, de verde a rojo, y *b*, de azul a amarillo). Describen el tono de la partícula: los pellets suelen ser amarillentos (valor de *b* alto), los fragmentos aparecen en muchos colores (blancos, azules, verdes...) y el alquitrán no tiene tono (valores cercanos a 0).
- **croma**: intensidad del color, calculada a partir de *a* y *b*. Separa las partículas con color de las que son prácticamente grises o negras: el alquitrán tiene una croma de alrededor de 1, frente a valores de 9-11 en fragmentos y pellets.
- **dsv_l**: variación del brillo dentro de la partícula, que describe su textura. El alquitrán tiene una superficie rugosa con pequeños brillos, por lo que su variación es mucho mayor (alrededor de 40) que la de fragmentos y pellets (entre 14 y 17), que tienen un color más uniforme.

Las diferencias de brillo y color se miden respecto al fondo y no en valor absoluto, para que dependan lo menos posible de la cámara o del escáner con el que se ha tomado la imagen.

**procesar_imagen()**: Lee una imagen, la reescala si se le indica, la segmenta con **segmentar()** y calcula las características de cada partícula con **caracteristicas()**. Opcionalmente, descarta los contornos que tocan el borde de la imagen, ya que en las fotografías de entrenamiento la esquina inferior derecha es oscura y se detectaba como si fuera una partícula.

**iou()**: Calcula el índice de solapamiento IoU (intersección sobre unión) entre dos cajas, es decir, el área de su intersección dividida entre el área de su unión. Toma valores entre 0, si no se tocan, y 1, si son idénticas.

**mostrar_metricas()**: Calcula la exactitud, la precisión, la exhaustividad y el valor F1 con las funciones **accuracy_score()**, **precision_score()**, **recall_score()** y **f1_score()** de *scikit-learn*, las imprime y devuelve la exactitud.

**mostrar_matriz_confusion()**: Obtiene la matriz de confusión con **confusion_matrix()** y la dibuja. Muestra para cada clase cuántas partículas se clasifican correctamente y cuántas se confunden con cada una de las otras dos.

**mostrar_resultado()**: Dibuja sobre la imagen de prueba el contorno de cada partícula con el color de la clase predicha, escribe encima la clase real de las que están mal clasificadas y marca con una caja gris las partículas anotadas que no se detectaron.

Estas tres funciones se utilizan tanto con el k-NN como con el árbol de decisión del apartado extra, para poder comparar ambos de la misma forma.

En el bloque principal del programa, comenzamos corrigiendo la diferencia de resolución entre imágenes. Para ello, calculamos el área mediana de todas las partículas detectadas en las imágenes de entrenamiento y en la imagen de prueba, y reducimos las imágenes de entrenamiento en la proporción necesaria para que ambas medianas coincidan. Es importante señalar que para este ajuste no utilizamos las anotaciones de la imagen de prueba, sino únicamente las partículas que detecta la segmentación. Se trata de una aproximación que funciona si las imágenes contienen partículas de tamaños parecidos; lo ideal sería conocer los milímetros por píxel de cada dispositivo, pero no disponíamos de esa información.

Una vez reescaladas, procesamos las tres imágenes de entrenamiento con **procesar_imagen()**. Como cada imagen contiene partículas de una sola clase, todas las partículas segmentadas en ella reciben directamente esa etiqueta, lo que nos permite construir el conjunto de entrenamiento. Para comprobar que la segmentación es correcta, mostramos los contornos obtenidos sobre cada imagen, y para analizar las características, una tabla con la media y la desviación típica de cada una por clase.

<img width="1790" height="477" alt="image" src="https://github.com/user-attachments/assets/7777c4c2-4594-49da-aa72-dc9be349e009" />
<img width="272" height="553" alt="image" src="https://github.com/user-attachments/assets/1cb1fee7-5f03-4715-ba34-c540de9b4dd9" />

A continuación, entrenamos el clasificador. Hemos utilizado el clasificador de los k vecinos más próximos (k-NN) de la biblioteca *scikit-learn*, con k=5: cada partícula nueva recibe la clase mayoritaria de las cinco partículas de entrenamiento más parecidas. A diferencia de otros métodos, como las redes neuronales, el k-NN no ajusta ningún parámetro durante el entrenamiento, sino que almacena las partículas de entrenamiento y, al clasificar, calcula la distancia a todas ellas. Por eso funciona bien con pocos datos, como en nuestro caso (unas 180 partículas). Además, no supone ninguna forma concreta en la frontera entre clases, es fácil de interpretar, ya que cada decisión se explica por sus vecinos, y solo requiere elegir el número de vecinos. Hemos tomado k=5 como término medio: con menos, el resultado depende demasiado de una sola partícula, que puede estar mal segmentada, y con más se mezclan partículas de otras clases. Como las características tienen escalas muy distintas (el área se mide en cientos de píxeles y las relaciones entre 0 y 1), las normalizamos previamente con **StandardScaler()** para que todas tengan el mismo peso al calcular la distancia. También probamos una máquina de vectores de soporte y un bosque aleatorio, que obtuvieron un acierto algo menor.

Después, pasamos a la imagen de prueba. Leemos las anotaciones del archivo CSV, segmentamos la imagen y emparejamos cada partícula anotada con la partícula detectada que más se solape con ella, utilizando **iou()**. Cada detección solo puede asociarse a una anotación, y el emparejamiento solo se acepta si el solapamiento es de al menos 0.3. Hemos elegido un umbral bajo porque las partículas miden pocos píxeles y las cajas anotadas son algo holgadas, de forma que con umbrales más altos quedaban sin emparejar partículas que estaban bien detectadas. Con este umbral se emparejan 110 de las 120 partículas anotadas.

Por último, clasificamos las partículas emparejadas con el k-NN y evaluamos el resultado llamando a **mostrar_metricas()**, **mostrar_matriz_confusion()** y **mostrar_resultado()**. Obtenemos una exactitud del 81.8 %, una precisión del 83.3 %, una exhaustividad del 81.8 % y un valor F1 del 81.1 %.


<img width="662" height="562" alt="image" src="https://github.com/user-attachments/assets/1349ccfc-e57f-44a5-b4a7-37b6a5fe323a" />
<img width="635" height="965" alt="image" src="https://github.com/user-attachments/assets/aecc7ab7-828d-4ccf-ab21-76ed2539f12c" />


En cuanto a los resultados, los errores más frecuentes son pellets clasificados como fragmentos y alquitrán clasificado como fragmento, debidos principalmente a que las imágenes de entrenamiento y de prueba se han tomado con dispositivos e iluminación muy diferentes, lo que hace que el color de una misma clase no sea idéntico en ambas. Además, las 10 partículas no detectadas corresponden sobre todo a partículas que se tocan entre sí, y que la segmentación une en una sola región, y a fragmentos blancos casi transparentes, que apenas se diferencian del fondo.

### Extra: clasificación con un árbol de decisión

Como alternativa más sencilla al k-NN, hemos probado un árbol de decisión de la biblioteca *scikit-learn* con **DecisionTreeClassifier()**. Un árbol de decisión clasifica cada partícula mediante una serie de preguntas del tipo "¿esta característica es menor que este umbral?", que aprende a partir de las partículas de entrenamiento eligiendo en cada paso la pregunta que mejor separa las clases. Hemos limitado su profundidad a 2, de modo que cada partícula se clasifica respondiendo como mucho a dos preguntas. A diferencia del k-NN, no necesita normalizar las características, ya que cada pregunta compara una sola característica con un umbral. Para evaluarlo, llamamos a las mismas funciones que con el k-NN (**mostrar_metricas()**, **mostrar_matriz_confusion()** y **mostrar_resultado()**), de forma que los resultados de ambos clasificadores son  comparables.

La principal ventaja de este método es que se puede interpretar: con **plot_tree()** dibujamos las reglas aprendidas. La primera pregunta es sobre la textura (**dsv_l**), que separa el alquitrán, rugoso y con brillos, del resto. A continuación, entre las partículas de textura uniforme, la relación entre la distancia mínima y máxima del centroide al contorno separa los pellets, que son redondos, de los fragmentos. Entre las de textura rugosa, la relación entre el ancho y el alto del contenedor separa las pocas partículas muy alargadas, que clasifica como fragmentos, del alquitrán. Estas reglas coinciden con lo que observamos en la tabla de características.

Con solo estas dos preguntas obtenemos una exactitud del 76.4 %, una precisión del 77.3 %, una exhaustividad del 76.4 % y un valor F1 del 75.7 %, algo por debajo del k-NN (81.8 % de exactitud). Esto demuestra que la mayor parte de la información para distinguir las tres clases está en muy pocas características, aunque el k-NN, al usar todas a la vez, consigue un resultado mejor. Como en el k-NN, los errores más frecuentes son pellets y alquitrán clasificados como fragmentos.

<img width="662" height="562" alt="image" src="https://github.com/user-attachments/assets/c398628d-ab8c-40d9-9a08-0158c38328c7" />
<img width="1105" height="581" alt="image" src="https://github.com/user-attachments/assets/6ae08b38-1266-4904-b86b-534c4957a5d8" />
<img width="635" height="967" alt="image" src="https://github.com/user-attachments/assets/8d897e2a-de5c-46a9-83a1-cf46436957bc" />


#### *Bibliografía*
- https://www.bcl.lu/en/Banknotes-and-Coins/billets_pieces/car_pieces/car_tec/index.html
- https://www.youtube.com/watch?v=FtKNeAbzl1o
- https://www.youtube.com/watch?v=H-GmodwGBDw
- https://doi.org/10.1109/ACCESS.2020.2970498
- https://www.ibm.com/es-es/think/topics/knn 
- https://aprendeia.com/2019/10/25/arboles-de-decision-clasificacion-scikit-learn-machine-learning/
- https://claude.ai/share/06a9976a-8281-48ca-aeed-025f1e5250e5

**Autores**: 
[Alicia María Rodríguez Trujillo](https://github.com/AliciaM05) y 
[Kilian Santana Delgado](https://github.com/kilian2113)
