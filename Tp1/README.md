## Ejecución

Es necesario tener instalado Python 3.8 o superior y las librerías listadas en `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Luego:

```bash
python tp1_1.py
python tp1_2.py
```

## Respuestas:

En cuanto a si conviene usar histogramas como atributos para clasificar imágenes, sí es posible, pero hay que considerar que un histograma descarta la información espacial. Esto implica que dos imágenes pueden compartir exactamente el mismo histograma y, aun así, tener contenidos totalmente distintos, como muestra la Figura 1.

![Comparación de histogramas](results/histogramas_tp1.png)

Por ese motivo, no lo sugeriría para tareas de clasificación. Hay enfoques que preservan mejor la estructura espacial, como la convolución con filtros, capaces de capturar rasgos locales y patrones, algo fundamental para este tipo de problemas.

## Algoritmo White Patch - Implementaciones

### Algoritmo Simple

Se desarrolló una versión básica que ofreció buenos resultados en las imágenes test_*.jpg, como puede verse en la Figura 2.

![Comparación de whitepathching](results/test_blue_comparison.png)
![Comparación de whitepathching](results/test_red_comparison.png)
![Comparación de whitepathching](results/test_green_comparison.png)

### Manejo de Casos Borde

La implementación inicial no contemplaba situaciones límite, por ejemplo, cuando hay píxeles con valores 0 o 255. Por eso se creó una segunda versión que sí trata estos casos. Estas dos primeras iteraciones mostraron buenos resultados en las imágenes wp_*.jpg, como se observa en la Figura 3.

![Comparación de whitepathching](results/wp_green_comparison.png)
![Comparación de whitepathching](results/wp_red_comparison.png)

### Algoritmo Inteligente

Por último, con ayuda de un agente en Copilot, se implementó una variante que primero clasifica la imagen a partir de la media y la desviación estándar del brillo y la saturación, para luego aplicar el método más adecuado según esa clasificación. A simple vista los resultados son buenos, aunque no afirmaría que superen a las versiones anteriores, como muestra la Figura 4.

![Comparación de whitepathching](results/wp_green2_comparison.png)
![Comparación de whitepathching](results/wp_red2_comparison.png)
![Comparación de whitepathching](results/wp_blue_comparison.png)
