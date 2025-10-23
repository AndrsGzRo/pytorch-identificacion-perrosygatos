# pytorch-identificacion-perrosygatos

## Introducción 
Han pasado más de dos años desde que se descubrió la vacuna contra los zombis. Ahora, un nuevo peligro amenaza al mundo: algunas razas de perros son inmunes a la vacuna y pueden crear una nueva cepa del virus.

La empresa estadounidense Small Pet, al darse cuenta del problema, alertó a todos los países. En México, la empresa Ciencia para el Futuro se ha propuesto apoyar con la creación de una aplicación para identificar a dichos perros.

Tu misión: crear una App capaz de identificar rápidamente a los perros inmunes, antes de que la nueva cepa del virus mortal afecte a su primera víctima. ¿Podrás lograrlo antes de que el mundo vuelva a caer en manos de los zombis?

## Objetivos
- Diseñar modelos de **redes neuronales profundas (Deep Learning)**, enfocados en la clasificación de imágenes para resolver problemas con relevancia social.

- Crear modelos de **Deep Neural Networks (DNN)** utilizando **PyTorch** y **Python**, seleccionando la arquitectura adecuada y analizando la exactitud del modelo para cumplir con los requerimientos del proyecto.

- Generar valor en diversos sectores mediante la aplicación de técnicas de inteligencia artificial a problemas reales.

- Comparar el rendimiento de un perceptrón multicapa (MLP) con una red neuronal convolucional (CNN) en la clasificación de imágenes de perros y gatos.
  
- Guardar y reutilizar modelos entrenados en PyTorch. 

## Tecnologías y herramientas
- Python
- PyTorch
- Torchvision (para datasets y transformaciones de imágenes)
- Matplotlib / Seaborn (para visualización de resultados)
- Pathlib (para manejo de rutas de archivos)


## Modelo 1: MultiLayerPerceptron (MLP)
El perceptrón multicapa (MLP) fue el primer modelo implementado para clasificar imágenes entre perros y gatos, usando únicamente capas totalmente conectadas. 
El MLP no analiza patrones espaciales, sino que trata cada píxel como una entrada independiente, lo que limita su capacidad de generalización en imágenes, pero es útil como punto de partida.

### Arquitectura MLP
| Capa | Tipo | Dimensiones | Función |
|------|------|--------------|----------|
| `fc1` | Linear | 150,528 → 512 | Capa de entrada, reduce la dimensionalidad del vector de píxeles. |
| `fc2` | Linear | 512 → 128 | Extrae patrones combinando activaciones previas. |
| `fc3` | Linear | 128 → 64 | Capa intermedia para abstracciones de nivel medio. |
| `fc4` | Linear | 64 → 2 | Capa de salida: logits para “Cat” y “Dog”. |

- **Activación**: ReLU en capas ocultas.
- **Función de pérdidas:** CrossEntropyLoss
- **Optimizador**: Adam(lr = 0.001)

### 📊 Resultados 
| Métrica | Entrenamiento | Validación |
|----------|----------------|-------------|
| Exactitud | 95.33 % | 61.9 %

El modelo aprende de manera excelente el conjunto de entrenamiento, pero muestra **sobreajuste**, ya que no generaliza bien las imágenes del conjunto de validación.

### ⚠️ Limitaciones 
- No aprovecha la estructura espacial de las imágenes.
- Requiere demasiados parámetros.
- Tiende a sobreajustar rápidamente datasets pequeños.

Por estas razones el MLP fue reemplzado por una **Red Convolucional (CNN)**.

## Modelo  2 - Convolutional Neural Network (CNN)

El segundo modelo implementado fue una **Red Neuronal Convolucional (CNN)**, diseñada para mejorar la capacidad del sistema al capturar **patrones espaciales** dentro de las imágenes. A diferencia del MLP, la CNN analiza las imágenes de forma matricial, aprendiendo automáticamente **bordes, texturas y formas** relevantes para distinguir entre **gatos y perros**. 

###  Arquitectura CNN
| Capa | Tipo | Parámetros | Tamaño de salida | Descripción |
|------|------|-------------|------------------|--------------|
| `conv1` | Conv2d | 3 → 32, kernel=3 | 32×32×32 | Detecta bordes y colores básicos. |
| `pool1` | MaxPool2d | 2×2 | 32×32×32 → 32×32×16 | Reduce la resolución, conserva rasgos más relevantes. |
| `conv2` | Conv2d | 32 → 64, kernel=3 | 64×16×16 | Aprende texturas y formas simples. |
| `pool2` | MaxPool2d | 2×2 | 64×16×16 → 64×16×8 | Reduce el tamaño de representación. |
| `conv3` | Conv2d | 64 → 128, kernel=3 | 128×8×8 | Capta características complejas y de alto nivel. |
| `pool3` | MaxPool2d | 2×2 | 128×8×8 → 128×8×8 | Consolidación final de características visuales. |
| `dropout` | Dropout(0.5) | — | — | Evita sobreajuste desactivando neuronas al azar. |
| `fc1` | Linear | 128×8×8 → 256 | — | Combina características extraídas por las convoluciones. |
| `fc2` | Linear | 256 → 2 | — | Capa de salida: logits para “Cat” y “Dog”. |

### Detalles del entrenamiento
- **Tamaño de entrada:** 64x64x3
- **Función de pérdida:** CrossEntropyLoss.
- **Optimizador:** Adam(lr=0.0005)
- **Regularización:** Dropout(0.5) + BatchNormalization.
- **Epochs:** 25
- **Scheduler:** Reduce learning rate cada 7 epochs. 
###  📈 Resultados CNN
| Métrica | Entrenamiento | Validación |
|----------|----------------|-------------|
| Exactitud | **81.4 %** | **76.4 %** |
| Pérdida | 0.40 | 0.48 |

La CNN muestra una clara mejora en precisión y estabilidad respecto al MLP, reduciendo el sobreajuste y generalizando mejor en datos no vistos. 

### Ventajas del modelo
- Aprende **características visuales jerárquicas**.
- Requiere **menos parámetros** que el MLP para lograr mayor rendimiento.
- Es más **robusta a traslaciones y variaciones** en las imágenes.
- Puede escalarse fácilmente usando arquitecturas preentrenadas.

## ⚖️ Comparativa entre MLP y CNN

| Característica | MLP | CNN |
|----------------|-----|-----|
| **Tipo de entrada** | Imagen aplanada (1D) | Imagen 2D (mantiene estructura espacial) |
| **Capas principales** | Lineales (Fully Connected) | Convolucionales + Pooling |
| **Capacidad para detectar patrones visuales** | ❌ Muy limitada | ✅ Alta (bordes, texturas, formas) |
| **Número de parámetros** | Muy alto (millones) | Mucho menor |
| **Riesgo de sobreajuste** | Alto | Moderado |
| **Exactitud en validación** | ~60 % | **~76 %** |
| **Generalización** | Pobre | Buena |
| **Eficiencia computacional** | Baja | Alta |
| **Uso recomendado** | Pruebas iniciales o datasets tabulares | Clasificación de imágenes y visión artificial |

## Conclusión 
> La CNN logra un equilibrio óptimo entre precisión y capacidad de generalización.
> Representa una mejora significativa sobre el MLP al aprovechar la estructura espacial de las imágenes, reduciendo parámetros y sobreajuste.


