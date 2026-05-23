# Proyecto 1 — Clasificación de Contaminaciones en Línea de Producción

Entrena y compara cuatro algoritmos de clasificación clásica sobre imágenes binarizadas 128×128 px para detectar **granos de arroz (contaminaciones)** en una línea de producción.

| Algoritmo | Idea base |
|---|---|
| Árbol de Decisión | Divide los datos con preguntas binarias sobre píxeles |
| Naive Bayes | Probabilidad de clase dado cada píxel, asumiendo independencia |
| KNN | Clasifica según los K vecinos más cercanos en el espacio de píxeles |
| SVM | Encuentra el hiperplano que mejor separa las dos clases |

---

## Estructura del repositorio

```
├── data/
│   ├── raw/                          # Fotos originales (.HEIC)
│   ├── procesed/                     # Imágenes convertidas a PNG 128x128 B&N
│   └── dataset.csv                   # Dataset combinado con encabezado pixel_1...pixel_16384, etiqueta
├── models/
│   └── C31037_Walther_Barrantes.joblib      # Modelo SVM exportado
├── reports/
│   ├── graficos/                     # Gráficas generadas por el entrenamiento
│   ├── informe_final_Walther_Barrantes.pdf  # Informe final del proyecto
│   └── resultados/
│       └── resultados.txt            # Salida completa del entrenamiento
├── src/
│   ├── notebook/                     # Versión Google Colab
│   │   ├── convertir_imagenes.ipynb
│   │   ├── imagenes_a_csv.ipynb
│   │   └── clasificacion.ipynb
│   └── script/                       # Versión ejecución local
│       ├── convertir_imagenes.py
│       ├── imagenes_a_csv.py
│       └── clasificacion.py
├── DATASET.md
└── README.md
```

---

## Requisitos

```bash
pip install scikit-learn pandas numpy matplotlib joblib pillow pillow-heif opencv-python
```

> Python 3.8+ recomendado.

---

## Flujo completo — de imagen a modelo

```
Fotos del celular  (data/raw/)
        │
        ▼
convertir_imagenes   →   PNG 128x128 B&N  (data/procesed/)
        │
        ▼
imagenes_a_csv       →   dataset.csv  (data/)
        │
        ▼
clasificacion        →   modelo .joblib  (models/)  +  gráficas  (reports/graficos/)
```

---

## Paso 1 — Convertir imágenes

### Script local

1. Colocá tus fotos en `data/raw/`.
2. Desde la raíz del repositorio ejecutá:

```bash
python src/script/convertir_imagenes.py
```

Las imágenes convertidas quedan en `data/procesed/` y se empaquetan en `converted_images.zip`.

### Notebook Colab

Abrí `src/notebook/convertir_imagenes.ipynb` en [Google Colab](https://colab.research.google.com/), ejecutá la celda y subí tus fotos cuando lo solicite.

**Formatos aceptados:** `.heic`, `.heif`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

---

## Paso 2 — Generar el CSV

### Script local

Colocá tus imágenes PNG binarizadas respetando la convención de nombres:

| Carpeta | Nombre del archivo | Etiqueta |
|---|---|---|
| `imagenes_entrenamiento/` | `1.1.png`, `1.2.png` ... | 0 — negativo (sin arroz) |
| `imagenes_entrenamiento/` | `2.1.png`, `2.2.png` ... | 1 — positivo (con arroz) |
| `imagenes_prueba/` | `negativa_1.png` ... | 0 — negativo |
| `imagenes_prueba/` | `positiva_1.png` ... | 1 — positivo |

Luego ejecutá:

```bash
python src/script/imagenes_a_csv.py
```

Genera `dataset_entrenamiento.csv` y `dataset_prueba.csv`.

### Notebook Colab

Abrí `src/notebook/imagenes_a_csv.ipynb` en Colab y subí tus imágenes cuando lo solicite.

**Formato del CSV resultante:**

```
pixel_1, pixel_2, ..., pixel_16384, etiqueta
0, 1, ..., 0, 1
1, 0, ..., 1, 0
```

---

## Paso 3 — Entrenamiento

### Script local

Editá tus datos al inicio de `src/script/clasificacion.py`:

```python
CARNE    = "X00000"
NOMBRE   = "TuNombre"
APELLIDO = "TuApellido"
```

Luego ejecutá:

```bash
# Con un CSV
python src/script/clasificacion.py --csv data/dataset.csv

# Con varios CSVs (se combinan automáticamente)
python src/script/clasificacion.py --csv data/dataset1.csv data/dataset2.csv
```

### Notebook Colab

1. Abrí `src/notebook/clasificacion.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Editá `CARNE`, `NOMBRE` y `APELLIDO` en la sección 1.
3. Ejecutá **Entorno de ejecución → Ejecutar todo**.
4. Subí tu CSV cuando la celda lo solicite.

### Parámetros configurables

| Parámetro | Default | Descripción |
|---|---|---|
| `TEST_SIZE` | `0.2` | Fracción del dataset para prueba |
| `N_REPETICIONES` | `20` | Iteraciones de evaluación robusta |
| `CV_FOLDS` | `5` | Pliegues para GridSearchCV |
| `RANDOM_STATE` | `42` | Semilla de aleatoriedad |
| `COLUMNA_ETIQUETA` | `"etiqueta"` | Nombre de la columna objetivo en el CSV |

### Salidas del entrenamiento

| Archivo | Descripción |
|---|---|
| `reports/graficos/distribucion_clases.png` | Distribución de clases (barra + torta) |
| `reports/graficos/muestra_dataset.png` | Muestra visual de imágenes reconstruidas |
| `reports/graficos/comparacion_modelos.png` | Exactitud y F1 medios con barras de error |
| `reports/graficos/boxplot_exactitudes.png` | Distribución de exactitudes en 20 repeticiones |
| `reports/graficos/matrices_confusion.png` | Matrices de confusión de los 4 modelos |
| `reports/graficos/arbol_decision.png` | Visualización del árbol de decisión |
| `reports/informe_final.pdf` | Informe final del proyecto |
| `models/CARNE_Nombre_Apellido.joblib` | Modelo ganador exportado |

La **Tabla Unificada de Métricas** impresa al final consolida: Accuracy, Precisión, Recall, F1, TP, TN, FP, FN, Acc CV y tiempo para cada modelo.

---

## Inferencia — usar el modelo exportado

Una vez generado el `.joblib`, podés usarlo para predecir nuevas imágenes sin reentrenar:

```python
import joblib
import numpy as np
import cv2

# 1. Cargar el modelo
modelo = joblib.load("models/C31037_Walther_Barrantes.joblib")

# 2. Preparar una imagen nueva (PNG 128x128 binarizada)
imagen = cv2.imread("mi_imagen.png", cv2.IMREAD_GRAYSCALE)
imagen = cv2.resize(imagen, (128, 128), interpolation=cv2.INTER_NEAREST)
vector = (imagen > 127).astype(int).flatten().reshape(1, -1)  # shape (1, 16384)

# 3. Predecir
prediccion = modelo.predict(vector)
print("Contaminación detectada" if prediccion[0] == 1 else "Sin contaminación")
```

Para predecir sobre múltiples imágenes a la vez:

```python
# X_nuevas: array de shape (N, 16384), una fila por imagen
predicciones = modelo.predict(X_nuevas)
print(predicciones)  # array de 0s y 1s
```

---

## Resultados

### Dataset
![Distribución de clases](reports/graficos/distribucion_clases.png)

120 imágenes con balance perfecto: 60 positivas (con arroz) y 60 negativas (sin arroz).

![Muestra del dataset](reports/graficos/muestra_dataset.png)

Fila superior: imágenes positivas (granos de arroz dispersos). Fila inferior: imágenes negativas (clips, sin contaminación).

### Comparación de modelos
![Comparación de modelos](reports/graficos/comparacion_modelos.png)

SVM domina con Acc media = 0.977 y F1 = 0.979. Árbol de Decisión queda segundo (Acc = 0.688). Naive Bayes y KNN rondan el nivel del azar.

![Distribución de exactitudes](reports/graficos/boxplot_exactitudes.png)

SVM es el más estable: caja muy angosta y sin dispersión entre las 20 iteraciones. El Árbol de Decisión muestra mayor varianza.

### Análisis detallado
![Matrices de confusión](reports/graficos/matrices_confusion.png)

SVM logra clasificación perfecta en prueba (Acc = 1.0, 0 FP y 0 FN). Naive Bayes y KNN predicen siempre la misma clase.

![Árbol de decisión](reports/graficos/arbol_decision.png)

El árbol usa píxeles específicos como puntos de corte binarios (≤ 0.5). La raíz parte por el píxel 14007, la región más discriminativa entre clases.

---

## Dataset

Ver [DATASET.md](DATASET.md) para la descripción completa del proceso de recolección y las limitaciones del dataset.
