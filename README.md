# Proyecto 1 — Clasificación de Contaminaciones en Línea de Producción

Entrenamiento y comparación de cuatro algoritmos de clasificación clásica sobre imágenes binarizadas (128×128 px) para detectar **granos de arroz (contaminaciones)** en una línea de producción.

| Algoritmo | Idea base |
|---|---|
| Árbol de Decisión | Divide los datos con preguntas binarias sobre píxeles |
| Naive Bayes | Probabilidad de clase dado cada píxel, asumiendo independencia |
| KNN | Clasifica según los K vecinos más cercanos en el espacio de píxeles |
| SVM | Encuentra el hiperplano que mejor separa las dos clases |

---

## Estructura del repositorio

```
├── clasificacion_proyecto1_v2.ipynb   # Notebook principal
├── README.md
└── modelos/                           # Carpeta donde se exportan los .joblib
```

---

## Requisitos

El notebook está diseñado para correr en **Google Colab** (sin instalación adicional). Si querés correrlo localmente:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

> Python 3.8+ recomendado.

---

## Cómo correr el entrenamiento

### En Google Colab (recomendado)

1. Abrí el notebook `clasificacion_proyecto1_v2.ipynb` en [Google Colab](https://colab.research.google.com/).
2. En la **sección 1 — Configuración**, editá tus datos:
   ```python
   CARNE    = "C00000"           # Tu carné
   NOMBRE   = "NombreEstudiante"
   APELLIDO = "ApellidoEstudiante"
   ```
3. Ejecutá **Entorno de ejecución → Ejecutar todo**.
4. Cuando la celda de carga lo solicite, subí tus archivos CSV con el formato requerido (ver abajo).
5. El notebook entrena los 4 modelos automáticamente con **GridSearchCV** (5 pliegues) y luego realiza una **evaluación repetida de 20 iteraciones** con particiones estratificadas.

### Localmente

```bash
jupyter notebook clasificacion_proyecto1_v2.ipynb
```

Reemplazá la celda de `files.upload()` (sección 1) con la ruta local a tus CSVs:

```python
ARCHIVOS_CSV = ["ruta/a/tu_dataset.csv"]
```

---

## Formato del dataset (CSV)

Cada archivo CSV debe tener:

- **16 384 columnas de píxeles** nombradas `pixel_1` … `pixel_16384` (imagen 128×128 aplanada)
- **1 columna de etiqueta** llamada `etiqueta` con valores `0` (sin arroz) o `1` (con arroz)

```
pixel_1, pixel_2, ..., pixel_16384, etiqueta
0, 1, ..., 0, 1
1, 0, ..., 1, 0
```

Podés subir varios CSVs a la vez; el notebook los combina automáticamente en un único dataset.

---

## Qué produce el entrenamiento

| Salida | Descripción |
|---|---|
| `distribucion_clases.png` | Distribución de clases (barra + torta) |
| `muestra_imagenes.png` | Muestra visual de imágenes del dataset |
| `comparacion_modelos.png` | Exactitud y F1 medios con barras de error |
| `boxplot_exactitudes.png` | Distribución de exactitudes en 20 repeticiones |
| `matrices_confusion.png` | Matrices de confusión de los 4 modelos |
| `arbol_decision.png` | Visualización del árbol de decisión |
| `CARNE_Nombre_Apellido.joblib` | Modelo ganador exportado |

La **Tabla Unificada de Métricas** (sección 9) consolida en un solo lugar: Accuracy, Precisión, Recall, F1, TP, TN, FP, FN, Acc CV y tiempo de entrenamiento para cada modelo.

---

## Cómo correr inferencia

Una vez exportado el modelo `.joblib`, podés usarlo para predecir nuevas imágenes:

```python
import joblib
import numpy as np

# Cargar el modelo exportado
modelo = joblib.load("C00000_Nombre_Apellido.joblib")

# nueva_imagen: array de shape (1, 16384) con valores 0/1
# Puede venir de aplanar una imagen 128×128 binarizada
nueva_imagen = np.array([...]).reshape(1, -1)

prediccion = modelo.predict(nueva_imagen)
print("Contaminación detectada" if prediccion[0] == 1 else "Sin contaminación")
```

Para predecir sobre múltiples imágenes a la vez:

```python
# X_nuevas: array de shape (N, 16384)
predicciones = modelo.predict(X_nuevas)
print(predicciones)  # array de 0s y 1s
```

---

## Parámetros configurables

Todos en la **sección 1** del notebook:

| Parámetro | Default | Descripción |
|---|---|---|
| `TEST_SIZE` | `0.2` | Fracción del dataset para prueba |
| `N_REPETICIONES` | `20` | Iteraciones de evaluación robusta |
| `CV_FOLDS` | `5` | Pliegues para GridSearchCV |
| `RANDOM_STATE` | `42` | Semilla de aleatoriedad |
| `COLUMNA_ETIQUETA` | `"etiqueta"` | Nombre de la columna objetivo en el CSV |
