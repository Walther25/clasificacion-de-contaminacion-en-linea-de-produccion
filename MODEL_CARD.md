# Model Card — Clasificador de Contaminaciones en Línea de Producción

**Modelo:** SVM con kernel RBF  
**Versión:** 1.0  
**Archivo:** `models/C31037_Walther_Barrantes.joblib`  
**Algoritmo base:** `sklearn.svm.SVC` — reentrenado con el 100% de los datos tras selección.

---

## Uso previsto

**Para qué sirve:**
Clasificar imágenes binarizadas 128×128 en dos categorías: presencia de granos de arroz (positivo) o ausencia (negativo), en el contexto de un ejercicio académico de detección de contaminaciones.

**Fuera de alcance:**
- Detección en tiempo real o sobre video.
- Contaminantes distintos a granos de arroz.
- Imágenes a color, con resolución distinta a 128×128, o no binarizadas.
- Producción industrial real; el modelo fue entrenado con datos de laboratorio controlados.

---

## Datos

El dataset consiste en 120 imágenes capturadas con un iPhone en interiores, convertidas a PNG 128×128 en escala de grises y binarizadas con umbral fijo (> 127 → 1). Cada imagen se representa como un vector de 16 384 valores binarios. Las condiciones de captura (distancia, iluminación, ángulo) no fueron controladas y varían entre imágenes.

Ver [DATASET.md](DATASET.md) para el detalle completo del preprocesamiento.

---

## Etiquetado

No hubo herramienta de anotación externa. La etiqueta se deduce del nombre del archivo asignado durante la captura (`1.N.png` = negativo, `2.N.png` = positivo). Esto elimina errores de anotación manual pero asume que la convención de nombres fue respetada consistentemente. No se realizó revisión posterior de calidad de etiquetas.

---

## Métricas

Evaluación sobre el mismo dataset de 120 imágenes con dos esquemas:

**Partición única (80/20 estratificado, `RANDOM_STATE=42`):**

| Modelo | Acc Prueba | Precisión | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **12** | **12** | **0** | **0** |
| Árbol de Decisión | 0.6250 | 0.5882 | 0.8333 | 0.6897 | 10 | 5 | 7 | 2 |
| KNN | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 12 | 0 | 12 | 0 |
| Naive Bayes | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 | 0 | 12 |

**Evaluación robusta (20 iteraciones con `StratifiedShuffleSplit`, 80/20):**

| Modelo | Acc Media | ± Std | F1 Media |
|---|---|---|---|
| **SVM** | **0.9771** | **±0.0384** | **0.9789** |
| Árbol de Decisión | 0.6875 | ±0.0639 | 0.7440 |
| KNN | 0.5042 | ±0.0125 | 0.6686 |
| Naive Bayes | 0.5188 | ±0.0246 | 0.0681 |

SVM fue seleccionado como modelo ganador por mayor exactitud media. Hiperparámetros óptimos: `C=1`, `kernel='rbf'`, encontrados con `GridSearchCV` (5 pliegues, `scoring='accuracy'`).

---

## Notas éticas y de sesgo

- **Sesgo de iluminación:** todas las capturas son en interiores con luz artificial similar. El modelo puede fallar bajo luz natural, sombras fuertes o cambios bruscos de exposición.
- **Sesgo de fondo:** el fondo es siempre una superficie clara de escritorio. Fondos con texturas, patrones o colores oscuros no están representados.
- **Sesgo de dispositivo:** todas las fotos provienen de un único iPhone. La respuesta de sensor de otras cámaras puede alterar los valores de píxel antes de la binarización.
- **Clase negativa artificial:** los clips no representan el objeto real que debería estar ausente en una línea de producción. El modelo aprende a separar arroz de clips, no arroz de ausencia de contaminación en general.

---

## Limitaciones técnicas

- **Objetos pequeños o parciales:** la binarización a 128×128 comprime la imagen. Granos de arroz en bordes o muy juntos pueden quedar fusionados o perderse.
- **Oclusión:** el modelo no fue entrenado con imágenes donde el arroz esté parcialmente cubierto.
- **Blur:** imágenes desenfocadas pueden binarizarse incorrectamente, produciendo vectores no representativos.
- **Umbral fijo:** el umbral de binarización (127) no se adapta a la exposición de cada imagen. Fotos sobreexpuestas o subexpuestas producen vectores distorsionados.
- **No hay separación de test independiente:** el conjunto de prueba forma parte del mismo dataset de 120 imágenes recolectadas bajo condiciones similares. Las métricas pueden ser optimistas.

---

## Reproducibilidad

**Entrenamiento:**

```bash
# Desde la raíz del repositorio
python src/script/clasificacion.py --csv data/dataset.csv
```

El script aplica `GridSearchCV` con los mismos hiperparámetros, selecciona el mejor modelo y lo exporta a `models/`.

**Semillas fijadas:** `RANDOM_STATE=42`, `np.random.seed(42)`.

**Hardware usado:** Google Colab (CPU, sin GPU). Tiempo total de búsqueda de hiperparámetros: ~18.7 s (todos los modelos).

**Dependencias:**

```
scikit-learn
pandas
numpy
matplotlib
joblib
opencv-python
pillow
pillow-heif
```

Ver [README.md](README.md) para los comandos de instalación y el flujo completo.
