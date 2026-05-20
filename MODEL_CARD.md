# Model Card — Clasificador de Contaminaciones en Línea de Producción

**Modelo:** SVM con kernel RBF  
**Versión:** 1.0  
**Archivo:** `models/C31037_Walther_Barrantes.joblib`  
**Algoritmo base:** `sklearn.svm.SVC` — reentrenado con el 100% de los datos tras selección.

---

## Model Name + Version / Nombre y versión del modelo

| Campo | Valor |
|---|---|
| Nombre | Clasificador de Contaminaciones — SVM RBF |
| Versión | 1.0 |
| Archivo | `models/C31037_Walther_Barrantes.joblib` |
| Clase | `sklearn.svm.SVC(C=1, kernel='rbf', probability=True)` |
| Entrenado con | 120 imágenes (100% del dataset tras selección por GridSearchCV) |

---

## Intended Use / Uso previsto

**Para qué sirve:**
Clasificar imágenes binarizadas 128×128 en dos categorías — presencia de granos de arroz (positivo) o ausencia (negativo) — en el contexto de un ejercicio académico de detección de contaminaciones.

**Out-of-scope / Fuera de alcance:**
- Detección en tiempo real o sobre video.
- Contaminantes distintos a granos de arroz.
- Imágenes a color, con resolución distinta a 128×128, o no binarizadas.
- Despliegue en producción industrial real; el modelo fue entrenado con datos de laboratorio no controlados.

---

## Data Summary / Resumen de los datos

120 imágenes capturadas con un iPhone (formato HEIC) en interiores, sobre superficies planas claras. Las condiciones de captura no fueron controladas: distancia, ángulo e iluminación varían entre tomas. Las imágenes se convirtieron a PNG 128×128 en escala de grises y se binarizaron con umbral fijo (píxel > 127 → 1). Cada imagen se representa como un vector de 16 384 valores binarios.

**Variaciones presentes en el dataset:**
- Iluminación artificial de intensidad variable.
- Distancia a los objetos no estandarizada.
- Ángulo de captura levemente distinto entre imágenes.

Ver [DATASET.md](DATASET.md) para el detalle completo del preprocesamiento y las limitaciones del dataset.

---

## Labeling Process / Proceso de etiquetado

No se usó herramienta de anotación externa. La etiqueta se deduce automáticamente del nombre del archivo asignado durante la captura:

| Prefijo | Clase | Etiqueta |
|---|---|---|
| `1.N.png` | Sin arroz | 0 — negativo |
| `2.N.png` | Con arroz | 1 — positivo |

**Calidad:** la convención elimina errores de anotación manual, pero asume que los nombres fueron asignados correctamente en el momento de la captura. No se realizó revisión posterior de consistencia entre etiqueta y contenido visual.

---

## Metrics / Métricas

Evaluación sobre el mismo dataset de 120 imágenes con dos esquemas:

**Partición única — 80/20 estratificado (`RANDOM_STATE=42`):**

| Modelo | Acc | Precisión | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **12** | **12** | **0** | **0** |
| Árbol de Decisión | 0.6250 | 0.5882 | 0.8333 | 0.6897 | 10 | 5 | 7 | 2 |
| KNN | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 12 | 0 | 12 | 0 |
| Naive Bayes | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 | 0 | 12 |

**Evaluación robusta — 20 iteraciones con `StratifiedShuffleSplit` (80/20):**

| Modelo | Acc Media | ± Std | F1 Media |
|---|---|---|---|
| **SVM** | **0.9771** | **±0.0384** | **0.9789** |
| Árbol de Decisión | 0.6875 | ±0.0639 | 0.7440 |
| KNN | 0.5042 | ±0.0125 | 0.6686 |
| Naive Bayes | 0.5188 | ±0.0246 | 0.0681 |

SVM fue seleccionado por mayor exactitud media. Hiperparámetros óptimos: `C=1`, `kernel='rbf'`, encontrados con `GridSearchCV` (5 pliegues, `scoring='accuracy'`).

---

## Ethical / Safety Notes / Notas éticas y de sesgo

- **Sesgo de iluminación:** todas las capturas son en interiores con luz artificial. El modelo puede fallar bajo luz natural, sombras fuertes o cambios bruscos de exposición.
- **Sesgo de dispositivo:** las imágenes provienen de un único iPhone. La respuesta de sensor de otras cámaras puede alterar los valores de píxel antes de la binarización.
- **Sesgo de fondo:** el fondo es siempre una superficie clara de escritorio. Fondos con texturas, patrones o colores oscuros no están representados.
- **Clase negativa artificial:** los clips de oficina no representan el objeto real ausente en una línea de producción. El modelo aprende a separar arroz de clips, no arroz de ausencia de contaminación en general.

---

## Limitations / Limitaciones técnicas

- **Objetos pequeños o parciales:** comprimir a 128×128 puede fusionar granos juntos o perder los que están en bordes.
- **Oclusión:** el modelo no fue entrenado con imágenes donde el arroz esté parcialmente cubierto por otro objeto.
- **Blur:** imágenes desenfocadas se binarizan incorrectamente, produciendo vectores no representativos del contenido real.
- **Umbral fijo:** el valor 127 no se adapta a la exposición de cada imagen; fotos sobreexpuestas o subexpuestas generan vectores distorsionados.
- **Sin test independiente:** el conjunto de prueba proviene del mismo dataset de 120 imágenes bajo condiciones similares. Las métricas reportadas pueden ser optimistas respecto al rendimiento en datos nuevos.

---

## Reproducibility / Reproducibilidad

**Entrenamiento:**

```bash
# Desde la raíz del repositorio
python src/script/clasificacion.py --csv data/dataset.csv
```

El script aplica `GridSearchCV` con los mismos hiperparámetros, selecciona el mejor modelo y lo exporta a `models/`.

**Semillas fijadas:** `RANDOM_STATE=42`, `np.random.seed(42)`.

**Hardware:** Google Colab, CPU estándar, sin GPU. Tiempo total de búsqueda de hiperparámetros: ~18.7 s (los 4 modelos).

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

Ver [README.md](README.md) para los comandos de instalación y el flujo completo de reproducción.
