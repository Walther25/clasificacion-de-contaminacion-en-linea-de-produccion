# =============================================================================
# clasificacion.py
# Entrena y compara Árbol de Decisión, Naive Bayes, KNN y SVM sobre un dataset
# de imágenes binarizadas 128x128 para detectar contaminaciones (arroz).
#
# Requisitos:
#   pip install scikit-learn pandas numpy matplotlib seaborn joblib
#
# Uso:
#   python clasificacion.py
#   python clasificacion.py --csv mi_dataset.csv
#   python clasificacion.py --csv dataset1.csv dataset2.csv
# =============================================================================

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# CONFIGURACIÓN — editá tus datos aquí
# =============================================================================

CARNE    = "C00000"            # <- Tu carné
NOMBRE   = "NombreEstudiante"  # <- Tu nombre
APELLIDO = "ApellidoEstudiante"  # <- Tu apellido

TEST_SIZE        = 0.2
N_REPETICIONES   = 20
CV_FOLDS         = 5
RANDOM_STATE     = 42
COLUMNA_ETIQUETA = "etiqueta"

# CSVs por defecto si no se pasan por argumento
CSV_POR_DEFECTO = ["dataset.csv"]

# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_un_csv(ruta_csv: str) -> tuple:
    df = pd.read_csv(ruta_csv)

    if COLUMNA_ETIQUETA not in df.columns:
        df = pd.read_csv(ruta_csv, header=None)
        col_etiqueta = df.columns[-1]
        print(f"  Sin encabezado reconocido -> etiqueta = col {col_etiqueta}")
    else:
        col_etiqueta = COLUMNA_ETIQUETA

    df = df.apply(pd.to_numeric, errors="coerce")
    filas_antes = len(df)
    df = df.dropna()
    if len(df) < filas_antes:
        print(f"  Filas descartadas por datos no numéricos: {filas_antes - len(df)}")

    X = df.drop(columns=[col_etiqueta]).values.astype(int)
    y = df[col_etiqueta].values.astype(int)
    return X, y


def cargar_datasets(lista_csv: list) -> tuple:
    Xs, ys = [], []
    print(f"Cargando {len(lista_csv)} archivo(s)...\n")

    for ruta in lista_csv:
        if not Path(ruta).exists():
            print(f"  ERROR: no se encontró el archivo '{ruta}'. Omitido.")
            continue

        print(f"Archivo: {ruta}")
        X_i, y_i = cargar_un_csv(ruta)

        if X_i.shape[1] != 128 * 128:
            print(f"  ERROR: se esperaban {128*128} píxeles, tiene {X_i.shape[1]}. Omitido.")
            continue

        print(f"  {X_i.shape[0]} imágenes  (positivos={y_i.sum()}, negativos={(y_i==0).sum()})")
        Xs.append(X_i)
        ys.append(y_i)

    if not Xs:
        raise ValueError("No se pudo cargar ningún CSV válido.")

    return np.vstack(Xs), np.concatenate(ys)

# =============================================================================
# MODELOS
# =============================================================================

def obtener_modelos() -> dict:
    return {
        "Arbol de Decision": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth":         [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "criterion":         ["gini", "entropy"],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11],
                "metric":      ["euclidean", "manhattan"],
            },
        ),
        "SVM": (
            SVC(random_state=RANDOM_STATE, probability=True),
            {
                "C":      [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            },
        ),
    }

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_grid(modelos: dict, X_train, y_train, X_test, y_test) -> tuple:
    resultados_grid, tiempos = {}, {}

    for nombre, (modelo, params) in modelos.items():
        print("=" * 55)
        print(f"Buscando hiperparámetros: {nombre}")
        print("=" * 55)

        inicio = time.time()
        grid = GridSearchCV(
            estimator=clone(modelo),
            param_grid=params,
            cv=CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        t = time.time() - inicio

        tiempos[nombre]        = t
        resultados_grid[nombre] = grid

        y_pred   = grid.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)

        print(f"  Mejores hiperparámetros : {grid.best_params_}")
        print(f"  Exactitud en CV (train) : {round(grid.best_score_ * 100, 2)}%")
        print(f"  Exactitud en prueba     : {round(acc_test * 100, 2)}%")
        print(f"  Tiempo de búsqueda      : {round(t, 1)}s\n")

    print("GridSearchCV completado para todos los modelos")
    return resultados_grid, tiempos

# =============================================================================
# EVALUACIÓN ROBUSTA
# =============================================================================

def evaluar_repetido(grid_ajustado, X, y) -> tuple:
    splitter = StratifiedShuffleSplit(
        n_splits=N_REPETICIONES, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    accuracies, f1s = [], []

    for train_idx, test_idx in splitter.split(X, y):
        m = clone(grid_ajustado.best_estimator_)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        accuracies.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))

    return accuracies, f1s

# =============================================================================
# GRÁFICAS
# =============================================================================

def graficar_distribucion(y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    clases  = ["Negativo (0)\nSin arroz", "Positivo (1)\nCon arroz"]
    conteos = [(y == 0).sum(), (y == 1).sum()]
    colores = ["#3498db", "#e74c3c"]

    bars = axes[0].bar(clases, conteos, color=colores, edgecolor="white", linewidth=1.5, width=0.5)
    axes[0].set_title("Distribucion de Clases", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Numero de imagenes")
    for bar, cnt in zip(bars, conteos):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                     str(cnt), ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, max(conteos) * 1.25)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].pie(conteos, labels=clases, colors=colores, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Proporcion de Clases", fontsize=13, fontweight="bold")

    plt.suptitle("Analisis del Dataset", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("distribucion_clases.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: distribucion_clases.png")


def graficar_muestra(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Muestra del Dataset (reconstruidas desde pixeles)", fontsize=13, fontweight="bold")

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n = min(5, len(idx_pos), len(idx_neg))

    for i in range(n):
        axes[0, i].imshow(X[idx_pos[i]].reshape(128, 128), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Positivo #{idx_pos[i]}\n(con arroz)", fontsize=9, color="#e74c3c")
        axes[0, i].axis("off")

        axes[1, i].imshow(X[idx_neg[i]].reshape(128, 128), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Negativo #{idx_neg[i]}\n(sin arroz)", fontsize=9, color="#3498db")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("muestra_imagenes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: muestra_imagenes.png")


def graficar_comparacion(eval_resultados):
    nombres    = list(eval_resultados.keys())
    acc_medias = [eval_resultados[n]["acc_media"] for n in nombres]
    acc_stds   = [eval_resultados[n]["acc_std"]   for n in nombres]
    f1_medias  = [eval_resultados[n]["f1_media"]  for n in nombres]
    f1_stds    = [eval_resultados[n]["f1_std"]    for n in nombres]

    orden     = np.argsort(acc_medias)[::-1]
    nombres_o = [nombres[i]    for i in orden]
    acc_med_o = [acc_medias[i] for i in orden]
    acc_std_o = [acc_stds[i]   for i in orden]
    f1_med_o  = [f1_medias[i]  for i in orden]
    f1_std_o  = [f1_stds[i]    for i in orden]
    colores   = ["#27ae60", "#3498db", "#9b59b6", "#e67e22"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Comparacion de Modelos — Evaluacion Repetida (N={N_REPETICIONES})",
                 fontsize=13, fontweight="bold")

    for ax, medias, stds, titulo, etiqueta in [
        (axes[0], acc_med_o, acc_std_o, "Exactitud Media +/- Std", "Exactitud"),
        (axes[1], f1_med_o,  f1_std_o,  "F1-Score Medio +/- Std",  "F1-Score"),
    ]:
        bars = ax.bar(nombres_o, medias, yerr=stds, capsize=6,
                      color=colores, edgecolor="white", linewidth=1.5)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_ylabel(etiqueta)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Azar (50%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val, std in zip(bars, medias, stds):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.02,
                    str(round(val, 3)), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: comparacion_modelos.png")

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    datos_box = [eval_resultados[n]["accuracies"] for n in nombres_o]
    bp = ax.boxplot(datos_box, labels=nombres_o, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"Distribucion de Exactitudes — {N_REPETICIONES} Repeticiones",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Exactitud por iteracion")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Azar (50%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("boxplot_exactitudes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: boxplot_exactitudes.png")

    return nombres_o


def graficar_matrices(resultados_grid, X_test, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Matrices de Confusion (conjunto de prueba principal)",
                 fontsize=12, fontweight="bold")

    for ax, (nombre, grid) in zip(axes, resultados_grid.items()):
        y_pred = grid.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negativo\n(sin arroz)", "Positivo\n(con arroz)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{nombre}\nAcc={round(accuracy_score(y_test, y_pred), 3)}",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("matrices_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: matrices_confusion.png")


def graficar_arbol(resultados_grid):
    arbol_grid  = resultados_grid["Arbol de Decision"]
    arbol_mejor = arbol_grid.best_estimator_
    depth_real  = arbol_mejor.get_depth()
    viz_depth   = min(depth_real, 4)

    print(f"  Profundidad real : {depth_real}")
    print(f"  Número de hojas  : {arbol_mejor.get_n_leaves()}")

    plt.figure(figsize=(20, 8))
    plot_tree(arbol_mejor, max_depth=viz_depth,
              class_names=["Negativo", "Positivo"],
              filled=True, rounded=True, fontsize=8, impurity=False)
    plt.title(
        f"Arbol de Decision (primeros {viz_depth} niveles de {depth_real})\n"
        "Naranja = Positivo (arroz), Azul = Negativo",
        fontsize=12, fontweight="bold"
    )
    plt.savefig("arbol_decision.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado: arbol_decision.png")

# =============================================================================
# TABLA UNIFICADA
# =============================================================================

def imprimir_tabla_unificada(resultados_grid, eval_resultados, tiempos, X_test, y_test):
    filas = []
    for nombre, grid in resultados_grid.items():
        r      = eval_resultados[nombre]
        y_pred = grid.predict(X_test)

        acc_test  = accuracy_score(y_test, y_pred)
        prec_test = precision_score(y_test, y_pred, zero_division=0)
        rec_test  = recall_score(y_test, y_pred, zero_division=0)
        f1_test   = f1_score(y_test, y_pred, zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])

        filas.append({
            "Modelo"           : nombre,
            "Acc Media (±Std)" : f"{r['acc_media']:.4f} ±{r['acc_std']:.4f}",
            "Acc Min"          : f"{min(r['accuracies']):.4f}",
            "Acc Max"          : f"{max(r['accuracies']):.4f}",
            "F1 Media (±Std)"  : f"{r['f1_media']:.4f} ±{r['f1_std']:.4f}",
            "Acc Prueba"       : f"{acc_test:.4f}",
            "Precisión"        : f"{prec_test:.4f}",
            "Recall"           : f"{rec_test:.4f}",
            "F1 Prueba"        : f"{f1_test:.4f}",
            "TP"               : int(tp),
            "TN"               : int(tn),
            "FP"               : int(fp),
            "FN"               : int(fn),
            "Acc CV (train)"   : f"{grid.best_score_:.4f}",
            "Tiempo (s)"       : f"{tiempos[nombre]:.1f}",
            "Mejores Params"   : str(r["best_params"]),
        })

    df = (pd.DataFrame(filas)
            .sort_values("Acc Media (±Std)", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║         TABLA UNIFICADA DE MÉTRICAS — TODOS LOS MODELOS         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(df.to_string())
    print()
    print("Leyenda: Acc=Exactitud | Prec=Precisión | Rec=Recall | F1=F1-Score")
    print("         TP=Verdadero Positivo | TN=Verdadero Negativo")
    print("         FP=Falso Positivo     | FN=Falso Negativo")
    print()
    mejor_idx = df.index[0]
    print(f"► MEJOR MODELO: {df.loc[mejor_idx, 'Modelo']}  "
          f"(Acc Media = {df.loc[mejor_idx, 'Acc Media (±Std)']})")
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clasificación de contaminaciones")
    parser.add_argument("--csv", nargs="+", default=CSV_POR_DEFECTO,
                        help="Uno o más archivos CSV de entrada")
    args = parser.parse_args()

    print(f"\nEstudiante  : {NOMBRE} {APELLIDO} ({CARNE})")
    print(f"Split       : {int((1-TEST_SIZE)*100)}% entrenamiento / {int(TEST_SIZE*100)}% prueba")
    print(f"Repeticiones: {N_REPETICIONES}\n")

    # 1. Cargar datos
    X, y = cargar_datasets(args.csv)
    print(f"\n{'='*45}\nDATASET COMBINADO\n{'='*45}")
    print(f"Total imágenes : {X.shape[0]}")
    print(f"Píxeles/imagen : {X.shape[1]} (128x128)")
    print(f"Positivos      : {(y==1).sum()} (con arroz)")
    print(f"Negativos      : {(y==0).sum()} (sin arroz)\n")

    # 2. Exploración
    graficar_distribucion(y)
    graficar_muestra(X, y)

    # 3. División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nEntrenamiento : {X_train.shape[0]} imágenes")
    print(f"Prueba        : {X_test.shape[0]} imágenes\n")

    # 4. Modelos
    modelos = obtener_modelos()
    print(f"{len(modelos)} algoritmos definidos:")
    for nombre, (_, params) in modelos.items():
        n_comb = 1
        for v in params.values():
            n_comb *= len(v)
        print(f"  {nombre}: {n_comb} combinaciones")

    # 5. GridSearchCV
    print()
    resultados_grid, tiempos = entrenar_grid(modelos, X_train, y_train, X_test, y_test)

    # 6. Evaluación robusta
    print(f"\nEvaluacion repetida ({N_REPETICIONES} iteraciones por modelo)...\n")
    eval_resultados = {}
    for nombre, grid in resultados_grid.items():
        print(f"  Evaluando: {nombre}...", end=" ", flush=True)
        accs, f1s = evaluar_repetido(grid, X, y)
        eval_resultados[nombre] = {
            "accuracies": accs, "f1_scores": f1s,
            "acc_media": np.mean(accs), "acc_std": np.std(accs),
            "f1_media": np.mean(f1s), "f1_std": np.std(f1s),
            "best_params": grid.best_params_,
        }
        print(f"Acc={round(np.mean(accs), 4)} ±{round(np.std(accs), 4)}  F1={round(np.mean(f1s), 4)}")

    # 7. Gráficas
    graficar_comparacion(eval_resultados)
    graficar_matrices(resultados_grid, X_test, y_test)
    graficar_arbol(resultados_grid)

    # 8. Tabla unificada
    imprimir_tabla_unificada(resultados_grid, eval_resultados, tiempos, X_test, y_test)

    # 9. Exportar mejor modelo
    mejor_nombre = max(eval_resultados, key=lambda n: eval_resultados[n]["acc_media"])
    mejor_r      = eval_resultados[mejor_nombre]
    mejor_grid   = resultados_grid[mejor_nombre]

    print(f"\nMODELO GANADOR: {mejor_nombre}")
    print(f"Reentrenando con el 100% de los datos...")
    modelo_final = clone(mejor_grid.best_estimator_)
    modelo_final.fit(X, y)

    nombre_archivo = f"{CARNE}_{NOMBRE}_{APELLIDO}.joblib"
    joblib.dump(modelo_final, nombre_archivo)

    modelo_cargado = joblib.load(nombre_archivo)
    y_verif = modelo_cargado.predict(X[:3])
    print(f"Verificación (primeras 3 filas):")
    print(f"  Predicciones : {y_verif.tolist()}")
    print(f"  Reales       : {y[:3].tolist()}")
    print(f"Modelo exportado: {nombre_archivo}")

    # 10. Resumen final
    print(f"\n{'='*65}")
    print("  RESUMEN FINAL — PROYECTO 1")
    print("  Clasificación de Contaminaciones en Línea de Producción")
    print(f"{'='*65}")
    print(f"  Estudiante  : {NOMBRE} {APELLIDO} ({CARNE})")
    print(f"  Dataset     : {len(X)} imágenes x {X.shape[1]} píxeles (128×128)")
    print(f"  Clases      : {(y==1).sum()} positivos, {(y==0).sum()} negativos")
    print(f"  Split       : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} — Evaluación repetida {N_REPETICIONES}x")
    print(f"\n  MODELO EXPORTADO : {mejor_nombre}")
    print(f"  Hiperparámetros  : {mejor_r['best_params']}")
    print(f"  Archivo          : {nombre_archivo}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
