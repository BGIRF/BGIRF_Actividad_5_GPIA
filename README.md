# Actividad 5 · Preparación de datos, entrenamiento y registro de experimentos con MLflow

Universidad Tecmilenio - Gestión de Proyectos de Inteligencia Artificial

Brandon Gilberto Imanol Retana Fuentes

Modulo 2

Proyecto aplicado de MLOps: se toma un dataset público, se limpia y versiona, se entrenan y ajustan tres algoritmos de clasificación con `GridSearchCV` y validación cruzada, y todo el experimento (parámetros, métricas, datasets, artefactos y modelos) se registra en **MLflow** para garantizar trazabilidad y reproducibilidad.

**Problema:** Predecir si un vino *vinho verde* es de **calidad alta** (puntaje sensorial ≥ 7) a partir de 11 mediciones fisicoquímicas de laboratorio y el tipo de vino (tinto/blanco). Clasificación binaria con clases desbalanceadas (19 % positivos).

---

## 1. Estructura del repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/                    # datos originales (v1.0.0, sin modificar)
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   ├── datos_limp/                   # datos limpios (v1.1.0)
│   │   ├── wine_quality_limpio.csv
│   │   └── reporte_limpieza.json     # conteos de cada paso de limpieza
│   └── MANIFEST.json                 # versiones del dato + SHA-256 + linaje
├── fuentes/
│   ├── entrena.ipynb                 # notebook (Colab): flujo completo ejecutado
│   ├── obtener_datos.py              # descarga y verificación de integridad
│   ├── datos_prep.py                 # funciones de limpieza y estandarización
│   ├── visualizaciones.py            # gráficas de calidad de datos y evaluación
│   └── train.py                      # entrenamiento, GridSearchCV y registro en MLflow
├── salidas/
│   ├── figuras/                      # gráficas usadas en el reporte
│   ├── metricas/                     # tablas de resultados, pruebas estadísticas, runs exportados
│   └── mlflow_capturas/              # evidencia visual del panel de MLflow
├── requirements.txt
├── README.md                         # guía de reproducción y decisiones (este archivo)
└── CHANGELOG.md                      # historial de versiones y reporte final
```

`mlflow.db` y `mlruns/` (seguimiento local de MLflow) **no se versionan**: se regeneran al ejecutar el entrenamiento. La evidencia portable está en `salidas/mlflow_capturas/` y `salidas/metricas/mlflow_runs_exportados.csv`.

---

## 2. Cómo reproducir

### Opción A · Local (recomendada para ver el panel de MLflow)

```bash
git clone https://github.com/BGIRF/BGIRF_Actividad_5_GPIA
cd BGIRF_Actividad_5_GPIA
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fuentes/obtener_datos.py     # 1. verifica SHA-256 de los datos originales
python fuentes/datos_prep.py        # 2. limpieza -> datos/datos_limp/ (v1.1.0)
python fuentes/train.py             # 3. entrenamiento + registro en MLflow (~8-15 min en CPU)

mlflow ui --backend-store-uri sqlite:///mlflow.db       # 4. abrir http://127.0.0.1:5000
```

* `python fuentes/train.py --rapido` corre rejillas reducidas en ~1 min (prueba de humo).
* `python fuentes/train.py --tracking-uri http://servidor:5000` envía el registro a un servidor remoto de MLflow.
* En el panel de MLflow, seleccionar el modo **Model training** (arriba a la izquierda) para ver runs, métricas y modelos.

### Opción B · Google Colab

1. Abrir `fuentes/entrena.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. La primera celda ya apunta a este repositorio (`REPO_URL`); no hay que modificar nada.
3. *Entorno de ejecución → Ejecutar todo*. La celda inicial clona el repo e instala `mlflow`, `xgboost`, `statsmodels` y `skops`.
4. Para ver el panel, descargar `mlflow.db` y `mlruns/` y abrirlos localmente con el comando del paso 4.

### Reproducibilidad garantizada por

* Semilla fija (`SEMILLA = 42`) en partición, validación cruzada, modelos y bootstrap.
* Verificación SHA-256 del dato original **y** del dato limpio contra `datos/MANIFEST.json` antes de entrenar (si no coinciden, el script se detiene).
* Versiones exactas de librerías en `requirements.txt` y registradas como *tags* en cada run.
* Partición entrenamiento/prueba exportada (`salidas/metricas/particion_train_test.csv`).
* Modelo campeón recargado desde el *Model Registry* y verificado: reproduce exactamente las mismas probabilidades.

---

## 3. Dataset

| Atributo | Valor |
|---|---|
| Nombre | Wine Quality (vinho verde, Portugal) |
| Fuente | Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). UCI Machine Learning Repository, id 186 |
| Licencia | CC BY 4.0 |
| Registros originales | 6,497 (1,599 tintos + 4,898 blancos) |
| Variables | 11 fisicoquímicas + tipo + calidad (0–10) |
| Obtención | `fuentes/obtener_datos.py` (UCI con espejo en GitHub; verificación SHA-256) |

### Diccionario de datos (nombres estandarizados)

| Variable | Original | Unidad | Descripción |
|---|---|---|---|
| `tipo` | (archivo) | categoría | tinto / blanco |
| `acidez_fija` | fixed acidity | g ácido tartárico/L | Ácidos no volátiles |
| `acidez_volatil` | volatile acidity | g ácido acético/L | En exceso da sabor a vinagre |
| `acido_citrico` | citric acid | g/L | Aporta frescura |
| `azucar_residual` | residual sugar | g/L | Azúcar remanente tras la fermentación |
| `cloruros` | chlorides | g NaCl/L | Contenido de sal |
| `so2_libre` | free sulfur dioxide | mg/L | SO₂ libre (antimicrobiano/antioxidante) |
| `so2_total` | total sulfur dioxide | mg/L | SO₂ libre + combinado |
| `densidad` | density | g/cm³ | Depende de alcohol y azúcar |
| `ph` | pH | – | Acidez |
| `sulfatos` | sulphates | g K₂SO₄/L | Aditivo que contribuye al SO₂ |
| `alcohol` | alcohol | % vol | Grado alcohólico |
| `calidad` | quality | 0–10 | Mediana de ≥ 3 catadores |
| **`calidad_alta`** | (derivada) | 0/1 | **Objetivo:** 1 si `calidad` ≥ 7 |
| `excede_so2_ue` | (derivada) | 0/1 | SO₂ total > límite del Reg. (UE) 2019/934 (solo trazabilidad) |
| `n_atipicos_extremos` | (derivada) | conteo | Variables fuera de Q1−3·IQR / Q3+3·IQR (por tipo) |

### Supuestos

1. Las mediciones de laboratorio son correctas y comparables entre tintos y blancos (mismo laboratorio y métodos).
2. La calidad sensorial (mediana de catadores) es una etiqueta confiable; el umbral ≥ 7 define "calidad alta", práctica común en la literatura del dataset.
3. Registros idénticos en todas las variables son duplicados de captura y no vinos distintos.
4. No hay información temporal ni de bodega: se asume que los registros son independientes (no hay riesgo de fuga temporal).
5. El azúcar residual se usa como aproximación de glucosa + fructosa para aplicar el límite regulatorio de SO₂.

---

## 4. Limpieza y estandarización (`fuentes/datos_prep.py`)

| Paso | Regla / criterio | Resultado |
|---|---|---|
| Unión y linaje | Tinto + blanco, `id_registro` R0001…/W0001… para rastrear cada fila a su archivo original | 6,497 filas |
| Nombres | `snake_case` en español, consistentes en código, MLflow y reporte | 13 columnas |
| Tipos de datos | Numéricas a `float64` (valores no numéricos → NaN), `calidad` entera, `tipo` texto normalizado | 0 conversiones fallidas |
| Nulos | Sin objetivo → se elimina la fila; nulos en variables → imputación por mediana **dentro del Pipeline** | 0 nulos |
| Duplicados | Coincidencia exacta en tipo + 11 variables + calidad; se conserva la primera | **1,177 eliminados** (240 tintos, 937 blancos); 0 con etiqueta en conflicto |
| Reglas de validez (R1–R4) | Concentraciones ≥ 0; SO₂ libre ≤ SO₂ total; pH 2.5–4.5; densidad 0.98–1.05; alcohol 7–16 %; calidad 0–10 | 0 violaciones (quedan como validaciones automáticas) |
| Regla regulatoria (R5) | SO₂ total > límite del Reg. (UE) 2019/934 (tinto 150/200 mg/L, blanco 200/250 mg/L según azúcar ≥ 5 g/L) | 63 marcados, **no** se eliminan |
| Atípicos | Valores fuera de Q1−3·IQR / Q3+3·IQR por tipo de vino | 302 registros marcados, **no** se eliminan |
| Objetivo | `calidad_alta` = 1 si `calidad` ≥ 7 | 19.0 % positivos (5,320 filas finales) |

**Escalado y codificación** no se guardan en el CSV limpio: se ajustan dentro del `Pipeline` de scikit-learn (solo con datos de entrenamiento de cada pliegue) para evitar *data leakage*:

* Numéricas: `SimpleImputer(median)` → `StandardScaler` o `PowerTransformer(yeo-johnson)` (regresión logística; el escalador es un hiperparámetro del *grid*) / sin escalado (árboles).
* `tipo`: `OneHotEncoder(drop="if_binary")` → una columna binaria.

---

## 5. Estrategia de versionamiento

| Qué se versiona | Cómo |
|---|---|
| **Datos** | Versionado semántico del dato: `v1.0.0` (originales) → `v1.1.0` (limpios). `datos/MANIFEST.json` guarda ruta, filas, SHA-256 y linaje de cada versión. |
| **Código** | Git + GitHub (historial de *commits* del repositorio); la versión entregada corresponde a la `v1.0.1` del CHANGELOG. |
| **Cambios** | `CHANGELOG.md` (formato *Keep a Changelog*). |
| **Experimentos** | MLflow: cada run guarda `dataset_version`, `dataset_sha256`, `git_commit` y versiones de librerías; los datasets de entrenamiento y prueba se registran con `mlflow.log_input` (pestaña *Datasets*). |
| **Modelos** | MLflow Model Registry: `wine_calidad_alta_clf`, alias `champion`, con métricas y umbral como *tags* de la versión. |

Así, cualquier predicción se puede rastrear: **modelo registrado → run → hiperparámetros y métricas → hash del dataset → commit de código → archivo original**.

> Siguiente paso natural (deuda técnica): migrar el versionamiento de datos a **DVC** con un *remote* (Google Drive/S3) para no almacenar CSV en Git cuando el dato crezca.

---

## 6. Modelos, hiperparámetros y registro en MLflow (`fuentes/train.py`)

**Protocolo común:** partición 80/20 estratificada por clase y tipo (4,256 / 1,064 vinos); `GridSearchCV` con `StratifiedKFold(k=5, shuffle, seed=42)`, siete métricas por pliegue y *refit* por **ROC-AUC** (independiente del umbral y adecuada con desbalance). El umbral de decisión se fija maximizando F1 en predicciones *out-of-fold* del entrenamiento; el conjunto de prueba solo se usa para el reporte final.

| Modelo | Por qué se incluye | Rejilla explorada | Mejor configuración |
|---|---|---|---|
| Línea base `DummyClassifier` | Piso mínimo: siempre predice la clase mayoritaria | – | – |
| Regresión logística | Modelo lineal, interpretable, rápido; referencia clásica | escalador {z-score, Yeo-Johnson} × C {0.01, 0.1, 1, 10, 100} × class_weight {None, balanced} = **20** | z-score, C = 1, class_weight = None |
| Random Forest | Ensamble *bagging*: no lineal, robusto a atípicos y a variables correlacionadas | max_depth {10, 20, None} × min_samples_leaf {1, 3, 5} × max_features {sqrt, 0.5} × class_weight {None, balanced_subsample}; 300 árboles = **36** | max_depth = 20, min_samples_leaf = 3, max_features = sqrt, balanced_subsample |
| XGBoost | Ensamble *boosting*: suele ser el estado del arte en datos tabulares | n_estimators {200, 400} × max_depth {3, 5, 7} × learning_rate {0.05, 0.1} × subsample {0.8, 1.0} × scale_pos_weight {1, 4} = **48** | 200 árboles, max_depth = 7, learning_rate = 0.05, subsample = 0.8, scale_pos_weight = 1 |

Justificación de las rejillas: `C` barre cuatro órdenes de magnitud de regularización; Yeo-Johnson se prueba porque azúcar, cloruros y SO₂ son muy asimétricos; `class_weight`/`scale_pos_weight` atacan el desbalance 81/19; en los árboles se controla la complejidad (`max_depth`, `min_samples_leaf`), la decorrelación (`max_features`, `subsample`) y el compromiso `learning_rate`–`n_estimators`.

### Organización del experimento en MLflow

Experimento `Actividad5_WineQuality_CalidadAlta`:

| Run | Contenido |
|---|---|
| `00_linea_base_dummy` | `DummyClassifier` (siempre predice la clase mayoritaria): referencia mínima |
| `01_logreg_gridsearch` + 20 hijos | un run hijo por combinación de la rejilla (parámetros + métricas de CV) |
| `02_rf_gridsearch` + 36 hijos | ídem |
| `03_xgb_gridsearch` + 48 hijos | ídem |
| `04_comparacion_estadistica` | CV repetida 5×3, pruebas estadísticas, tablas y figuras comparativas |

Cada run padre registra: mejores hiperparámetros, la rejilla completa, métricas de CV (media y desviación) y de prueba, umbral de decisión, datasets de entrada, `cv_results.csv`, reporte de clasificación y el **modelo** (serializado con *skops*, con firma de entrada/salida y ejemplo). El campeón se registra en el *Model Registry* con alias `champion`.

### Evidencia del panel de MLflow

Capturas en `salidas/mlflow_capturas/`:

| Archivo | Qué demuestra |
|---|---|
| `01_runs_padres_con_metricas.png` | Los 5 runs principales con dataset, modelo registrado y métricas de CV/prueba |
| `02_runs_hijos_gridsearch_random_forest.png` | Runs anidados: cada combinación del *grid* con sus hiperparámetros y ROC-AUC de CV |
| `03_detalle_run_random_forest.png` | Detalle del run: métricas, *tags* de trazabilidad (hash del dataset, versiones), datasets y modelo registrado |
| `04_coordenadas_paralelas_grid_random_forest.png` | Comparación de las 36 combinaciones de Random Forest (hiperparámetros vs. ROC-AUC) |
| `05_artefactos_run_comparacion.png` | Artefactos (figuras y tablas) guardados en MLflow |
| `06_modelos_logueados_experimento.png` | Modelos serializados por run, con su métrica de CV |
| `07_model_registry_alias_champion.png` | *Model Registry*: versión 1 con alias `champion` y *tags* de métricas |

**Guion sugerido para grabar el video de evidencia (2–3 min):** (1) ejecutar `python fuentes/train.py` y mostrar la salida en consola; (2) `mlflow ui --backend-store-uri sqlite:///mlflow.db` y cambiar a **Model training**; (3) en *Runs*, filtrar con `tags.rol != 'combinacion_grid'` y agregar las columnas `cv_roc_auc_media`, `test_roc_auc` y `test_f1`; (4) expandir `02_rf_gridsearch` para mostrar los runs hijos; (5) abrir el run de Random Forest (parámetros, métricas, *tags*, datasets); (6) seleccionar los hijos de un modelo → *Compare* → coordenadas paralelas; (7) pestaña *Artifacts* del run de comparación; (8) *Model registry* → `wine_calidad_alta_clf` → alias `champion`.

---

## 7. Resultados

### Validación cruzada (entrenamiento, 5 pliegues) y conjunto de prueba (umbral optimizado)

| Modelo | ROC-AUC CV | PR-AUC CV | ROC-AUC prueba | PR-AUC prueba | Precisión | Recall | F1 | Accuracy | Umbral |
|---|---|---|---|---|---|---|---|---|---|
| Línea base | 0.500 | 0.190 | 0.500 | 0.190 | 0.000 | 0.000 | 0.000 | 0.810 | 0.50 |
| Regresión logística | 0.825 ± 0.017 | 0.506 | 0.846 | 0.557 | 0.452 | 0.772 | 0.570 | 0.779 | 0.22 |
| **Random Forest (campeón)** | **0.854 ± 0.017** | **0.576** | 0.868 | 0.612 | 0.483 | 0.752 | 0.588 | 0.800 | 0.36 |
| XGBoost | 0.848 ± 0.015 | 0.556 | 0.871 | 0.619 | 0.507 | 0.738 | 0.601 | 0.814 | 0.24 |

### Comparación estadística

* **Friedman** sobre ROC-AUC en CV repetida 5×3 (15 pliegues pareados): χ² = 28.1, p < 0.001 → al menos un modelo difiere.
* **Árboles vs. regresión logística:** RF y XGBoost ganan en los 15 pliegues; t corregida de Nadeau-Bengio con ajuste de Holm p = 0.003 (ambos pares).
* **Random Forest vs. XGBoost:** Δ = +0.005 a favor de RF en CV (14/15 pliegues), t corregida p = 0.059 → **no significativa**. En prueba XGBoost queda 0.003 arriba, IC 95 % bootstrap de la diferencia [−0.011, 0.005] y McNemar p = 0.07 → **estadísticamente equivalentes**.
* **Campeón:** Random Forest, por el criterio fijado de antemano (mayor ROC-AUC de CV). Registrado como `models:/wine_calidad_alta_clf@champion` (v1); al recargarlo reproduce exactamente las mismas predicciones.

Detalle completo: `salidas/metricas/` (CSV), `salidas/figuras/` y el reporte técnico.

---

## 8. Repositorio

https://github.com/BGIRF/BGIRF_Actividad_5_GPIA

---

## 9. Referencias

* Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553.
* Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.
* Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.
* Reglamento Delegado (UE) 2019/934 de la Comisión, Anexo I, Parte B (límites de anhídrido sulfuroso en vinos).
* MLflow Project (2026). *MLflow documentation*. https://mlflow.org/docs/latest/
* Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
