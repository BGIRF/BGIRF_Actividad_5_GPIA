# CHANGELOG · Actividad 5

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Se versionan por separado el **código** (tags de Git `vX.Y.Z`) y el **dato** (`dataset vX.Y.Z`, con huellas SHA-256 en `datos/MANIFEST.json`).

---

## [1.0.1] · 2026-10-18 · Reejecución completa y evidencias actualizadas

### Cambiado
- Se reejecutó el flujo completo (`datos_prep.py` → `train.py`, desde `fuentes/entrena.ipynb`) con la misma semilla (42). El dataset limpio conserva su huella SHA-256 (v1.1.0) y todas las métricas son idénticas a las de la v1.0.0.
- Experimento de MLflow regenerado desde cero (109 runs) y modelo campeón registrado de nuevo como `wine_calidad_alta_clf` v1 con alias `champion`; capturas de `salidas/mlflow_capturas/` actualizadas.
- Run IDs nuevos en `salidas/metricas/resumen_ejecucion.json` y `salidas/metricas/mlflow_runs_exportados.csv`.
- README y notebook apuntan al repositorio `BGIRF/BGIRF_Actividad_5_GPIA` (clonación local y celda de Colab).

## [1.0.0] · 2026-10-15 · Modelos entrenados y registrados en MLflow

### Añadido
- `fuentes/train.py`: entrenamiento de Regresión logística, Random Forest y XGBoost con `GridSearchCV` (5-fold estratificado, 104 combinaciones en total) más una línea base `DummyClassifier`.
- Registro en MLflow (`sqlite:///mlflow.db`): un run padre por modelo con mejores hiperparámetros, métricas de CV y de prueba, datasets de entrada (`log_input`), artefactos y modelo serializado con *skops* (firma + ejemplo); un run hijo por cada combinación de la rejilla.
- Run `04_comparacion_estadistica`: CV repetida 5×3, Friedman, t corregida de Nadeau-Bengio y Wilcoxon (Holm), McNemar y bootstrap pareado en prueba.
- Model Registry: `wine_calidad_alta_clf` v1 (Random Forest) con alias `champion` y *tags* de métricas, umbral y versión del dato.
- `fuentes/entrena.ipynb` (Colab) ejecutado de inicio a fin; `fuentes/visualizaciones.py`; figuras en `salidas/figuras/`; tablas en `salidas/metricas/`; capturas del panel en `salidas/mlflow_capturas/`.

### Decisiones
- Métrica de selección: ROC-AUC de validación cruzada (no la prueba).
- Umbral de decisión por modelo: máximo F1 en predicciones *out-of-fold* del entrenamiento.
- Serialización *skops* con tipos de confianza explícitos en lugar de *pickle* (más segura al cargar).

## [0.2.0] · 2026-10-15 · Dataset limpio v1.1.0

### Añadido
- `fuentes/datos_prep.py` con funciones de limpieza y estandarización.
- `datos/datos_limp/wine_quality_limpio.csv` (5,320 filas) y `reporte_limpieza.json`.
- `datos/MANIFEST.json` con versión, filas, SHA-256 y linaje de cada versión del dato.

### Cambiado (dataset 1.0.0 → 1.1.0)
- Unión de tintos y blancos con columna `tipo` e `id_registro` de linaje (R0001…/W0001…).
- Nombres de columnas en `snake_case` español y tipos explícitos.
- **Eliminados 1,177 duplicados exactos** (240 tintos, 937 blancos; 18.1 %).
- Nuevas columnas: `calidad_alta` (objetivo, calidad ≥ 7), `excede_so2_ue` (63 vinos sobre el límite del Reg. (UE) 2019/934) y `n_atipicos_extremos` (302 registros con algún valor fuera de 3·IQR).
- Reglas de validez R1–R4: 0 filas eliminadas (se conservan como validaciones automáticas).

## [0.1.0] · 2026-10-15 · Estructura y datos originales (dataset v1.0.0)

### Añadido
- Estructura del repositorio, `.gitignore` y `requirements.txt` con versiones fijas.
- `datos/datos_ini/winequality-red.csv` (1,599) y `winequality-white.csv` (4,898), sin modificar.
- `fuentes/obtener_datos.py`: descarga (UCI con espejo en GitHub) y verificación SHA-256.

---

# Reporte final

## Resumen de resultados

| Modelo | ROC-AUC CV | ROC-AUC prueba | PR-AUC prueba | F1 prueba | Recall | Precisión |
|---|---|---|---|---|---|---|
| Línea base | 0.500 | 0.500 | 0.190 | 0.000 | 0.000 | 0.000 |
| Regresión logística | 0.825 | 0.846 | 0.557 | 0.570 | 0.772 | 0.452 |
| **Random Forest** ★ | **0.854** | 0.868 | 0.612 | 0.588 | 0.752 | 0.483 |
| XGBoost | 0.848 | 0.871 | 0.619 | 0.601 | 0.738 | 0.507 |

- Los ensambles de árboles superan a la regresión logística de forma significativa (p corregida = 0.003). Random Forest y XGBoost son estadísticamente equivalentes (t corregida p = 0.059 en CV; IC 95 % de ΔAUC en prueba [−0.011, 0.005]).
- Campeón registrado: **Random Forest** (`max_depth=20`, `min_samples_leaf=3`, `max_features=sqrt`, `class_weight=balanced_subsample`, umbral 0.36) → `models:/wine_calidad_alta_clf@champion`.
- Variable más influyente en todos los modelos: **alcohol**. La regresión logística depende además de densidad y azúcar (colinealidad).

## Deuda técnica identificada y acciones MLOps

| Deuda técnica | Riesgo | Acción propuesta |
|---|---|---|
| Brecha entrenamiento–CV de 0.14–0.15 de ROC-AUC en los ensambles | Sobreajuste; sensibilidad a datos nuevos | Ampliar la rejilla hacia más regularización (`min_samples_leaf`, `min_child_weight`, `reg_lambda`), búsqueda bayesiana (Optuna) y validación anidada |
| Datos versionados como CSV en Git | El repositorio crece; sin *remote* de datos | Migrar a **DVC** con *remote* (Drive/S3) y `dvc.yaml` para el pipeline datos → modelo |
| Seguimiento local en SQLite | Sin colaboración ni historial compartido | Servidor de MLflow con base de datos y *artifact store* remotos |
| Umbral calibrado una sola vez; probabilidades no calibradas | Decisiones inconsistentes si cambia la prevalencia | Calibración (Platt/isotónica) y umbral definido con costos del negocio; revisión periódica |
| Sin pruebas automáticas | Un cambio en la limpieza puede romper el modelo sin aviso | `pytest` para reglas de negocio y esquema (Pandera/Great Expectations) + CI en GitHub Actions |
| Sin monitoreo en producción | *Data drift* (nuevas cosechas, otras bodegas) no detectado | Monitoreo de distribución de variables y desempeño (Evidently) con reentrenamiento disparado por umbral |
| Dataset de una sola región (vinho verde, 2004–2007) | Generalización limitada | Documentar alcance en la *model card*; validar con datos de otras regiones antes de usar fuera de este contexto |
