"""
train.py
--------
Entrenamiento, ajuste de hiperparámetros y registro de experimentos en MLflow.

Problema: clasificación binaria de vinos de calidad alta (calidad >= 7) a partir
de 11 variables fisicoquímicas + tipo de vino (dataset limpio v1.1.0).

Modelos comparados (mismas particiones, misma semilla):
    00  Línea base: DummyClassifier (predice siempre la clase mayoritaria)
    01  Regresión logística   + GridSearchCV (20 combinaciones)
    02  Random Forest         + GridSearchCV (36 combinaciones)
    03  XGBoost               + GridSearchCV (48 combinaciones)
    04  Comparación estadística (CV repetida 5x3, Friedman, t de Nadeau-Bengio,
        Wilcoxon, McNemar y bootstrap) + registro del modelo campeón.

Cada combinación de la rejilla queda como run hijo (nested) en MLflow; el run
padre de cada modelo guarda los mejores hiperparámetros, métricas de CV y de
prueba, artefactos y el modelo serializado con firma y ejemplo de entrada.

Uso:
    python fuentes/train.py                    # experimento completo
    python fuentes/train.py --rapido           # rejillas reducidas (prueba de humo)
    python fuentes/train.py --tracking-uri http://127.0.0.1:5000
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # ver resultados
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,  # noqa: E402
                             classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402
import visualizaciones as viz  # noqa: E402

from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# ---------------------------------------------------------------------------
# Configuración (todo lo que afecta el resultado queda registrado en MLflow)
# ---------------------------------------------------------------------------
RAIZ = dp.RAIZ
DIR_MET = RAIZ / "salidas" / "metricas"
SEMILLA = 42
PROPORCION_PRUEBA = 0.20
K_PLIEGUES = 5
REPETICIONES_COMPARACION = 3
N_BOOTSTRAP = 2000
EXPERIMENTO = "Actividad5_WineQuality_CalidadAlta"
MODELO_REGISTRADO = "wine_calidad_alta_clf"
AUTOR = "Brandon Gilberto Imanol Retana Fuentes"
METRICA_SELECCION = "roc_auc"
ESCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "precision": "precision",
    "recall": "recall", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
}

# Tipos que skops no confía por defecto; se declaran explícitamente (verificado con skops.io.get_untrusted_types)
TIPOS_CONFIABLES = ["numpy.dtype", "sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
NOMBRES = {"dummy": "Línea base (Dummy)", "logreg": "Regresión logística",
           "rf": "Random Forest", "xgb": "XGBoost"}


def preprocesador(escalar: bool) -> ColumnTransformer:
    """Imputación por mediana + (escalado opcional) para numéricas; one-hot para `tipo`.
    Se ajusta DENTRO de cada pliegue -> sin fuga de información."""
    num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler() if escalar else "passthrough")])
    return ColumnTransformer([("num", num, dp.VARIABLES),
                              ("cat", OneHotEncoder(drop="if_binary"), dp.CATEGORICAS)],
                             verbose_feature_names_out=False)


def definir_modelos(rapido: bool = False) -> dict:
    """Pipelines y rejillas de hiperparámetros con su justificación técnica."""
    modelos = {
        "logreg": {
            "pipeline": Pipeline([("prep", preprocesador(True)),
                                  ("clf", LogisticRegression(max_iter=5000, random_state=SEMILLA))]),
            "rejilla": {
                # z-score vs. Yeo-Johnson: varias variables son asimétricas (azúcar, cloruros, SO2)
                "prep__num__escalar": [StandardScaler(), PowerTransformer(method="yeo-johnson")],
                # C barre 4 órdenes de magnitud de regularización L2
                "clf__C": [0.01, 0.1, 1, 10, 100],
                # desbalance 81/19: se prueba reponderar la clase minoritaria
                "clf__class_weight": [None, "balanced"],
            },
        },
        "rf": {
            "pipeline": Pipeline([("prep", preprocesador(False)),
                                  ("clf", RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=1))]),
            "rejilla": {
                "clf__max_depth": [10, 20, None],          # complejidad de cada árbol
                "clf__min_samples_leaf": [1, 3, 5],        # suavizado / control de sobreajuste
                "clf__max_features": ["sqrt", 0.5],        # decorrelación entre árboles
                "clf__class_weight": [None, "balanced_subsample"],
            },
        },
        "xgb": {
            "pipeline": Pipeline([("prep", preprocesador(False)),
                                  ("clf", XGBClassifier(colsample_bytree=0.8, tree_method="hist",
                                                        eval_metric="logloss", random_state=SEMILLA, n_jobs=1))]),
            "rejilla": {
                "clf__n_estimators": [200, 400],           # número de rondas de boosting
                "clf__max_depth": [3, 5, 7],               # interacciones de orden bajo a alto
                "clf__learning_rate": [0.05, 0.1],         # tamaño de paso (trade-off con n_estimators)
                "clf__subsample": [0.8, 1.0],              # submuestreo estocástico de filas
                "clf__scale_pos_weight": [1, 4],           # ~ razón negativos/positivos (4.3)
            },
        },
    }
    if rapido:
        modelos["logreg"]["rejilla"] = {"clf__C": [0.1, 1]}
        modelos["rf"]["rejilla"] = {"clf__max_depth": [10, None]}
        modelos["rf"]["pipeline"].set_params(clf__n_estimators=50)
        modelos["xgb"]["rejilla"] = {"clf__max_depth": [3, 5]}
        modelos["xgb"]["pipeline"].set_params(clf__n_estimators=50)
    return modelos


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def configurar_mlflow(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        artefactos = (RAIZ / "mlruns").as_uri() if uri.startswith("sqlite") or uri.startswith("file") else None
        mlflow.create_experiment(EXPERIMENTO, artifact_location=artefactos, tags={
            "proyecto": "Actividad 5 · Gestión de proyectos de IA",
            "problema": "clasificación binaria: calidad_alta (calidad >= 7)",
            "mlflow.note.content": "Comparación LogReg vs Random Forest vs XGBoost con GridSearchCV (5-fold) "
                                   "sobre Wine Quality limpio v1.1.0. Selección por ROC-AUC de validación cruzada.",
        })
    mlflow.set_experiment(EXPERIMENTO)
    return uri


def commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "sin-git"


def etiquetas_comunes(hash_datos: str) -> dict:
    return {
        "autor": AUTOR, "dataset": "wine_quality", "dataset_version": dp.VERSION_DATOS_LIMP,
        "dataset_sha256": hash_datos, "git_commit": commit_git(), "semilla": str(SEMILLA),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__, "mlflow": mlflow.__version__,
    }


def nombre_param(clave: str) -> str:
    return {"prep__num__escalar": "escalador"}.get(clave, clave.split("__")[-1])


def valor_param(v) -> str:
    return type(v).__name__ if hasattr(v, "fit") else str(v)


def metricas(y, proba, umbral: float) -> dict:
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "especificidad": tn / (tn + fp),
    }


def umbral_optimo_f1(estimador, X, y, cv) -> tuple[float, np.ndarray]:
    """Umbral que maximiza F1 sobre predicciones fuera de pliegue (OOF) del ENTRENAMIENTO."""
    oof = cross_val_predict(clone(estimador), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)]), oof


def guardar_modelo(modelo, firma, ejemplo):
    """Serializa con skops (formato seguro, por defecto en MLflow 3) declarando los tipos de
    confianza; si un estimador no es compatible, recurre a cloudpickle y lo deja etiquetado."""
    try:
        info = mlflow.sklearn.log_model(modelo, name="modelo", signature=firma, input_example=ejemplo,
                                        serialization_format="skops", skops_trusted_types=TIPOS_CONFIABLES)
        mlflow.set_tag("serializacion", "skops")
    except Exception:  # noqa: BLE001
        info = mlflow.sklearn.log_model(modelo, name="modelo", signature=firma, input_example=ejemplo,
                                        serialization_format="cloudpickle")
        mlflow.set_tag("serializacion", "cloudpickle")
    return info


def cargar_particiones():
    df = pd.read_csv(dp.ARCHIVO_LIMPIO)
    X = df[dp.VARIABLES + dp.CATEGORICAS]
    y = df[dp.OBJETIVO].astype(int)
    estrato = df["tipo"] + "_" + y.astype(str)          # conserva proporción de clase Y de tipo
    idx_tr, idx_te = train_test_split(df.index, test_size=PROPORCION_PRUEBA, stratify=estrato,
                                      random_state=SEMILLA)
    DIR_MET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_registro": df["id_registro"],
                  "conjunto": np.where(df.index.isin(idx_tr), "entrenamiento", "prueba")}
                 ).to_csv(DIR_MET / "particion_train_test.csv", index=False)
    return df, X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]


def registrar_datasets(df, X_tr, y_tr, X_te, y_te) -> None:
    fuente = str(dp.ARCHIVO_LIMPIO.relative_to(RAIZ))
    for X, y, ctx in ((X_tr, y_tr, "entrenamiento"), (X_te, y_te, "prueba")):
        ds = mlflow.data.from_pandas(X.assign(calidad_alta=y.values), source=fuente, targets="calidad_alta",
                                     name=f"wine_quality_limpio_v{dp.VERSION_DATOS_LIMP}_{ctx}")
        mlflow.log_input(ds, context=ctx)


# ---------------------------------------------------------------------------
# Entrenamiento de un modelo con GridSearchCV + registro en MLflow
# ---------------------------------------------------------------------------
def entrenar_modelo(clave, cfg, orden, datos, cv, etiquetas) -> dict:
    df, X_tr, X_te, y_tr, y_te = datos
    nombre = NOMBRES[clave]
    rejilla = cfg["rejilla"]
    n_comb = int(np.prod([len(v) for v in rejilla.values()]))
    print(f"\n[{orden:02d}] {nombre}: GridSearchCV con {n_comb} combinaciones x {K_PLIEGUES} pliegues")

    with mlflow.start_run(run_name=f"{orden:02d}_{clave}_gridsearch",
                          tags={**etiquetas, "modelo": nombre, "rol": "modelo_padre"},
                          description=f"{nombre} ajustado con GridSearchCV estratificado {K_PLIEGUES}-fold; "
                                      f"selección por {METRICA_SELECCION}.") as run:
        registrar_datasets(df, X_tr, y_tr, X_te, y_te)
        gs = GridSearchCV(cfg["pipeline"], rejilla, scoring=ESCORING, refit=METRICA_SELECCION, cv=cv,
                          n_jobs=-1, return_train_score=True, error_score="raise")
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        t_grid = time.time() - t0
        res = pd.DataFrame(gs.cv_results_)

        # --- runs hijos: una combinación de la rejilla por run -------------------
        for i, fila in res.iterrows():
            with mlflow.start_run(run_name=f"{clave}-cfg{i + 1:02d}", nested=True,
                                  tags={"rol": "combinacion_grid", "modelo": nombre}):
                mlflow.log_params({nombre_param(k): valor_param(v) for k, v in fila["params"].items()})
                mlflow.log_metrics({**{f"cv_{m}_media": fila[f"mean_test_{m}"] for m in ESCORING},
                                    "cv_roc_auc_desv": fila["std_test_roc_auc"],
                                    "train_roc_auc_media": fila["mean_train_roc_auc"],
                                    "ranking_roc_auc": int(fila["rank_test_roc_auc"])})

        mejor = gs.best_estimator_
        i_mejor = gs.best_index_
        umbral, _ = umbral_optimo_f1(mejor, X_tr, y_tr, cv)
        proba_te = mejor.predict_proba(X_te)[:, 1]
        m_umbral = metricas(y_te, proba_te, umbral)
        m_05 = metricas(y_te, proba_te, 0.5)

        mlflow.log_params({
            **{f"mejor_{nombre_param(k)}": valor_param(v) for k, v in gs.best_params_.items()},
            "algoritmo": type(mejor.named_steps["clf"]).__name__, "n_combinaciones": n_comb,
            "cv": f"StratifiedKFold(k={K_PLIEGUES}, shuffle, seed={SEMILLA})",
            "metrica_seleccion": METRICA_SELECCION, "umbral_decision": round(umbral, 4),
            "criterio_umbral": "max F1 en predicciones OOF de entrenamiento",
            "rejilla": json.dumps({nombre_param(k): [valor_param(x) for x in v] for k, v in rejilla.items()}),
        })
        cv_mejor = {f"cv_{m}_media": res.loc[i_mejor, f"mean_test_{m}"] for m in ESCORING}
        cv_mejor.update({f"cv_{m}_desv": res.loc[i_mejor, f"std_test_{m}"] for m in ESCORING})
        cv_mejor["train_roc_auc_media"] = res.loc[i_mejor, "mean_train_roc_auc"]
        cv_mejor["brecha_train_cv_roc_auc"] = cv_mejor["train_roc_auc_media"] - cv_mejor["cv_roc_auc_media"]
        mlflow.log_metrics({**cv_mejor, **{f"test_{k}": v for k, v in m_umbral.items()},
                            **{f"test_{k}_umbral05": v for k, v in m_05.items()
                               if k not in ("roc_auc", "pr_auc")},
                            "tiempo_grid_s": t_grid})

        # --- artefactos -----------------------------------------------------------
        tmp = RAIZ / "salidas" / "_tmp_mlflow"
        tmp.mkdir(parents=True, exist_ok=True)
        cols = [c for c in res.columns if c.startswith(("param_", "mean_", "std_", "rank_"))]
        res[cols].to_csv(tmp / "cv_results.csv", index=False)
        (tmp / "reporte_clasificacion.txt").write_text(
            f"{nombre} · umbral {umbral:.3f}\n\n" + classification_report(
                y_te, (proba_te >= umbral).astype(int), target_names=["estandar", "calidad_alta"], digits=3),
            encoding="utf-8")
        mlflow.log_artifacts(str(tmp), artifact_path="evaluacion")
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()

        firma = infer_signature(X_tr, mejor.predict(X_tr))
        info = guardar_modelo(mejor, firma, X_tr.head(3))
        print(f"     mejor CV ROC-AUC={gs.best_score_:.4f} · prueba ROC-AUC={m_umbral['roc_auc']:.4f} "
              f"F1={m_umbral['f1']:.4f} (umbral {umbral:.2f}) · {t_grid:.0f}s")
        return {"clave": clave, "nombre": nombre, "run_id": run.info.run_id, "model_uri": info.model_uri,
                "estimador": mejor, "cv_results": res, "mejores_params": gs.best_params_, "umbral": umbral,
                "proba_te": proba_te, "metricas_prueba": m_umbral, "metricas_cv": cv_mejor, "n_comb": n_comb,
                "tiempo_grid_s": t_grid}


def linea_base(datos, cv, etiquetas) -> dict:
    df, X_tr, X_te, y_tr, y_te = datos
    with mlflow.start_run(run_name="00_linea_base_dummy", tags={**etiquetas, "modelo": NOMBRES["dummy"],
                                                                 "rol": "linea_base"},
                          description="Clasificador trivial: siempre predice la clase mayoritaria. "
                                      "Referencia mínima que cualquier modelo debe superar.") as run:
        registrar_datasets(df, X_tr, y_tr, X_te, y_te)
        modelo = Pipeline([("prep", preprocesador(False)), ("clf", DummyClassifier(strategy="prior"))])
        cvr = cross_validate(modelo, X_tr, y_tr, cv=cv, scoring=ESCORING)
        modelo.fit(X_tr, y_tr)
        proba = modelo.predict_proba(X_te)[:, 1]
        m = metricas(y_te, proba, 0.5)
        mlflow.log_params({"algoritmo": "DummyClassifier", "strategy": "prior", "umbral_decision": 0.5})
        mlflow.log_metrics({**{f"cv_{k}_media": np.mean(cvr[f"test_{k}"]) for k in ESCORING},
                            **{f"test_{k}": v for k, v in m.items()}})
        print(f"\n[00] Línea base: accuracy prueba={m['accuracy']:.3f} · F1={m['f1']:.3f} · ROC-AUC={m['roc_auc']:.3f}")
        return {"clave": "dummy", "nombre": NOMBRES["dummy"], "run_id": run.info.run_id, "estimador": modelo,
                "umbral": 0.5, "proba_te": proba, "metricas_prueba": m,
                "metricas_cv": {f"cv_{k}_media": np.mean(cvr[f"test_{k}"]) for k in ESCORING}}


# ---------------------------------------------------------------------------
# Comparación estadística
# ---------------------------------------------------------------------------
def t_corregida_nadeau_bengio(a, b, n_train, n_test):
    """t pareada con varianza corregida por solapamiento entre pliegues (Nadeau y Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)
    j = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return float(t), float(p)


def metricas_rapidas(y, proba, umbral) -> dict:
    """Mismas métricas que `metricas`, vectorizadas con numpy (para miles de remuestreos)."""
    pred = proba >= umbral
    pos = y == 1
    tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos); tn = np.sum(~pred & ~pos)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    rangos = stats.rankdata(proba)
    n1, n0 = pos.sum(), (~pos).sum()
    auc = (rangos[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    orden = np.argsort(-proba, kind="mergesort")
    ys = pos[orden]
    tp_acum = np.cumsum(ys)
    # AP con empates agrupados (equivalente a sklearn.average_precision_score)
    ps = proba[orden]
    ult = np.r_[np.diff(ps) != 0, True]
    tp_c, k_c = tp_acum[ult], np.arange(1, len(ys) + 1)[ult]
    ap = np.sum(np.diff(np.r_[0, tp_c / n1]) * (tp_c / k_c))
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "roc_auc": auc, "pr_auc": ap,
            "balanced_accuracy": (rec + tn / (tn + fp)) / 2, "especificidad": tn / (tn + fp)}


def bootstrap_ic(y, probas: dict, umbrales: dict, n=N_BOOTSTRAP, semilla=SEMILLA):
    """IC 95 % percentil para cada métrica y cada modelo, con los MISMOS remuestreos (pareado)."""
    rng = np.random.default_rng(semilla)
    y = np.asarray(y)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    muestras = {m: [] for m in probas}
    for i in idx:
        for m, p in probas.items():
            muestras[m].append(metricas_rapidas(y[i], p[i], umbrales[m]))
    ic = {m: {k: tuple(np.percentile([s[k] for s in v], [2.5, 97.5])) for k in v[0]} for m, v in muestras.items()}
    return ic, muestras


def comparar(resultados: dict, datos, etiquetas, registrar_modelo: bool = True) -> dict:
    df, X_tr, X_te, y_tr, y_te = datos
    modelos = [k for k in ("logreg", "rf", "xgb") if k in resultados]
    nombres = [NOMBRES[k] for k in modelos]
    print("\n[04] Comparación estadística y selección del modelo campeón")

    with mlflow.start_run(run_name="04_comparacion_estadistica",
                          tags={**etiquetas, "rol": "comparacion"},
                          description="CV repetida 5x3 con pliegues idénticos para todos los modelos, "
                                      "pruebas de Friedman, t corregida y Wilcoxon (Holm), McNemar y bootstrap en prueba."):
        # 1) CV repetida con pliegues pareados
        rcv = RepeatedStratifiedKFold(n_splits=K_PLIEGUES, n_repeats=REPETICIONES_COMPARACION, random_state=SEMILLA)
        pliegues = list(rcv.split(X_tr, y_tr))
        punt = {}
        for k in modelos:
            r = cross_validate(clone(resultados[k]["estimador"]), X_tr, y_tr, cv=pliegues,
                               scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}, n_jobs=-1)
            punt[NOMBRES[k]] = r
        auc_cv = pd.DataFrame({n: punt[n]["test_roc_auc"] for n in nombres})
        prauc_cv = pd.DataFrame({n: punt[n]["test_pr_auc"] for n in nombres})
        auc_cv.to_csv(DIR_MET / "auc_cv_repetida_por_pliegue.csv", index_label="pliegue")

        fr_stat, fr_p = stats.friedmanchisquare(*[auc_cv[n] for n in nombres])
        n_tr_fold = len(y_tr) * (K_PLIEGUES - 1) / K_PLIEGUES
        n_te_fold = len(y_tr) / K_PLIEGUES
        filas = []
        for a, b in combinations(nombres, 2):
            t, p_t = t_corregida_nadeau_bengio(auc_cv[a], auc_cv[b], n_tr_fold, n_te_fold)
            w_stat, p_w = stats.wilcoxon(auc_cv[a], auc_cv[b])
            filas.append({"modelo_a": a, "modelo_b": b, "dif_media_auc": auc_cv[a].mean() - auc_cv[b].mean(),
                          "pliegues_a_gana": int((auc_cv[a] > auc_cv[b]).sum()), "t_corregida": t,
                          "p_t_corregida": p_t, "wilcoxon_W": w_stat, "p_wilcoxon": p_w})
        pruebas = pd.DataFrame(filas)
        pruebas["p_t_holm"] = multipletests(pruebas["p_t_corregida"], method="holm")[1]
        pruebas["p_wilcoxon_holm"] = multipletests(pruebas["p_wilcoxon"], method="holm")[1]

        # 2) Selección del campeón: mayor ROC-AUC media en CV (NO se usa la prueba para elegir)
        campeon = max(modelos, key=lambda k: resultados[k]["metricas_cv"]["cv_roc_auc_media"])
        n_camp = NOMBRES[campeon]

        # 3) Conjunto de prueba: bootstrap pareado + McNemar
        todos = ["dummy"] + modelos
        probas = {NOMBRES[k]: resultados[k]["proba_te"] for k in todos}
        umbrales = {NOMBRES[k]: resultados[k]["umbral"] for k in todos}
        ic, muestras = bootstrap_ic(y_te, probas, umbrales)
        mc_filas = []
        y_np = y_te.values
        for a, b in combinations(nombres, 2):
            ca = (probas[a] >= umbrales[a]).astype(int) == y_np
            cb = (probas[b] >= umbrales[b]).astype(int) == y_np
            tabla = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
            r = mcnemar(tabla, exact=True)
            dif = np.array([s_a["roc_auc"] - s_b["roc_auc"] for s_a, s_b in zip(muestras[a], muestras[b])])
            mc_filas.append({"modelo_a": a, "modelo_b": b, "a_acierta_b_falla": tabla[0][1],
                             "a_falla_b_acierta": tabla[1][0], "p_mcnemar": float(r.pvalue),
                             "dif_auc_prueba": roc_auc_score(y_np, probas[a]) - roc_auc_score(y_np, probas[b]),
                             "ic95_dif_auc_inf": float(np.percentile(dif, 2.5)),
                             "ic95_dif_auc_sup": float(np.percentile(dif, 97.5))})
        prueba_pares = pd.DataFrame(mc_filas)

        # 4) Tablas resumen
        tabla_prueba = pd.DataFrame({NOMBRES[k]: resultados[k]["metricas_prueba"] for k in todos}).T
        tabla_prueba["umbral"] = [resultados[k]["umbral"] for k in todos]
        tabla_cv = pd.DataFrame({NOMBRES[k]: {
            "roc_auc": resultados[k]["metricas_cv"]["cv_roc_auc_media"],
            "roc_auc_desv": resultados[k]["metricas_cv"].get("cv_roc_auc_desv", np.nan),
            "pr_auc": resultados[k]["metricas_cv"]["cv_pr_auc_media"],
            "f1": resultados[k]["metricas_cv"]["cv_f1_media"],
            "train_roc_auc": resultados[k]["metricas_cv"].get("train_roc_auc_media", np.nan),
            "auc_cv_repetida_media": auc_cv[NOMBRES[k]].mean() if k != "dummy" else np.nan,
            "auc_cv_repetida_desv": auc_cv[NOMBRES[k]].std(ddof=1) if k != "dummy" else np.nan,
            "pr_auc_cv_repetida_media": prauc_cv[NOMBRES[k]].mean() if k != "dummy" else np.nan,
        } for k in todos}).T
        ic_tabla = pd.DataFrame([{"modelo": m, "metrica": k, "ic95_inf": v[0], "ic95_sup": v[1]}
                                 for m, d in ic.items() for k, v in d.items()])
        tabla_prueba.to_csv(DIR_MET / "metricas_prueba.csv", index_label="modelo")
        tabla_cv.to_csv(DIR_MET / "metricas_validacion_cruzada.csv", index_label="modelo")
        ic_tabla.to_csv(DIR_MET / "ic95_bootstrap_prueba.csv", index=False)
        pruebas.to_csv(DIR_MET / "pruebas_estadisticas_cv.csv", index=False)
        prueba_pares.to_csv(DIR_MET / "pruebas_estadisticas_prueba.csv", index=False)

        # 5) Importancia por permutación (todas las variables originales, incl. tipo)
        imp = {}
        for k in modelos:
            r = permutation_importance(resultados[k]["estimador"], X_te, y_te, scoring="roc_auc", n_repeats=10,
                                       random_state=SEMILLA, n_jobs=-1)
            imp[NOMBRES[k]] = pd.Series(r.importances_mean, index=X_te.columns)
        imp = pd.DataFrame(imp)
        imp.to_csv(DIR_MET / "importancia_permutacion.csv", index_label="variable")

        # 6) Figuras
        curvas = {}
        for k in todos:
            n = NOMBRES[k]
            fpr, tpr, _ = roc_curve(y_np, probas[n])
            prec, rec, _ = precision_recall_curve(y_np, probas[n])
            curvas[n] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
                         "auc": roc_auc_score(y_np, probas[n]), "ap": average_precision_score(y_np, probas[n])}
            if k != "dummy":
                mm = resultados[k]["metricas_prueba"]
                curvas[n]["umbral_punto"] = (mm["recall"], mm["precision"])
        matrices = {NOMBRES[k]: {"cm": confusion_matrix(y_np, (resultados[k]["proba_te"] >= resultados[k]["umbral"]).astype(int)),
                                 "umbral": resultados[k]["umbral"]} for k in modelos}
        figuras = []
        try:  # requiere la rejilla completa (en modo --rapido no aplica)
            figuras.append(viz.fig_sensibilidad({NOMBRES[k]: resultados[k]["cv_results"] for k in modelos}))
        except KeyError:
            pass
        figuras += [
            viz.fig_curvas_roc_pr(curvas),
            viz.fig_matrices_confusion(matrices),
            viz.fig_comparacion(auc_cv, tabla_prueba, ic),
            viz.fig_importancia(imp),
        ]

        # 7) Registro en MLflow
        mlflow.log_params({"modelos_comparados": ", ".join(nombres), "cv_comparacion": f"RepeatedStratifiedKFold({K_PLIEGUES}x{REPETICIONES_COMPARACION})",
                           "n_bootstrap": N_BOOTSTRAP, "criterio_campeon": "max ROC-AUC media en CV (GridSearch)",
                           "modelo_campeon": n_camp})
        mlflow.log_metrics({"friedman_chi2": fr_stat, "friedman_p": fr_p,
                            **{f"auc_cv_rep_{k}": auc_cv[NOMBRES[k]].mean() for k in modelos}})
        mlflow.set_tag("modelo_campeon", n_camp)
        mlflow.log_artifacts(str(DIR_MET), artifact_path="tablas")
        for f in figuras:
            mlflow.log_artifact(str(f), artifact_path="figuras")

    # 8) Registro del campeón en el Model Registry con alias "champion"
    registro = {}
    if registrar_modelo:
        cliente = MlflowClient()
        r = resultados[campeon]
        mv = mlflow.register_model(r["model_uri"], MODELO_REGISTRADO)
        cliente.update_registered_model(MODELO_REGISTRADO, description=(
            "Clasificador de vinos de calidad alta (>=7) a partir de 11 variables fisicoquímicas + tipo. "
            "Dataset Wine Quality limpio v1.1.0."))
        cliente.update_model_version(MODELO_REGISTRADO, mv.version, description=(
            f"{n_camp} · mejores hiperparámetros {json.dumps({nombre_param(k): valor_param(v) for k, v in r['mejores_params'].items()})} "
            f"· umbral de decisión {r['umbral']:.3f}"))
        for etq, val in {"cv_roc_auc": f"{r['metricas_cv']['cv_roc_auc_media']:.4f}",
                         "test_roc_auc": f"{r['metricas_prueba']['roc_auc']:.4f}",
                         "test_f1": f"{r['metricas_prueba']['f1']:.4f}", "umbral_decision": f"{r['umbral']:.4f}",
                         "dataset_version": dp.VERSION_DATOS_LIMP, "run_id": r["run_id"]}.items():
            cliente.set_model_version_tag(MODELO_REGISTRADO, mv.version, etq, val)
        cliente.set_registered_model_alias(MODELO_REGISTRADO, "champion", mv.version)
        # Verificación de trazabilidad: recargar por alias y reproducir predicciones
        recargado = mlflow.sklearn.load_model(f"models:/{MODELO_REGISTRADO}@champion")
        identico = bool(np.allclose(recargado.predict_proba(X_te)[:, 1], r["proba_te"]))
        registro = {"nombre": MODELO_REGISTRADO, "version": int(mv.version), "alias": "champion",
                    "verificacion_recarga": identico}
        print(f"     Campeón: {n_camp} -> models:/{MODELO_REGISTRADO}@champion (v{mv.version}); "
              f"recarga reproduce predicciones: {identico}")

    print(f"     Friedman χ²={fr_stat:.2f}, p={fr_p:.4f}")
    return {"campeon": n_camp, "tabla_prueba": tabla_prueba, "tabla_cv": tabla_cv, "auc_cv": auc_cv,
            "pruebas_cv": pruebas, "pruebas_prueba": prueba_pares, "friedman": (fr_stat, fr_p), "ic": ic,
            "importancia": imp, "registro": registro, "figuras": figuras}


def exportar_runs() -> Path:
    """Exporta todos los runs del experimento a CSV (evidencia portable, sin depender de mlflow.db)."""
    runs = mlflow.search_runs(experiment_names=[EXPERIMENTO])
    cols = [c for c in runs.columns if c.startswith(("run_id", "start_time", "status", "params.", "metrics.cv_",
                                                         "metrics.test_", "tags.mlflow.runName", "tags.mlflow.parentRunId",
                                                         "tags.rol", "tags.modelo", "tags.dataset_"))]
    ruta = DIR_MET / "mlflow_runs_exportados.csv"
    runs[cols].sort_values("start_time").to_csv(ruta, index=False)
    return ruta


# ---------------------------------------------------------------------------
def main(rapido: bool = False, tracking_uri: str | None = None, registrar: bool = True) -> dict:
    np.random.seed(SEMILLA)
    if not dp.ARCHIVO_LIMPIO.exists():
        dp.ejecutar_pipeline()
    hash_datos = dp.sha256(dp.ARCHIVO_LIMPIO)
    manifest = json.loads(dp.MANIFEST.read_text(encoding="utf-8"))
    esperado = manifest["versiones"][-1]["archivos"][0]["sha256"]
    if hash_datos != esperado:
        raise ValueError("El dataset limpio no coincide con MANIFEST.json; ejecuta fuentes/datos_prep.py")

    uri = configurar_mlflow(tracking_uri)
    print(f"MLflow tracking URI: {uri}\nExperimento: {EXPERIMENTO}")
    datos = cargar_particiones()
    _, X_tr, X_te, y_tr, y_te = datos
    print(f"Entrenamiento: {len(y_tr)} ({y_tr.mean():.1%} positivos) · Prueba: {len(y_te)} ({y_te.mean():.1%} positivos)")
    cv = StratifiedKFold(n_splits=K_PLIEGUES, shuffle=True, random_state=SEMILLA)
    etiquetas = etiquetas_comunes(hash_datos)
    if rapido:
        etiquetas["modo"] = "rapido (prueba de humo)"

    resultados = {"dummy": linea_base(datos, cv, etiquetas)}
    for orden, (clave, cfg) in enumerate(definir_modelos(rapido).items(), start=1):
        resultados[clave] = entrenar_modelo(clave, cfg, orden, datos, cv, etiquetas)
    comparacion = comparar(resultados, datos, etiquetas, registrar_modelo=registrar)

    resumen = {
        "tracking_uri": uri, "experimento": EXPERIMENTO, "dataset_sha256": hash_datos,
        "n_entrenamiento": int(len(y_tr)), "n_prueba": int(len(y_te)),
        "campeon": comparacion["campeon"], "registro": comparacion["registro"],
        "modelos": {r["nombre"]: {"run_id": r["run_id"], "umbral": round(r["umbral"], 4),
                                  "mejores_params": {nombre_param(k): valor_param(v) for k, v in r.get("mejores_params", {}).items()},
                                  "n_combinaciones": r.get("n_comb"), "tiempo_grid_s": round(r.get("tiempo_grid_s", 0), 1)}
                    for r in resultados.values()},
    }
    exportar_runs()
    (DIR_MET / "resumen_ejecucion.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nResumen de métricas en prueba:")
    print(comparacion["tabla_prueba"].round(3).to_string())
    return {"resultados": resultados, "comparacion": comparacion, "resumen": resumen, "datos": datos}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="rejillas reducidas para una prueba de humo")
    ap.add_argument("--tracking-uri", default=None, help="URI de MLflow (por defecto sqlite:///mlflow.db)")
    ap.add_argument("--sin-registro", action="store_true", help="no registrar el modelo campeón")
    a = ap.parse_args()
    main(rapido=a.rapido, tracking_uri=a.tracking_uri, registrar=not a.sin_registro)
