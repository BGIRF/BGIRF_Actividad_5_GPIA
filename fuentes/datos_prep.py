"""
datos_prep.py
-------------
Funciones de limpieza, estandarización y versionamiento del dataset
Wine Quality (vinho verde tinto y blanco; Cortez et al., 2009).

Flujo (`ejecutar_pipeline`):
    1. cargar_datos_originales   -> une tinto + blanco y asigna id de linaje
    2. estandarizar_columnas     -> nombres snake_case en español
    3. convertir_tipos           -> tipos numéricos/categóricos explícitos
    4. manejar_nulos             -> reglas para nulos (objetivo vs. variables)
    5. eliminar_duplicados       -> duplicados exactos (criterio documentado)
    6. aplicar_reglas_negocio    -> validez físico-química + bandera regulatoria SO2 (UE)
    7. marcar_atipicos           -> atípicos extremos por IQR (se marcan, no se borran)
    8. crear_objetivo            -> calidad_alta = 1 si calidad >= 7
    9. guardar + MANIFEST.json   -> datos/datos_limp/ y huella SHA-256 (versión 1.1.0)

El ESCALADO y la CODIFICACIÓN de `tipo` NO se hacen aquí: se ajustan dentro del
Pipeline de scikit-learn (train.py) solo con datos de entrenamiento, para evitar
fuga de información (data leakage) hacia validación y prueba.

Uso:
    python fuentes/datos_prep.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:  # ejecución como script o como módulo importado desde el notebook
    from obtener_datos import ARCHIVOS, sha256, verificar_datos
except ImportError:  # pragma: no cover
    from fuentes.obtener_datos import ARCHIVOS, sha256, verificar_datos

RAIZ = Path(__file__).resolve().parents[1]
DIR_INI = RAIZ / "datos" / "datos_ini"
DIR_LIMP = RAIZ / "datos" / "datos_limp"
ARCHIVO_LIMPIO = DIR_LIMP / "wine_quality_limpio.csv"
ARCHIVO_REPORTE = DIR_LIMP / "reporte_limpieza.json"
MANIFEST = RAIZ / "datos" / "MANIFEST.json"

VERSION_DATOS_INI = "1.0.0"
VERSION_DATOS_LIMP = "1.1.0"

# ---------------------------------------------------------------------------
# Diccionario de datos (nombre original -> nombre estandarizado, unidad)
# ---------------------------------------------------------------------------
DICCIONARIO = {
    "fixed acidity":        ("acidez_fija",     "g ácido tartárico/L",  "Ácidos no volátiles (tartárico, málico)"),
    "volatile acidity":     ("acidez_volatil",  "g ácido acético/L",    "Ácido acético; en exceso da sabor a vinagre"),
    "citric acid":          ("acido_citrico",   "g/L",                  "Aporta frescura; en baja proporción"),
    "residual sugar":       ("azucar_residual", "g/L",                  "Azúcar remanente tras la fermentación"),
    "chlorides":            ("cloruros",        "g cloruro de sodio/L", "Contenido de sal"),
    "free sulfur dioxide":  ("so2_libre",       "mg/L",                 "SO2 libre (antimicrobiano/antioxidante)"),
    "total sulfur dioxide": ("so2_total",       "mg/L",                 "SO2 libre + combinado"),
    "density":              ("densidad",        "g/cm³",                "Depende de alcohol y azúcar"),
    "pH":                   ("ph",              "-",                    "Acidez (escala 0-14)"),
    "sulphates":            ("sulfatos",        "g sulfato de potasio/L", "Aditivo que contribuye al SO2"),
    "alcohol":              ("alcohol",         "% vol",                "Grado alcohólico"),
    "quality":              ("calidad",         "puntaje 0-10",         "Mediana de >=3 evaluaciones sensoriales"),
}
RENOMBRAR = {k: v[0] for k, v in DICCIONARIO.items()}
VARIABLES = [v[0] for k, v in DICCIONARIO.items() if k != "quality"]   # 11 fisicoquímicas
CATEGORICAS = ["tipo"]
OBJETIVO = "calidad_alta"
UMBRAL_CALIDAD_ALTA = 7

# Reglas de validez físico-química (fuera de rango = registro imposible -> se elimina)
RANGOS_VALIDOS = {
    "ph": (2.5, 4.5),
    "densidad": (0.98, 1.05),
    "alcohol": (7.0, 16.0),
}
# Límite máximo de SO2 total en vinos tranquilos, Reglamento Delegado (UE) 2019/934,
# Anexo I, Parte B (mg/L). Si glucosa+fructosa >= 5 g/L el límite sube.
LIMITE_SO2_UE = {("tinto", False): 150, ("tinto", True): 200,
                 ("blanco", False): 200, ("blanco", True): 250}
K_IQR_EXTREMO = 3.0


# ---------------------------------------------------------------------------
# Pasos del pipeline
# ---------------------------------------------------------------------------
def cargar_datos_originales() -> pd.DataFrame:
    """Lee los CSV originales (separador ';') y conserva el linaje de cada registro."""
    partes = []
    for archivo, tipo, prefijo in [("winequality-red.csv", "tinto", "R"),
                                   ("winequality-white.csv", "blanco", "W")]:
        df = pd.read_csv(DIR_INI / archivo, sep=";")
        df.insert(0, "id_registro", [f"{prefijo}{i:04d}" for i in range(1, len(df) + 1)])
        df.insert(1, "tipo", tipo)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Nombres en snake_case y en español (consistentes en código, reporte y MLflow)."""
    df = df.rename(columns=lambda c: c.strip())
    return df.rename(columns=RENOMBRAR)


def convertir_tipos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fuerza tipos explícitos. Valores no numéricos se vuelven NaN y se reportan."""
    df = df.copy()
    no_numericos = {}
    for col in VARIABLES + ["calidad"]:
        antes = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        no_numericos[col] = int(df[col].isna().sum() - antes)
    df["tipo"] = df["tipo"].astype("string").str.strip().str.lower()
    return df, no_numericos


def manejar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Regla: sin objetivo no hay ejemplo de entrenamiento -> se elimina la fila.
    Nulos en variables explicativas se CONSERVAN y se imputan con la mediana
    dentro del Pipeline (ajustada solo con entrenamiento).
    """
    nulos_por_columna = {c: int(v) for c, v in df.isna().sum().items()}
    sin_objetivo = int(df["calidad"].isna().sum())
    df = df.dropna(subset=["calidad"]).copy()
    df["calidad"] = df["calidad"].astype("int64")
    return df, {"nulos_por_columna": nulos_por_columna, "filas_sin_objetivo_eliminadas": sin_objetivo}


def eliminar_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Criterio: dos registros son duplicados si coinciden EXACTAMENTE en tipo,
    las 11 variables fisicoquímicas y la calidad. Se conserva la primera aparición.
    Motivo: un duplicado puede caer en entrenamiento y en validación a la vez,
    inflando artificialmente las métricas (fuga entre particiones).
    """
    clave = ["tipo"] + VARIABLES + ["calidad"]
    es_dup = df.duplicated(subset=clave, keep="first")
    # ¿Hay registros con las mismas variables pero distinta calidad? (etiquetas en conflicto)
    conflicto = df.drop_duplicates(subset=clave).duplicated(subset=["tipo"] + VARIABLES, keep=False)
    # Coincidencias entre tipos distintos (mismas variables, distinto tipo) -> se conservan
    entre_tipos = int(df.drop_duplicates(subset=clave).duplicated(subset=VARIABLES + ["calidad"]).sum())
    reporte = {
        "criterio": "coincidencia exacta en tipo + 11 variables + calidad; se conserva la primera",
        "duplicados_eliminados": int(es_dup.sum()),
        "duplicados_por_tipo": {t: int(v) for t, v in df.loc[es_dup, "tipo"].value_counts().items()},
        "etiquetas_en_conflicto": int(conflicto.sum()),
        "coincidencias_entre_tipos_conservadas": entre_tipos,
    }
    return df.loc[~es_dup].copy(), reporte


def aplicar_reglas_negocio(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    R1. Concentraciones no negativas.               -> eliminar si se viola
    R2. so2_libre <= so2_total.                       -> eliminar si se viola
    R3. pH, densidad y alcohol en rangos plausibles.  -> eliminar si se viola
    R4. Calidad entera en [0, 10].                    -> eliminar si se viola
    R5. SO2 total por encima del límite UE (Reg. 2019/934) -> SOLO bandera
        `excede_so2_ue` (trazabilidad regulatoria; el modelo predice calidad
        sensorial y las muestras son de 2004-2007).
    """
    df = df.copy()
    r1 = (df[VARIABLES] < 0).any(axis=1)
    r2 = df["so2_libre"] > df["so2_total"]
    r3 = pd.Series(False, index=df.index)
    for col, (lo, hi) in RANGOS_VALIDOS.items():
        r3 |= ~df[col].between(lo, hi) & df[col].notna()
    r4 = ~df["calidad"].between(0, 10)
    invalido = r1 | r2 | r3 | r4

    dulce = df["azucar_residual"] >= 5
    limite = [LIMITE_SO2_UE[(t, bool(d))] for t, d in zip(df["tipo"], dulce)]
    df["excede_so2_ue"] = (df["so2_total"] > np.array(limite)).astype("int64")

    reporte = {
        "R1_negativos": int(r1.sum()), "R2_so2_libre_mayor_total": int(r2.sum()),
        "R3_fuera_de_rango": int(r3.sum()), "R4_calidad_invalida": int(r4.sum()),
        "filas_eliminadas_por_reglas": int(invalido.sum()),
        "R5_excede_so2_ue_marcadas": int(df.loc[~invalido, "excede_so2_ue"].sum()),
        "R5_por_tipo": {t: int(v) for t, v in df.loc[~invalido].groupby("tipo")["excede_so2_ue"].sum().items()},
    }
    return df.loc[~invalido].copy(), reporte


def marcar_atipicos(df: pd.DataFrame, k: float = K_IQR_EXTREMO) -> tuple[pd.DataFrame, dict]:
    """
    Atípico extremo: valor fuera de [Q1 - k·IQR, Q3 + k·IQR] calculado POR TIPO de vino
    (tinto y blanco tienen perfiles químicos distintos). Se marcan pero NO se eliminan:
    son vinos reales (p. ej., blancos dulces) y los modelos de árboles son robustos;
    para la regresión logística se prueba una transformación Yeo-Johnson en el grid.
    """
    df = df.copy()
    conteo = pd.Series(0, index=df.index, dtype="int64")
    por_variable = {}
    for col in VARIABLES:
        q1 = df.groupby("tipo")[col].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby("tipo")[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        fuera = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
        conteo += fuera.astype("int64")
        por_variable[col] = int(fuera.sum())
    df["n_atipicos_extremos"] = conteo
    reporte = {"k_iqr": k, "registros_con_algun_atipico": int((conteo > 0).sum()),
               "atipicos_por_variable": por_variable}
    return df, reporte


def crear_objetivo(df: pd.DataFrame, umbral: int = UMBRAL_CALIDAD_ALTA) -> pd.DataFrame:
    """Variable objetivo binaria: 1 = calidad alta (>= 7), 0 = estándar (<= 6)."""
    df = df.copy()
    df[OBJETIVO] = (df["calidad"] >= umbral).astype("int64")
    return df


def actualizar_manifest(reporte: dict, hash_limpio: str) -> dict:
    """Registro de versiones del dato (SemVer + SHA-256 + linaje)."""
    manifest = {
        "dataset": "Wine Quality (vinho verde) - Cortez et al. (2009), UCI ML Repository id 186",
        "licencia": "CC BY 4.0",
        "versiones": [
            {
                "version": VERSION_DATOS_INI,
                "descripcion": "Archivos originales sin modificar",
                "archivos": [{"ruta": f"datos/datos_ini/{n}", "sha256": m["sha256"], "filas": m["filas"]}
                             for n, m in ARCHIVOS.items()],
            },
            {
                "version": VERSION_DATOS_LIMP,
                "descripcion": "Unión tinto+blanco, nombres estandarizados, sin duplicados, reglas de negocio, objetivo binario",
                "archivos": [{"ruta": "datos/datos_limp/wine_quality_limpio.csv", "sha256": hash_limpio,
                              "filas": reporte["filas_finales"]}],
                "derivado_de": VERSION_DATOS_INI,
                "generado_por": "fuentes/datos_prep.py",
            },
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def ejecutar_pipeline(guardar: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Ejecuta la limpieza completa y devuelve (df_limpio, reporte)."""
    verificar_datos()
    df = cargar_datos_originales()
    reporte = {"version_datos": VERSION_DATOS_LIMP, "filas_originales": len(df),
               "filas_originales_por_tipo": {t: int(v) for t, v in df["tipo"].value_counts().items()}}

    df = estandarizar_columnas(df)
    df, reporte["valores_no_numericos"] = convertir_tipos(df)
    df, reporte["nulos"] = manejar_nulos(df)
    df, reporte["duplicados"] = eliminar_duplicados(df)
    df, reporte["reglas_negocio"] = aplicar_reglas_negocio(df)
    df, reporte["atipicos"] = marcar_atipicos(df)
    df = crear_objetivo(df)

    columnas = ["id_registro", "tipo"] + VARIABLES + ["calidad", OBJETIVO, "excede_so2_ue", "n_atipicos_extremos"]
    df = df[columnas].reset_index(drop=True)
    reporte["filas_finales"] = len(df)
    reporte["filas_finales_por_tipo"] = {t: int(v) for t, v in df["tipo"].value_counts().items()}
    reporte["prevalencia_calidad_alta"] = round(float(df[OBJETIVO].mean()), 4)
    reporte["prevalencia_por_tipo"] = {t: round(float(v), 4) for t, v in df.groupby("tipo")[OBJETIVO].mean().items()}

    if guardar:
        DIR_LIMP.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARCHIVO_LIMPIO, index=False, float_format="%.6g", lineterminator="\n")
        hash_limpio = sha256(ARCHIVO_LIMPIO)
        reporte["sha256_limpio"] = hash_limpio
        reporte["generado_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ARCHIVO_REPORTE.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
        actualizar_manifest(reporte, hash_limpio)

    if verbose:
        d, r = reporte["duplicados"], reporte["reglas_negocio"]
        print(f"Filas originales: {reporte['filas_originales']}  ->  finales: {reporte['filas_finales']}")
        print(f"Duplicados eliminados: {d['duplicados_eliminados']} {d['duplicados_por_tipo']}")
        print(f"Filas eliminadas por reglas de validez: {r['filas_eliminadas_por_reglas']}")
        print(f"Marcadas SO2 > límite UE: {r['R5_excede_so2_ue_marcadas']} {r['R5_por_tipo']}")
        print(f"Registros con algún atípico extremo: {reporte['atipicos']['registros_con_algun_atipico']}")
        print(f"Prevalencia calidad alta: {reporte['prevalencia_calidad_alta']:.1%}")
    return df, reporte


def diccionario_de_datos() -> pd.DataFrame:
    """Tabla del diccionario de datos (para README/reporte)."""
    filas = [{"variable": v[0], "nombre_original": k, "unidad": v[1], "descripcion": v[2]}
             for k, v in DICCIONARIO.items()]
    filas.insert(0, {"variable": "tipo", "nombre_original": "(archivo de origen)", "unidad": "categoría",
                     "descripcion": "tinto / blanco"})
    filas.append({"variable": OBJETIVO, "nombre_original": "(derivada)", "unidad": "0/1",
                  "descripcion": f"1 si calidad >= {UMBRAL_CALIDAD_ALTA}"})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ejecutar_pipeline()
