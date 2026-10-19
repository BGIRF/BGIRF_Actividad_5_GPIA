"""
visualizaciones.py
------------------
Gráficas de calidad de datos (EDA) y de evaluación de modelos.
Todas se guardan en `salidas/figuras/` (PNG, 200 dpi) y train.py las registra
además como artefactos en MLflow.

Paleta categórica validada para daltonismo (orden fijo, nunca ciclado):
    modelos  -> azul / naranja / aqua (+ gris para la línea base)
    tipo     -> tinto rojo / blanco ámbar
    clase    -> estándar gris / calidad alta violeta
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

try:
    from datos_prep import LIMITE_SO2_UE, VARIABLES
except ImportError:  # pragma: no cover
    from fuentes.datos_prep import LIMITE_SO2_UE, VARIABLES

RAIZ = Path(__file__).resolve().parents[1]
DIR_FIG = RAIZ / "salidas" / "figuras"

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8f8e89"
REJILLA = "#e4e3df"
FONDO = "#ffffff"

COLOR_MODELO = {
    "Regresión logística": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
    "Línea base (Dummy)": "#8f8e89",
}
COLOR_TIPO = {"tinto": "#e34948", "blanco": "#eda100"}
COLOR_CLASE = {0: "#b5b4ad", 1: "#6250d6"}
ETIQUETA_CLASE = {0: "Estándar (≤6)", 1: "Calidad alta (≥7)"}
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32a2a"])
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

NOMBRE_LEGIBLE = {
    "acidez_fija": "Acidez fija", "acidez_volatil": "Acidez volátil", "acido_citrico": "Ácido cítrico",
    "azucar_residual": "Azúcar residual", "cloruros": "Cloruros", "so2_libre": "SO₂ libre",
    "so2_total": "SO₂ total", "densidad": "Densidad", "ph": "pH", "sulfatos": "Sulfatos",
    "alcohol": "Alcohol", "calidad": "Calidad", "tipo": "Tipo",
}


def estilo() -> None:
    """Estilo sobrio: rejilla tenue, sin bordes superiores/derechos, texto en tinta neutra."""
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": TINTA_2,
        "axes.edgecolor": TINTA_3, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "text.color": TINTA, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2,
    })


def guardar(fig, nombre: str) -> Path:
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIG / nombre
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# EDA y calidad de datos
# ---------------------------------------------------------------------------
def fig_calidad_y_duplicados(df: pd.DataFrame, reporte: dict) -> Path:
    estilo()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
    puntajes = list(range(3, 10))
    ancho = 0.4
    a.axvspan(6.5, 9.5, color="#efedfb", zorder=0)
    a.text(8.0, 0.97, "Calidad alta (≥7)", transform=a.get_xaxis_transform(), ha="center", va="top",
           color="#4a3aa7", fontsize=8, fontweight="bold")
    for i, tipo in enumerate(["tinto", "blanco"]):
        conteo = df[df["tipo"] == tipo]["calidad"].value_counts().reindex(puntajes, fill_value=0)
        x = np.array(puntajes) + (i - 0.5) * ancho
        a.bar(x, conteo.values, width=ancho - 0.04, color=COLOR_TIPO[tipo], label=tipo.capitalize())
    a.set_xticks(puntajes)
    a.set_xlabel("Puntaje sensorial de calidad")
    a.set_ylabel("Registros (datos limpios)")
    a.set_title("(a) Distribución de la calidad por tipo de vino")
    a.legend(loc="upper left")
    a.grid(axis="x", visible=False)

    tipos = ["tinto", "blanco"]
    finales = [reporte["filas_finales_por_tipo"][t] for t in tipos]
    dups = [reporte["duplicados"]["duplicados_por_tipo"].get(t, 0) for t in tipos]
    y = np.arange(len(tipos))
    b.barh(y, finales, color=[COLOR_TIPO[t] for t in tipos], height=0.55)
    b.barh(y, dups, left=np.array(finales) + 25, color="#d9d8d3", height=0.55, hatch="////",
           edgecolor="#a8a7a1", linewidth=0)
    for yi, f, d in zip(y, finales, dups):
        b.text(f / 2, yi, f"{f:,}", ha="center", va="center", color=TINTA, fontsize=8.5, fontweight="bold")
        b.text(f + 25 + d + 60, yi, f"{d:,} eliminados\n({d / (f + d):.0%} del original)", ha="left",
               va="center", color=TINTA_2, fontsize=8)
    b.set_yticks(y, [t.capitalize() for t in tipos])
    b.set_xlim(0, max(f + d for f, d in zip(finales, dups)) * 1.45)
    b.set_xlabel("Registros originales")
    b.set_title("(b) Duplicados exactos eliminados")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TINTA_3),
                      plt.Rectangle((0, 0), 1, 1, facecolor="#d9d8d3", hatch="////", edgecolor="#a8a7a1")],
             labels=["Conservados", "Duplicados eliminados"], loc="center right", fontsize=7.5)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    return guardar(fig, "fig01_calidad_y_duplicados.png")


def fig_correlaciones(df: pd.DataFrame) -> Path:
    estilo()
    cols = VARIABLES + ["calidad"]
    corr = df[cols].corr(method="spearman")
    # triángulo inferior SIN diagonal: filas 2..n, columnas 1..n-1
    filas, columnas = cols[1:], cols[:-1]
    sub = corr.loc[filas, columnas].values
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(np.ma.array(sub, mask=mask), cmap=CMAP_DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columnas)), [NOMBRE_LEGIBLE[c] for c in columnas], rotation=45, ha="right")
    ax.set_yticks(range(len(filas)), [NOMBRE_LEGIBLE[c] for c in filas])
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i in range(len(filas)):
        for j in range(i + 1):
            v = sub[i, j]
            txt = f"{v:.2f}".replace("-0.", "−.").replace("0.", ".")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else TINTA,
                    fontweight="bold" if filas[i] == "calidad" else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ρ de Spearman", color=TINTA_2)
    cb.outline.set_visible(False)
    ax.set_title("Correlación de Spearman entre variables y calidad")
    fig.tight_layout()
    return guardar(fig, "fig02_correlaciones.png")


def fig_distribuciones_por_clase(df: pd.DataFrame) -> Path:
    estilo()
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for ax, col in zip(axes.flat, VARIABLES):
        datos = [df.loc[df["calidad_alta"] == c, col].values for c in (0, 1)]
        bp = ax.boxplot(datos, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=True,
                        medianprops={"color": TINTA, "linewidth": 1.4},
                        whiskerprops={"color": TINTA_3, "linewidth": 0.9},
                        capprops={"color": TINTA_3, "linewidth": 0.9},
                        flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": TINTA_3,
                                    "markeredgecolor": "none", "alpha": 0.6})
        for patch, c in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(COLOR_CLASE[c])
            patch.set_edgecolor(FONDO)
        ax.set_xticks([0, 1], ["Estándar", "Alta"])
        ax.set_title(NOMBRE_LEGIBLE[col], fontsize=9)
        ax.grid(axis="x", visible=False)
        if col in ("azucar_residual", "cloruros", "so2_libre"):
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(NOMBRE_LEGIBLE[col] + " (log)", fontsize=9)
    leg = axes.flat[-1]
    leg.axis("off")
    for c, yy in ((0, 0.62), (1, 0.42)):
        leg.add_patch(plt.Rectangle((0.05, yy), 0.12, 0.1, color=COLOR_CLASE[c], transform=leg.transAxes))
        leg.text(0.22, yy + 0.05, ETIQUETA_CLASE[c], transform=leg.transAxes, va="center", fontsize=9)
    leg.text(0.05, 0.22, "Puntos: valores fuera de\n1.5·IQR (posibles atípicos)", transform=leg.transAxes,
             fontsize=8, color=TINTA_2, va="center")
    fig.suptitle("Distribución de cada variable fisicoquímica por clase", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return guardar(fig, "fig03_distribuciones_por_clase.png")


def fig_atipicos_y_regulatorio(df: pd.DataFrame, reporte: dict) -> Path:
    estilo()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    conteo = pd.Series(reporte["atipicos"]["atipicos_por_variable"]).sort_values()
    a.barh([NOMBRE_LEGIBLE[c] for c in conteo.index], conteo.values, color="#2a78d6", height=0.6)
    for i, v in enumerate(conteo.values):
        a.text(v + 2, i, str(v), va="center", fontsize=7.5, color=TINTA_2)
    a.set_xlabel("Registros con valor extremo (fuera de Q1−3·IQR / Q3+3·IQR, por tipo)")
    a.set_title("(a) Atípicos extremos por variable")
    a.grid(axis="y", visible=False)

    for tipo in ("blanco", "tinto"):
        d = df[df["tipo"] == tipo]
        exc = d["excede_so2_ue"] == 1
        b.scatter(d.loc[~exc, "azucar_residual"], d.loc[~exc, "so2_total"], s=6, color=COLOR_TIPO[tipo],
                  alpha=0.35, linewidths=0, label=f"{tipo.capitalize()}")
        b.scatter(d.loc[exc, "azucar_residual"], d.loc[exc, "so2_total"], s=22, facecolor=COLOR_TIPO[tipo],
                  edgecolor=TINTA, linewidths=0.8, label=f"{tipo.capitalize()} > límite UE ({exc.sum()})")
        x = [0.5, 5, 5, 70]
        lim_bajo, lim_alto = LIMITE_SO2_UE[(tipo, False)], LIMITE_SO2_UE[(tipo, True)]
        b.plot(x, [lim_bajo, lim_bajo, lim_alto, lim_alto], color=COLOR_TIPO[tipo], linewidth=1.3,
               linestyle="--")
        b.text(70, lim_alto + 6, f"Límite {tipo} {lim_alto}", ha="right", fontsize=7, color=TINTA_2)
    b.set_xscale("log")
    b.set_xticks([0.6, 1, 2, 5, 10, 20, 50], ["0.6", "1", "2", "5", "10", "20", "50"])
    b.minorticks_off()
    b.set_xlabel("Azúcar residual (g/L, escala log)")
    b.set_ylabel("SO₂ total (mg/L)")
    b.set_title("(b) SO₂ total vs. límite del Reg. (UE) 2019/934")
    b.legend(loc="upper left", fontsize=7, markerscale=1.2)
    fig.tight_layout()
    return guardar(fig, "fig04_atipicos_y_regulatorio.png")


# ---------------------------------------------------------------------------
# Evaluación de modelos
# ---------------------------------------------------------------------------
def fig_sensibilidad(cv_resultados: dict) -> Path:
    """Mejor ROC-AUC de validación cruzada al variar el hiperparámetro principal de cada modelo."""
    estilo()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    # Regresión logística: C por escalador
    r = cv_resultados["Regresión logística"]
    ax = axes[0]
    estilos = {"StandardScaler()": ("-", "Estandarización z"), "PowerTransformer()": ("--", "Yeo-Johnson")}
    for esc, (ls, lab) in estilos.items():
        sub = r[r["param_prep__num__escalar"].astype(str) == esc]
        g = sub.groupby("param_clf__C")["mean_test_roc_auc"].max()
        ax.plot(g.index, g.values, ls, marker="o", markersize=4, color=COLOR_MODELO["Regresión logística"],
                label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverso de la regularización)")
    ax.set_ylabel("ROC-AUC media (CV 5-fold)")
    ax.set_title("Regresión logística")
    ax.legend(loc="upper left")
    # Random Forest: max_depth x min_samples_leaf
    r = cv_resultados["Random Forest"].copy()
    ax = axes[1]
    r["prof"] = r["param_clf__max_depth"].astype(object).where(r["param_clf__max_depth"].notna(), "Sin límite").astype(str)
    for hoja, ls in zip(sorted(r["param_clf__min_samples_leaf"].unique()), ["-", "--", ":"]):
        g = r[r["param_clf__min_samples_leaf"] == hoja].groupby("prof")["mean_test_roc_auc"].max()
        orden = [o for o in ["10", "20", "Sin límite"] if o in g.index]
        ax.plot(orden, g.reindex(orden).values, ls, marker="o", markersize=4,
                color=COLOR_MODELO["Random Forest"], label=f"min_samples_leaf={hoja}")
    ax.set_xlabel("max_depth")
    ax.set_title("Random Forest")
    ax.legend(loc="lower right")
    # XGBoost: max_depth x learning_rate
    r = cv_resultados["XGBoost"]
    ax = axes[2]
    for lr, ls in zip(sorted(r["param_clf__learning_rate"].unique()), ["-", "--"]):
        g = r[r["param_clf__learning_rate"] == lr].groupby("param_clf__max_depth")["mean_test_roc_auc"].max()
        ax.plot(g.index.astype(int), g.values, ls, marker="o", markersize=4, color=COLOR_MODELO["XGBoost"],
                label=f"learning_rate={lr}")
    ax.set_xlabel("max_depth")
    ax.set_title("XGBoost")
    ax.legend(loc="lower right")
    lo = min(ax_.get_ylim()[0] for ax_ in axes)
    hi = max(ax_.get_ylim()[1] for ax_ in axes)
    for ax_ in axes:
        ax_.set_ylim(lo, hi)
    fig.suptitle("Sensibilidad a hiperparámetros (máximo sobre el resto de la rejilla)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return guardar(fig, "fig05_sensibilidad_hiperparametros.png")


def fig_curvas_roc_pr(curvas: dict) -> Path:
    estilo()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for nombre, c in curvas.items():
        ls = "--" if "Dummy" in nombre else "-"
        lw = 1.3 if "Dummy" in nombre else 2
        a.plot(c["fpr"], c["tpr"], ls, lw=lw, color=COLOR_MODELO[nombre], label=f"{nombre} · AUC {c['auc']:.3f}")
        if "Dummy" in nombre:  # puntaje constante: la curva PR es la prevalencia (línea horizontal)
            b.plot([0, 1], [c["ap"], c["ap"]], ls, lw=lw, color=COLOR_MODELO[nombre],
                   label=f"{nombre} · AP {c['ap']:.3f} (= prevalencia)")
        else:
            b.plot(c["rec"], c["prec"], ls, lw=lw, color=COLOR_MODELO[nombre], label=f"{nombre} · AP {c['ap']:.3f}")
        if "umbral_punto" in c:
            b.plot(*c["umbral_punto"], "o", ms=7, color=COLOR_MODELO[nombre], mec=FONDO, mew=1.5)
    a.plot([0, 1], [0, 1], color=REJILLA, lw=1)
    a.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    a.set_ylabel("Tasa de verdaderos positivos (recall)")
    a.set_title("(a) Curva ROC · conjunto de prueba")
    a.legend(loc="lower right")
    b.set_xlabel("Recall")
    b.set_ylabel("Precisión")
    b.set_title("(b) Curva precisión-recall · prueba (● = umbral elegido)")
    b.legend(loc="upper right")
    b.set_ylim(0, 1.02)
    fig.tight_layout()
    return guardar(fig, "fig06_curvas_roc_pr.png")


def fig_matrices_confusion(matrices: dict) -> Path:
    estilo()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.1))
    vmax = max(m["cm"].max() for m in matrices.values())
    for ax, (nombre, m) in zip(np.atleast_1d(axes), matrices.items()):
        cm = m["cm"]
        ax.imshow(cm, cmap=CMAP_SECUENCIAL, vmin=0, vmax=vmax)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        tot = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / tot[i, 0]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > vmax * 0.55 else TINTA)
        ax.set_xticks([0, 1], ["Estándar", "Alta"])
        ax.set_yticks([0, 1], ["Estándar", "Alta"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{nombre}\numbral = {m['umbral']:.2f}", fontsize=9)
    fig.tight_layout()
    return guardar(fig, "fig07_matrices_confusion.png")


def fig_comparacion(puntajes_cv: pd.DataFrame, tabla_prueba: pd.DataFrame, ic: dict) -> Path:
    """(a) ROC-AUC por pliegue en CV repetida (15 pliegues pareados); (b) métricas de prueba con IC 95 %."""
    estilo()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    modelos = [m for m in COLOR_MODELO if m in puntajes_cv.columns]
    # líneas tenues que unen el mismo pliegue (diseño pareado); mismo desplazamiento por pliegue
    jit = np.linspace(-0.09, 0.09, len(puntajes_cv))
    for f, (_, fila) in enumerate(puntajes_cv[modelos].iterrows()):
        a.plot(np.arange(len(modelos)) + jit[f], fila.values, color=REJILLA, lw=0.8, zorder=1)
    for i, m in enumerate(modelos):
        v = puntajes_cv[m].values
        a.scatter(i + jit, v, s=14, color=COLOR_MODELO[m], zorder=2, linewidths=0)
        a.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=TINTA, lw=1.6, zorder=3)
        a.text(i + 0.28, v.mean(), f"{v.mean():.3f}", va="center", fontsize=8, color=TINTA)
    a.set_xticks(range(len(modelos)), [m.replace("Regresión logística", "Reg. logística") for m in modelos])
    a.set_xlim(-0.5, len(modelos) - 0.3)
    a.set_ylabel("ROC-AUC por pliegue")
    a.set_title("(a) CV estratificada 5×3 (pliegues pareados)")
    a.grid(axis="x", visible=False)

    metricas = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    etiquetas = ["Accuracy", "Precisión", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ancho = 0.8 / len(modelos)
    x = np.arange(len(metricas))
    for i, m in enumerate(modelos):
        vals = tabla_prueba.loc[m, metricas].values.astype(float)
        err = np.array([[vals[k] - ic[m][met][0], ic[m][met][1] - vals[k]] for k, met in enumerate(metricas)]).T
        b.bar(x + (i - (len(modelos) - 1) / 2) * ancho, vals, width=ancho - 0.03, color=COLOR_MODELO[m],
              label=m, yerr=err, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TINTA_2})
    b.set_xticks(x, etiquetas)
    b.set_ylim(0, 1.05)
    b.set_ylabel("Valor en prueba (IC 95 % bootstrap)")
    b.set_title("(b) Métricas en el conjunto de prueba")
    b.legend(loc="upper left", ncol=3, fontsize=7.5, bbox_to_anchor=(0, 1.0))
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    return guardar(fig, "fig08_comparacion_modelos.png")


def fig_importancia(importancias: pd.DataFrame) -> Path:
    """Importancia por permutación (caída de ROC-AUC en prueba), comparable entre modelos."""
    estilo()
    orden = importancias.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(orden))
    desplaz = {m: d for m, d in zip(importancias.columns, np.linspace(0.22, -0.22, len(importancias.columns)))}
    for m in importancias.columns:
        v = importancias.loc[orden, m].values
        ax.hlines(y + desplaz[m], 0, v, color=COLOR_MODELO[m], lw=1.2, alpha=0.6)
        ax.plot(v, y + desplaz[m], "o", ms=5.5, color=COLOR_MODELO[m], mec=FONDO, mew=1, label=m)
    ax.axvline(0, color=TINTA_3, lw=0.8)
    ax.set_yticks(y, [NOMBRE_LEGIBLE.get(c, c) for c in orden])
    ax.set_xlabel("Caída media de ROC-AUC al permutar la variable (10 repeticiones)")
    ax.set_title("Importancia por permutación en el conjunto de prueba")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return guardar(fig, "fig09_importancia_permutacion.png")
