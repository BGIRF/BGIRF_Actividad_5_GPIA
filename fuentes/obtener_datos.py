"""
obtener_datos.py
----------------
Obtiene y verifica los archivos ORIGINALES del dataset Wine Quality
(Cortez et al., 2009 · UCI Machine Learning Repository, id 186).

Los archivos ya están versionados en `datos/datos_ini/`. Este script sirve para
reconstruirlos desde cero y, sobre todo, para comprobar su integridad mediante
SHA-256 antes de entrenar (si el hash no coincide, el pipeline se detiene).

Uso:
    python fuentes/obtener_datos.py            # verifica (y descarga si faltan)
    python fuentes/obtener_datos.py --forzar   # vuelve a descargar
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_INI = RAIZ / "datos" / "datos_ini"

# Fuente primaria (UCI) y espejo público en GitHub con los archivos idénticos.
ARCHIVOS = {
    "winequality-red.csv": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-red.csv",
        ],
        "sha256": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
        "filas": 1599,
    },
    "winequality-white.csv": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-white.csv",
        ],
        "sha256": "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
        "filas": 4898,
    },
}


def sha256(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo (huella de versión del dato)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(nombre: str, destino: Path) -> None:
    ultimo_error = None
    for url in ARCHIVOS[nombre]["urls"]:
        try:
            print(f"  Descargando {nombre} desde {url}")
            urllib.request.urlretrieve(url, destino)
            return
        except Exception as e:  # noqa: BLE001 - se intenta la siguiente fuente
            ultimo_error = e
            print(f"    No disponible ({e.__class__.__name__}); probando otra fuente...")
    raise RuntimeError(f"No se pudo descargar {nombre}: {ultimo_error}")


def verificar_datos(forzar: bool = False) -> dict:
    """Garantiza que los archivos originales existen y que su hash es el esperado."""
    DIR_INI.mkdir(parents=True, exist_ok=True)
    resultado = {}
    for nombre, meta in ARCHIVOS.items():
        ruta = DIR_INI / nombre
        if forzar or not ruta.exists():
            descargar(nombre, ruta)
        hash_actual = sha256(ruta)
        if hash_actual != meta["sha256"]:
            raise ValueError(
                f"Integridad comprometida en {nombre}: sha256={hash_actual} "
                f"(esperado {meta['sha256']}). No se continúa con el pipeline."
            )
        resultado[nombre] = {"ruta": str(ruta.relative_to(RAIZ)), "sha256": hash_actual}
        print(f"  OK  {nombre}  sha256={hash_actual[:16]}...")
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forzar", action="store_true", help="volver a descargar aunque existan")
    args = parser.parse_args()
    try:
        verificar_datos(forzar=args.forzar)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        sys.exit(1)
