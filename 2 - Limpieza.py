#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import re
import sys
import warnings
from html import unescape
from pathlib import Path

import pandas as pd

# --- Configuración ----------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

FORBIDDEN_PATTERNS = [
    r"from\s+wikipedia",
    r"also\s+contained\s+in",
    r"summary\s+adapted\s+from",
    r"this\s+article\s+is\s+about",
    r"see\s+also",
    r"pronunciation",
    r"further\s+reading",
    r"this\s+page\s+was\s+last\s+edited",
    r"goodreads",
    r"wikidata",
]

# --- Funciones de limpieza --------------------------------------------------

def strip_html(text: str) -> str:
    """Elimina etiquetas HTML y decodifica entidades (&amp;, &quot;...)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text)


def remove_urls(text: str) -> str:
    """Elimina URLs (http, https, www, ftp)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"(?:https?|ftp)://\S+|www\.\S+", " ", text, flags=re.I)


def remove_references(text: str) -> str:
    """Elimina referencias tipo [1], [Wikipedia], (goodreads), etc."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\[.*?(wikipedia|goodreads).*?\]", " ", text, flags=re.I)
    text = re.sub(r"\(.*?(wikipedia|goodreads|pronunciation).*?\)", " ", text, flags=re.I)
    return text


def remove_symbols(text: str) -> str:
    """Quita asteriscos, guiones múltiples y símbolos repetidos."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\*#_]+", " ", text)
    text = re.sub(r"[-–—]{2,}", " ", text)
    return text


def remove_parentheses(text: str) -> str:
    """Elimina texto entre paréntesis con aclaraciones de idioma o pronunciación."""
    if not isinstance(text, str):
        return ""
    return re.sub(
        r"\([^)]{0,80}(pronunciation|translation|idioma|language|traducción)[^)]*\)",
        " ",
        text,
        flags=re.I,
    )


def remove_quotes(text: str) -> str:
    """Elimina comillas rectas y curvas."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\"'‘’“”«»]", "", text)


def remove_forbidden_patterns(text: str) -> str:
    """Elimina frases o patrones indeseados."""
    if not isinstance(text, str):
        return ""
    for pattern in FORBIDDEN_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    return text


def normalize_whitespace(text: str) -> str:
    """Normaliza espacios en blanco múltiples."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Limpieza completa del texto (sin etiquetas de idioma ni estadísticas)."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = strip_html(text)
    text = remove_urls(text)
    text = remove_references(text)
    text = remove_symbols(text)
    text = remove_parentheses(text)
    text = remove_quotes(text)
    text = remove_forbidden_patterns(text)
    text = normalize_whitespace(text)

    return text


def coerce_rating(val) -> float:
    """Convierte valores numéricos al rango [0, 5]."""
    try:
        num = float(val)
        if math.isnan(num):
            return math.nan
        return max(0.0, min(5.0, num))
    except (ValueError, TypeError):
        return math.nan


# --- Procesamiento del DataFrame -------------------------------------------

def clean_dataframe(df: pd.DataFrame, min_desc_len: int = 30, drop_no_author: bool = False) -> pd.DataFrame:
    """Limpia el DataFrame y aplica filtros básicos."""
    # Limpieza de columnas
    for col in ["title", "description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)

    if "authors" in df.columns:
        df["authors"] = df["authors"].astype(str).str.replace("nan", "", regex=False).str.replace("None", "", regex=False)

    if "avg_rating" in df.columns:
        df["avg_rating"] = df["avg_rating"].apply(coerce_rating)

    # Filtros
    df = df[df["description"].str.len() >= min_desc_len]
    df = df[df["title"].str.len() >= 2]

    if drop_no_author and "authors" in df.columns:
        df = df[df["authors"].str.strip() != ""]

    df = df.drop_duplicates(subset=["title", "authors"], keep="first")

    return df


def generate_report(df: pd.DataFrame, report_path: str) -> None:
    """Genera un reporte simple."""
    report = f"""# Limpieza completada
- Filas finales: {len(df):,}
- Columnas: {', '.join(df.columns)}

---
*Generado por limpieza_mejorado.py*
"""
    Path(report_path).write_text(report, encoding="utf-8")
    logger.info(f"Reporte guardado: {report_path}")


# --- CLI y flujo principal --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Script de limpieza de datos de libros")
    p.add_argument("--in", dest="inp", default="fiction_books.csv", help="Archivo CSV de entrada")
    p.add_argument("--out", dest="out", default="fiction_books_clean.csv", help="Archivo CSV de salida")
    p.add_argument("--min_desc_len", type=int, default=30, help="Longitud mínima de descripción")
    p.add_argument("--drop_no_author", action="store_true", help="Eliminar filas sin autor")
    p.add_argument("--report", type=str, default="report.md", help="Archivo de reporte")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    inp_path = script_dir / args.inp
    out_path = script_dir / args.out
    report_path = script_dir / args.report

    if not inp_path.exists():
        logger.error(f"No existe el archivo {inp_path}")
        sys.exit(1)

    logger.info(f"Leyendo archivo: {inp_path}")
    df = pd.read_csv(inp_path)

    logger.info("Procesando limpieza...")
    df = clean_dataframe(df, args.min_desc_len, args.drop_no_author)

    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Archivo limpio guardado: {out_path}")

    generate_report(df, report_path)

    # --- Apertura automática del archivo limpio ---
    try:
        logger.info("Abriendo archivo limpio...")
        os.system(f'open "{out_path}"')  # macOS
        # os.startfile(out_path)  # Descomentar si usás Windows
    except Exception as e:
        logger.warning(f"No se pudo abrir automáticamente el archivo: {e}")


if __name__ == "__main__":
    main()
