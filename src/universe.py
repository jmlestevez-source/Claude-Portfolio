# src/universe.py
"""
Obtiene los tickers del Russell 1000 desde Wikipedia
y los sube automáticamente a GitHub.
"""

import os
import json
import base64
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

WIKI_URL = (
    "https://en.wikipedia.org/wiki/"
    "Russell_1000_Index#Components"
)
TABLE_ID  = "constituents"
LOCAL_PATH = Path("data/universe/tickers.json")

# Tickers conocidos como problemáticos en yfinance
KNOWN_BAD_TICKERS = {
    "BRK.B", "BF.B",   # Yahoo usa BRK-B, BF-B
}

TICKER_FIXES = {
    "BRK.B": "BRK-B",
    "BF.B":  "BF-B",
}


def scrape_russell_1000() -> list[dict]:
    """
    Parsea la tabla 'constituents' de Wikipedia.
    Devuelve lista de dicts con ticker, nombre, sector.
    """
    print("  Descargando Russell 1000 de Wikipedia...")

    try:
        tables = pd.read_html(
            WIKI_URL,
            attrs={"id": TABLE_ID},
            flavor="lxml",
        )
        if not tables:
            raise ValueError("Tabla no encontrada")

        df = tables[0]
        print(f"  Columnas detectadas: {list(df.columns)}")

        # Mapear columnas (pueden variar levemente)
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "symbol" in cl or "ticker" in cl:
                col_map["symbol"] = col
            elif "company" in cl or "security" in cl:
                col_map["company"] = col
            elif "sector" in cl and "sub" not in cl:
                col_map["sector"] = col
            elif "sub" in cl or "industry" in cl:
                col_map["sub_industry"] = col

        if "symbol" not in col_map:
            raise ValueError(
                f"Columna Symbol no encontrada. "
                f"Columnas: {list(df.columns)}"
            )

        components = []
        for _, row in df.iterrows():
            ticker = str(row[col_map["symbol"]]).strip()
            if not ticker or ticker == "nan":
                continue

            # Aplicar correcciones conocidas
            ticker = TICKER_FIXES.get(ticker, ticker)

            components.append({
                "ticker":       ticker,
                "company":      str(
                    row.get(col_map.get("company", ""), "")
                ).strip(),
                "sector":       str(
                    row.get(col_map.get("sector", ""), "")
                ).strip(),
                "sub_industry": str(
                    row.get(col_map.get("sub_industry", ""), "")
                ).strip(),
                "source":       "Russell1000_Wikipedia",
                "updated":      datetime.now().strftime(
                    "%Y-%m-%d"
                ),
            })

        print(f"  ✓ {len(components)} tickers extraídos")
        return components

    except Exception as e:
        print(f"  Error scraping Wikipedia: {e}")
        raise


def save_locally(components: list[dict]) -> None:
    """Guarda el universo en data/universe/tickers.json"""
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "updated":    datetime.now().isoformat(),
        "source":     WIKI_URL,
        "count":      len(components),
        "components": components,
        # Lista plana de tickers para compatibilidad
        "tickers":    [c["ticker"] for c in components],
    }

    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Guardado local: {LOCAL_PATH}")


def push_to_github(components: list[dict]) -> bool:
    """
    Sube/actualiza data/universe/tickers.json en GitHub
    usando la API REST.
    Requiere GITHUB_TOKEN y GITHUB_REPO en .env
    """
    token = os.getenv("GITHUB_TOKEN")
    repo  = os.getenv("GITHUB_REPO")   # "usuario/repositorio"

    if not token or not repo:
        print(
            "  ⚠ GITHUB_TOKEN o GITHUB_REPO no configurados. "
            "Solo se guarda localmente."
        )
        return False

    file_path = "data/universe/tickers.json"
    api_url   = (
        f"https://api.github.com/repos/{repo}"
        f"/contents/{file_path}"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
    }

    payload = {
        "updated":    datetime.now().isoformat(),
        "source":     WIKI_URL,
        "count":      len(components),
        "components": components,
        "tickers":    [c["ticker"] for c in components],
    }

    content_bytes = json.dumps(
        payload, indent=2, ensure_ascii=False
    ).encode("utf-8")
    content_b64 = base64.b64encode(content_bytes).decode()

    # Obtener SHA del archivo actual (para actualizarlo)
    sha = None
    r = requests.get(api_url, headers=headers, timeout=30)
    if r.status_code == 200:
        sha = r.json().get("sha")
        print(f"  Archivo GitHub existente, SHA: {sha[:8]}...")
    elif r.status_code == 404:
        print("  Archivo GitHub no existe, se creará.")
    else:
        print(f"  Error obteniendo SHA: {r.status_code}")
        return False

    # Crear o actualizar
    body: dict = {
        "message": (
            f"Update Russell 1000 universe "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        ),
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha

    r2 = requests.put(
        api_url, headers=headers,
        json=body, timeout=30,
    )

    if r2.status_code in (200, 201):
        print(f"  ✓ GitHub actualizado: {file_path}")
        return True
    else:
        print(
            f"  Error GitHub: {r2.status_code} "
            f"{r2.text[:200]}"
        )
        return False


def update_universe() -> list[str]:
    """
    Pipeline completo:
    1. Scraping Wikipedia
    2. Guarda local
    3. Sube a GitHub
    Devuelve lista de tickers.
    """
    print("\n📋 Actualizando universo Russell 1000...")

    components = scrape_russell_1000()
    save_locally(components)
    push_to_github(components)

    tickers = [c["ticker"] for c in components]
    print(f"  Universo final: {len(tickers)} tickers\n")
    return tickers


def load_universe() -> list[str]:
    """
    Carga el universo local.
    Si no existe o tiene más de 7 días, lo actualiza.
    """
    if LOCAL_PATH.exists():
        try:
            data = json.load(open(LOCAL_PATH))
            updated = datetime.fromisoformat(
                data.get("updated", "2000-01-01")
            )
            age_days = (datetime.now() - updated).days

            tickers = data.get("tickers", [])
            if tickers and age_days < 7:
                print(
                    f"  Universo local OK: "
                    f"{len(tickers)} tickers "
                    f"(actualizado hace {age_days}d)"
                )
                return tickers
            else:
                print(
                    f"  Universo local de {age_days}d, "
                    f"actualizando..."
                )
        except Exception as e:
            print(f"  Error leyendo universo local: {e}")

    return update_universe()
