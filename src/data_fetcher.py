# src/data_fetcher.py
"""
Descarga de datos de mercado:
- Paralela con ThreadPoolExecutor
- Caché diaria para evitar repetir llamadas
- Descarga histórica en batch con yfinance.download()
"""

import time
import pickle
import threading
import concurrent.futures
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime


# ── Caché ─────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/cache")

class DataCache:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        safe  = key.replace("/", "_").replace("^", "X")
        return CACHE_DIR / f"{safe}_{today}.pkl"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            try:
                with self._lock:
                    return pickle.load(open(p, "rb"))
            except Exception:
                pass
        return None

    def set(self, key: str, data) -> None:
        p = self._path(key)
        try:
            with self._lock:
                pickle.dump(data, open(p, "wb"))
        except Exception:
            pass

    def cleanup_old(self, keep_days: int = 3) -> None:
        """Borra cachés de más de keep_days días."""
        today = datetime.now().strftime("%Y-%m-%d")
        for f in CACHE_DIR.glob("*.pkl"):
            if today not in f.name:
                try:
                    f.unlink()
                except Exception:
                    pass

cache = DataCache()


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, calls_per_second: float = 3.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_called  = 0.0
        self._lock        = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self.last_called
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_called = time.time()

yf_limiter = RateLimiter(calls_per_second=3.0)


# ── Fundamentales ─────────────────────────────────────────────────────────────

FUNDAMENTAL_FIELDS = {
    "price":          ("currentPrice", "regularMarketPrice"),
    "forward_pe":     ("forwardPE",),
    "trailing_pe":    ("trailingPE",),
    "revenue_growth": ("revenueGrowth",),
    "earnings_growth":("earningsGrowth",),
    "gross_margins":  ("grossMargins",),
    "operating_margins": ("operatingMargins",),
    "free_cashflow":  ("freeCashflow",),
    "market_cap":     ("marketCap",),
    "enterprise_value": ("enterpriseValue",),
    "ev_to_ebitda":   ("enterpriseToEbitda",),
    "debt_to_equity": ("debtToEquity",),
    "current_ratio":  ("currentRatio",),
    "roe":            ("returnOnEquity",),
    "roa":            ("returnOnAssets",),
    "52w_high":       ("fiftyTwoWeekHigh",),
    "52w_low":        ("fiftyTwoWeekLow",),
    "beta":           ("beta",),
    "dividend_yield": ("dividendYield",),
    "sector":         ("sector",),
    "industry":       ("industry",),
    "description":    ("longBusinessSummary",),
    "employees":      ("fullTimeEmployees",),
    "country":        ("country",),
}

def _get_field(info: dict, keys: tuple):
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def fetch_fundamentals(ticker: str) -> dict:
    """
    Descarga fundamentales de un ticker.
    Usa caché diaria. Thread-safe.
    """
    cached = cache.get(f"fund_{ticker}")
    if cached is not None:
        return cached

    yf_limiter.wait()
    result = {"ticker": ticker, "_data_ok": False}

    try:
        info = yf.Ticker(ticker).info

        for field, keys in FUNDAMENTAL_FIELDS.items():
            v = _get_field(info, keys)
            # Truncar descripción
            if field == "description" and v:
                v = str(v)[:300]
            result[field] = v

        # Verificar que tenemos precio mínimo
        result["_data_ok"] = result.get("price") is not None
        result["_fetched_at"] = datetime.now().isoformat()

        cache.set(f"fund_{ticker}", result)

    except Exception as e:
        result["_error"] = str(e)

    return result


def fetch_fundamentals_parallel(
    tickers: list[str],
    max_workers: int = 8,
) -> dict[str, dict]:
    """
    Descarga fundamentales en paralelo.
    Devuelve dict {ticker: datos}.
    """
    results: dict[str, dict] = {}
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_ticker = {
            executor.submit(fetch_fundamentals, t): t
            for t in tickers
        }
        done = 0
        for future in concurrent.futures.as_completed(
            future_to_ticker
        ):
            ticker = future_to_ticker[future]
            done  += 1
            try:
                data = future.result(timeout=45)
                results[ticker] = data
                if done % 50 == 0:
                    print(
                        f"    [{done}/{total}] "
                        f"fundamentales descargados..."
                    )
            except Exception as e:
                results[ticker] = {
                    "ticker":   ticker,
                    "_data_ok": False,
                    "_error":   str(e),
                }

    ok  = sum(1 for d in results.values() if d.get("_data_ok"))
    bad = total - ok
    print(f"    ✓ {ok}/{total} OK | {bad} sin datos")
    return results


# ── Histórico de precios ──────────────────────────────────────────────────────

def fetch_price_history(
    tickers: list[str],
    period: str = "1y",
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Descarga histórico de precios en chunks.
    Usa yfinance.download() que es más eficiente
    que llamar Ticker por ticker.
    Devuelve DataFrame con Close prices.
    """
    cached = cache.get(f"hist_{period}_{len(tickers)}")
    if cached is not None:
        print(
            f"    Histórico desde caché: "
            f"{cached.shape}"
        )
        return cached

    all_closes = []
    print(
        f"    Descargando histórico {period} "
        f"para {len(tickers)} tickers en chunks "
        f"de {chunk_size}..."
    )

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            hist = yf.download(
                chunk,
                period=period,
                progress=False,
                threads=True,
                auto_adjust=True,
            )

            if hist.empty:
                continue

            # yfinance devuelve MultiIndex si >1 ticker
            if isinstance(hist.columns, pd.MultiIndex):
                close = hist["Close"]
            else:
                # Un solo ticker
                close = hist[["Close"]]
                close.columns = [chunk[0]]

            all_closes.append(close)
            time.sleep(0.5)  # pausa entre chunks

        except Exception as e:
            print(
                f"    Error chunk {i//chunk_size + 1}: {e}"
            )
            continue

    if not all_closes:
        return pd.DataFrame()

    combined = pd.concat(all_closes, axis=1)
    # Eliminar columnas duplicadas
    combined = combined.loc[
        :, ~combined.columns.duplicated()
    ]

    cache.set(f"hist_{period}_{len(tickers)}", combined)
    print(
        f"    ✓ Histórico: {combined.shape[0]} días, "
        f"{combined.shape[1]} tickers"
    )
    return combined


# ── Macro ─────────────────────────────────────────────────────────────────────

def fetch_macro_data() -> dict:
    """Datos de mercado para contexto macro."""
    cached = cache.get("macro")
    if cached:
        return cached

    result = {}
    macro_tickers = {
        "SPY":  "^GSPC",
        "VIX":  "^VIX",
        "TNX":  "^TNX",   # 10Y Treasury yield
        "DXY":  "DX-Y.NYB",
        "QQQ":  "QQQ",
        "IWM":  "IWM",    # Russell 2000
        "GLD":  "GLD",
        "HYG":  "HYG",    # High Yield
    }

    for label, ticker in macro_tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="20d")
            if not hist.empty:
                price   = float(hist["Close"].iloc[-1])
                ret_5d  = (
                    price / float(hist["Close"].iloc[-5]) - 1
                ) * 100 if len(hist) >= 5 else 0
                ret_20d = (
                    price / float(hist["Close"].iloc[0]) - 1
                ) * 100
                result[label] = {
                    "price":    price,
                    "ret_5d":   ret_5d,
                    "ret_20d":  ret_20d,
                }
        except Exception:
            pass

    cache.set("macro", result)
    return result
