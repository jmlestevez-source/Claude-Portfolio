"""
S&P 500 RSI Monitor API
API Flask para generar informes de RSI del S&P 500
Desplegar en Render como Web Service
"""

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import traceback

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def get_sp500_tickers():
    """Obtiene los tickers del S&P 500 desde Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers_data = df[['Symbol', 'GICS Sector']].copy()
        tickers_data.columns = ['ticker', 'sector']
        tickers_data['ticker'] = tickers_data['ticker'].str.replace('.', '-', regex=False)
        logger.info(f"Obtenidos {len(tickers_data)} tickers del S&P 500")
        return tickers_data
    except Exception as e:
        logger.error(f"Error obteniendo tickers: {e}")
        return None


def calculate_rsi(series, period=14):
    """Calcula el RSI de una serie de precios."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None


def get_rsi_for_tickers(tickers_data, batch_size=50):
    """Obtiene el RSI de 14 días para todos los tickers."""
    tickers = tickers_data['ticker'].tolist()
    results = []
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Descargando: {i+1}-{min(i+batch_size, total)} de {total}")

        try:
            data = yf.download(batch, period='2mo', progress=False, group_by='ticker')

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        close_prices = data['Close']
                    else:
                        close_prices = data[ticker]['Close'] if ticker in data.columns else None

                    if close_prices is not None and len(close_prices) > 15:
                        rsi = calculate_rsi(close_prices)
                        sector = tickers_data[tickers_data['ticker'] == ticker]['sector'].values[0]

                        if rsi is not None:
                            results.append({
                                'ticker': ticker,
                                'sector': sector,
                                'rsi': round(rsi, 2),
                                'oversold': rsi < 30
                            })
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error en lote {i}: {e}")
            continue

    return pd.DataFrame(results)


def analyze_by_sector(df):
    """Analiza los datos por sector."""
    if df.empty:
        return None

    analysis = df.groupby('sector').agg(
        total=('ticker', 'count'),
        oversold=('oversold', 'sum')
    ).reset_index()

    analysis['pct_oversold'] = (analysis['oversold'] / analysis['total'] * 100).round(1)
    analysis = analysis.sort_values('pct_oversold', ascending=False)

    return analysis


def generate_report(analysis, df):
    """Genera el informe completo."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    total_tickers = len(df)
    total_oversold = int(df['oversold'].sum())
    pct_total = round(total_oversold / total_tickers * 100, 1)
    sectors_with_oversold = len(analysis[analysis['oversold'] > 0])
    total_sectors = len(analysis)

    # Barra visual
    def get_bar(pct, width=8):
        filled = int(pct / 100 * width)
        return '█' * filled + '░' * (width - filled)

    # Mensaje para Telegram (HTML)
    message = f"""📊 <b>S&P 500 RSI MONITOR</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 {now}

📈 <b>RESUMEN GENERAL</b>
• Total analizados: <b>{total_tickers}</b> acciones
• En sobreventa (RSI&lt;30): <b>{total_oversold}</b>
• Porcentaje total: <b>{pct_total}%</b>
• Sectores afectados: <b>{sectors_with_oversold}/{total_sectors}</b>

📉 <b>DETALLE POR SECTOR</b>
"""

    for _, row in analysis.iterrows():
        bar = get_bar(row['pct_oversold'])
        message += f"<code>{row['sector'][:15]:15}</code> {bar} {row['pct_oversold']}% ({int(row['oversold'])}/{int(row['total'])})\n"

    # Top oversold
    oversold_stocks = df[df['oversold']].nsmallest(min(10, total_oversold), 'rsi')[['ticker', 'rsi']]

    if not oversold_stocks.empty:
        message += f"""
🔴 <b>TOP SOBREVENDIDAS</b>
"""
        for i, (_, row) in enumerate(oversold_stocks.iterrows(), 1):
            message += f"  {i}. {row['ticker']}: RSI <b>{row['rsi']:.1f}</b>\n"

    message += """
━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 <i>RSI &lt; 30 indica sobreventa. Posible señal de rebote.</i>"""

    return message


@app.route('/')
def index():
    """Endpoint raíz."""
    return jsonify({
        "service": "S&P 500 RSI Monitor API",
        "endpoints": {
            "/rsi-report": "GET - Genera informe RSI completo",
            "/health": "GET - Health check"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/rsi-report')
def rsi_report():
    """
    Genera el informe completo de RSI del S&P 500.
    Devuelve JSON con el mensaje formateado para Telegram y estadísticas.
    """
    try:
        logger.info("Iniciando generación de informe RSI...")

        # 1. Obtener tickers
        tickers_data = get_sp500_tickers()
        if tickers_data is None:
            return jsonify({"error": "No se pudieron obtener los tickers"}), 500

        # 2. Obtener RSI
        df = get_rsi_for_tickers(tickers_data)
        if df.empty:
            return jsonify({"error": "No se obtuvieron datos"}), 500

        logger.info(f"Datos obtenidos para {len(df)} acciones")

        # 3. Analizar
        analysis = analyze_by_sector(df)

        # 4. Generar informe
        message = generate_report(analysis, df)

        # 5. Preparar respuesta
        response = {
            "success": True,
            "message": message,
            "stats": {
                "total_tickers": int(len(df)),
                "total_oversold": int(df['oversold'].sum()),
                "pct_oversold": round(df['oversold'].sum() / len(df) * 100, 1),
                "sectors_affected": int(len(analysis[analysis['oversold'] > 0])),
                "total_sectors": int(len(analysis))
            },
            "sectors": analysis.to_dict('records'),
            "oversold_stocks": df[df['oversold']].nsmallest(20, 'rsi')[['ticker', 'sector', 'rsi']].to_dict('records'),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M")
        }

        logger.info("Informe generado correctamente")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generando informe: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(5000))