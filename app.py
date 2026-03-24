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


# Lista hardcodeada de tickers S&P 500 con sectores (actualizada marzo 2026)
# Esta lista se puede actualizar manualmente cuando cambie la composición del índice
SP500_TICKERS = [
    # Communication Services
    ("AAPL", "Communication Services"), ("GOOGL", "Communication Services"), ("GOOG", "Communication Services"),
    ("META", "Communication Services"), ("NFLX", "Communication Services"), ("DIS", "Communication Services"),
    ("CMCSA", "Communication Services"), ("VZ", "Communication Services"), ("T", "Communication Services"),
    ("TMUS", "Communication Services"), ("CHTR", "Communication Services"), ("ATVI", "Communication Services"),
    ("EA", "Communication Services"), ("TTWO", "Communication Services"), ("FOX", "Communication Services"),
    ("FOXA", "Communication Services"), ("CBS", "Communication Services"), ("VIA", "Communication Services"),
    ("VIAC", "Communication Services"), ("OMC", "Communication Services"), ("IPG", "Communication Services"),
    # Information Technology
    ("MSFT", "Information Technology"), ("NVDA", "Information Technology"), ("AVGO", "Information Technology"),
    ("AMD", "Information Technology"), ("INTC", "Information Technology"), ("CSCO", "Information Technology"),
    ("QCOM", "Information Technology"), ("TXN", "Information Technology"), ("ADBE", "Information Technology"),
    ("CRM", "Information Technology"), ("ORCL", "Information Technology"), ("IBM", "Information Technology"),
    ("INTU", "Information Technology"), ("NOW", "Information Technology"), ("UBER", "Information Technology"),
    ("PYPL", "Information Technology"), ("ADP", "Information Technology"), ("FIS", "Information Technology"),
    ("FISV", "Information Technology"), ("GPN", "Information Technology"), ("V", "Information Technology"),
    ("MA", "Information Technology"), ("ACN", "Information Technology"), ("IT", "Information Technology"),
    ("CTSH", "Information Technology"), ("INFY", "Information Technology"), ("WDC", "Information Technology"),
    ("STX", "Information Technology"), ("NTAP", "Information Technology"), ("ANET", "Information Technology"),
    ("CSCO", "Information Technology"), ("JNPR", "Information Technology"), ("FFIV", "Information Technology"),
    ("PANW", "Information Technology"), ("SPLK", "Information Technology"), ("ZI", "Information Technology"),
    ("SNPS", "Information Technology"), ("CDNS", "Information Technology"), ("SWKS", "Information Technology"),
    ("XLNX", "Information Technology"), ("MCHP", "Information Technology"), ("ON", "Information Technology"),
    ("NXPI", "Information Technology"), ("MU", "Information Technology"), ("LRCX", "Information Technology"),
    ("KLAC", "Information Technology"), ("TER", "Information Technology"), ("ENTG", "Information Technology"),
    ("ADI", "Information Technology"), ("QRVO", "Information Technology"), ("IDXX", "Information Technology"),
    ("ZTS", "Information Technology"), ("ROP", "Information Technology"), ("TDY", "Information Technology"),
    ("HUBS", "Information Technology"), ("ZEN", "Information Technology"), ("DOCU", "Information Technology"),
    ("OKTA", "Information Technology"), ("TEAM", "Information Technology"), ("SAIL", "Information Technology"),
    ("NET", "Information Technology"), ("DDOG", "Information Technology"), ("ESTC", "Information Technology"),
    ("CRWD", "Information Technology"), ("ZS", "Information Technology"), ("SNOW", "Information Technology"),
    ("PLTR", "Information Technology"), ("COUP", "Information Technology"), ("BILL", "Information Technology"),
    # Health Care
    ("UNH", "Health Care"), ("JNJ", "Health Care"), ("LLY", "Health Care"), ("PFE", "Health Care"),
    ("ABBV", "Health Care"), ("MRK", "Health Care"), ("TMO", "Health Care"), ("ABT", "Health Care"),
    ("DHR", "Health Care"), ("BMY", "Health Care"), ("AMGN", "Health Care"), ("MDT", "Health Care"),
    ("GILD", "Health Care"), ("CVS", "Health Care"), ("CI", "Health Care"), ("HUM", "Health Care"),
    ("ELV", "Health Care"), ("CNC", "Health Care"), ("DGX", "Health Care"), ("LH", "Health Care"),
    ("ISRG", "Health Care"), ("IDXX", "Health Care"), ("ZTS", "Health Care"), ("REGN", "Health Care"),
    ("VRTX", "Health Care"), ("BIIB", "Health Care"), ("INCY", "Health Care"), ("ILMN", "Health Care"),
    ("IEX", "Health Care"), ("WAT", "Health Care"), ("TECH", "Health Care"), ("BRK-B", "Health Care"),
    # Financials
    ("BRK-A", "Financials"), ("JPM", "Financials"), ("V", "Financials"), ("MA", "Financials"),
    ("BAC", "Financials"), ("WFC", "Financials"), ("GS", "Financials"), ("MS", "Financials"),
    ("C", "Financials"), ("BLK", "Financials"), ("SCHW", "Financials"), ("AXP", "Financials"),
    ("USB", "Financials"), ("PNC", "Financials"), ("TFC", "Financials"), ("COF", "Financials"),
    ("SPGI", "Financials"), ("ICE", "Financials"), ("CME", "Financials"), ("NDAQ", "Financials"),
    ("MMC", "Financials"), ("AON", "Financials"), ("WTW", "Financials"), ("AJG", "Financials"),
    ("BRO", "Financials"), ("PGR", "Financials"), ("TRV", "Financials"), ("ALL", "Financials"),
    ("CB", "Financials"), ("CINF", "Financials"), ("MET", "Financials"), ("PRU", "Financials"),
    ("AFL", "Financials"), ("LNC", "Financials"), ("AMP", "Financials"), ("TROW", "Financials"),
    ("BEN", "Financials"), ("BLK", "Financials"), ("IVZ", "Financials"), ("STT", "Financials"),
    ("BK", "Financials"), ("NTRS", "Financials"), ("FRC", "Financials"), ("KEY", "Financials"),
    ("FITB", "Financials"), ("HBAN", "Financials"), ("RF", "Financials"), ("CFG", "Financials"),
    ("ZION", "Financials"), ("MTB", "Financials"), ("PNFP", "Financials"), ("SIVB", "Financials"),
    ("SYF", "Financials"), ("CMA", "Financials"),
    # Consumer Discretionary
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"), ("HD", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("NKE", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("TJX", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"), ("BKNG", "Consumer Discretionary"),
    ("ABNB", "Consumer Discretionary"), ("MAR", "Consumer Discretionary"), ("RCL", "Consumer Discretionary"),
    ("CCL", "Consumer Discretionary"), ("LVS", "Consumer Discretionary"), ("WYNN", "Consumer Discretionary"),
    ("MGM", "Consumer Discretionary"), ("HAS", "Consumer Discretionary"), ("MAT", "Consumer Discretionary"),
    ("GM", "Consumer Discretionary"), ("F", "Consumer Discretionary"), ("RIVN", "Consumer Discretionary"),
    ("LCID", "Consumer Discretionary"), ("EBAY", "Consumer Discretionary"), ("ETSY", "Consumer Discretionary"),
    ("BIDU", "Consumer Discretionary"), ("EXPE", "Consumer Discretionary"), ("TRIP", "Consumer Discretionary"),
    ("YELP", "Consumer Discretionary"), ("GRMN", "Consumer Discretionary"), ("DLTR", "Consumer Discretionary"),
    ("DG", "Consumer Discretionary"), ("FIVE", "Consumer Discretionary"), ("OLLI", "Consumer Discretionary"),
    ("ROSS", "Consumer Discretionary"), ("BBY", "Consumer Discretionary"), ("KSS", "Consumer Discretionary"),
    ("M", "Consumer Discretionary"), ("JWN", "Consumer Discretionary"), ("DDS", "Consumer Discretionary"),
    # Consumer Staples
    ("PG", "Consumer Staples"), ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("WMT", "Consumer Staples"), ("PM", "Consumer Staples"),
    ("MO", "Consumer Staples"), ("MDLZ", "Consumer Staples"), ("CL", "Consumer Staples"),
    ("KMB", "Consumer Staples"), ("KHC", "Consumer Staples"), ("GIS", "Consumer Staples"),
    ("CAG", "Consumer Staples"), ("SJM", "Consumer Staples"), ("CPB", "Consumer Staples"),
    ("HRL", "Consumer Staples"), ("TSN", "Consumer Staples"), ("HRL", "Consumer Staples"),
    ("WBA", "Consumer Staples"), ("COST", "Consumer Staples"), ("TGT", "Consumer Staples"),
    ("DG", "Consumer Staples"), ("DLTR", "Consumer Staples"), ("BV", "Consumer Staples"),
    ("CHD", "Consumer Staples"), ("CLX", "Consumer Staples"), ("COLG", "Consumer Staples"),
    ("EL", "Consumer Staples"), ("ESTEE", "Consumer Staples"),
    # Industrials
    ("GE", "Industrials"), ("CAT", "Industrials"), ("DE", "Industrials"), ("HON", "Industrials"),
    ("UPS", "Industrials"), ("FDX", "Industrials"), ("BA", "Industrials"), ("RTX", "Industrials"),
    ("LMT", "Industrials"), ("NOC", "Industrials"), ("GD", "Industrials"), ("TDY", "Industrials"),
    ("TXT", "Industrials"), ("LHX", "Industrials"), ("HWM", "Industrials"), ("PNR", "Industrials"),
    ("ETN", "Industrials"), ("EMR", "Industrials"), ("ROK", "Industrials"), ("ABBV", "Industrials"),
    ("PH", "Industrials"), ("IR", "Industrials"), ("MMM", "Industrials"), ("HII", "Industrials"),
    ("SWK", "Industrials"), ("BLDR", "Industrials"), ("MAS", "Industrials"), ("FBHS", "Industrials"),
    ("WSO", "Industrials"), ("GWW", "Industrials"), ("FAST", "Industrials"), ("MSI", "Industrials"),
    ("XYL", "Industrials"), ("IDEX", "Industrials"), ("IEX", "Industrials"), ("ITW", "Industrials"),
    ("AME", "Industrials"), ("ROP", "Industrials"), ("CTAS", "Industrials"), ("RE", "Industrials"),
    ("AON", "Industrials"), ("MMC", "Industrials"), ("WTW", "Industrials"),
    # Energy
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"), ("SLB", "Energy"),
    ("EOG", "Energy"), ("PSX", "Energy"), ("VLO", "Energy"), ("MPC", "Energy"),
    ("OXY", "Energy"), ("FANG", "Energy"), ("APA", "Energy"), ("DVN", "Energy"),
    ("MRO", "Energy"), ("HAL", "Energy"), ("NOV", "Energy"), ("WMB", "Energy"),
    ("KMI", "Energy"), ("OKE", "Energy"), ("EPD", "Energy"), ("ET", "Energy"),
    ("PBA", "Energy"), ("ENB", "Energy"), ("TRP", "Energy"), ("CNQ", "Energy"),
    # Utilities
    ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"), ("D", "Utilities"),
    ("EXC", "Utilities"), ("AEP", "Utilities"), ("SRE", "Utilities"), ("XEL", "Utilities"),
    ("WEC", "Utilities"), ("PEG", "Utilities"), ("ES", "Utilities"), ("ED", "Utilities"),
    ("EIX", "Utilities"), ("PPL", "Utilities"), ("FE", "Utilities"), ("CMS", "Utilities"),
    ("AES", "Utilities"), ("NRG", "Utilities"), ("VST", "Utilities"), ("CEG", "Utilities"),
    # Real Estate
    ("AMT", "Real Estate"), ("PLD", "Real Estate"), ("CCI", "Real Estate"), ("EQIX", "Real Estate"),
    ("PSA", "Real Estate"), ("SPG", "Real Estate"), ("O", "Real Estate"), ("WELL", "Real Estate"),
    ("DLR", "Real Estate"), ("AVB", "Real Estate"), ("EQR", "Real Estate"), ("VTR", "Real Estate"),
    ("INVH", "Real Estate"), ("SBAC", "Real Estate"), ("WY", "Real Estate"), ("ARE", "Real Estate"),
    ("VNO", "Real Estate"), ("BXP", "Real Estate"), ("SLG", "Real Estate"), ("HST", "Real Estate"),
    # Materials
    ("LIN", "Materials"), ("APD", "Materials"), ("SHW", "Materials"), ("FCX", "Materials"),
    ("NEM", "Materials"), ("DOW", "Materials"), ("DD", "Materials"), ("EMN", "Materials"),
    ("FMC", "Materials"), ("CTVA", "Materials"), ("CF", "Materials"), ("MOS", "Materials"),
    ("NUE", "Materials"), ("STLD", "Materials"), ("X", "Materials"), ("AA", "Materials"),
    ("IFF", "Materials"), ("PPG", "Materials"), ("ALB", "Materials"), ("MPC", "Materials"),
]


def get_sp500_tickers():
    """Obtiene los tickers del S&P 500 desde la lista hardcodeada."""
    try:
        df = pd.DataFrame(SP500_TICKERS, columns=['ticker', 'sector'])
        # Limpiar tickers (algunos pueden tener puntos)
        df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['ticker'])
        logger.info(f"Obtenidos {len(df)} tickers del S&P 500")
        return df
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
