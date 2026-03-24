"""
S&P 500 RSI Monitor API
API Flask para generar informes de RSI del S&P 500
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

# Lista optimizada de tickers S&P 500 (sin duplicados)
SP500_TICKERS = [
    # Technology
    ("AAPL", "Information Technology"), ("MSFT", "Information Technology"), 
    ("NVDA", "Information Technology"), ("GOOGL", "Communication Services"),
    ("AMZN", "Consumer Discretionary"), ("META", "Communication Services"),
    ("TSLA", "Consumer Discretionary"), ("AVGO", "Information Technology"),
    ("AMD", "Information Technology"), ("INTC", "Information Technology"),
    ("CSCO", "Information Technology"), ("ORCL", "Information Technology"),
    ("ADBE", "Information Technology"), ("CRM", "Information Technology"),
    ("QCOM", "Information Technology"), ("TXN", "Information Technology"),
    ("INTU", "Information Technology"), ("NOW", "Information Technology"),
    ("PYPL", "Information Technology"), ("IBM", "Information Technology"),
    ("ANET", "Information Technology"), ("PANW", "Information Technology"),
    ("MU", "Information Technology"), ("LRCX", "Information Technology"),
    ("KLAC", "Information Technology"), ("SNPS", "Information Technology"),
    ("CDNS", "Information Technology"), ("MCHP", "Information Technology"),
    ("NXPI", "Information Technology"), ("ON", "Information Technology"),
    ("CRWD", "Information Technology"), ("SNOW", "Information Technology"),
    
    # Communication Services
    ("NFLX", "Communication Services"), ("DIS", "Communication Services"),
    ("CMCSA", "Communication Services"), ("VZ", "Communication Services"),
    ("T", "Communication Services"), ("TMUS", "Communication Services"),
    
    # Health Care
    ("UNH", "Health Care"), ("JNJ", "Health Care"), ("LLY", "Health Care"),
    ("ABBV", "Health Care"), ("MRK", "Health Care"), ("TMO", "Health Care"),
    ("ABT", "Health Care"), ("DHR", "Health Care"), ("PFE", "Health Care"),
    ("BMY", "Health Care"), ("AMGN", "Health Care"), ("MDT", "Health Care"),
    ("GILD", "Health Care"), ("CVS", "Health Care"), ("CI", "Health Care"),
    ("ISRG", "Health Care"), ("REGN", "Health Care"), ("VRTX", "Health Care"),
    ("ZTS", "Health Care"), ("HUM", "Health Care"),
    
    # Financials
    ("JPM", "Financials"), ("V", "Financials"), ("MA", "Financials"),
    ("BAC", "Financials"), ("WFC", "Financials"), ("GS", "Financials"),
    ("MS", "Financials"), ("BLK", "Financials"), ("SPGI", "Financials"),
    ("C", "Financials"), ("SCHW", "Financials"), ("AXP", "Financials"),
    ("PGR", "Financials"), ("CB", "Financials"), ("MMC", "Financials"),
    ("ICE", "Financials"), ("CME", "Financials"), ("AON", "Financials"),
    ("USB", "Financials"), ("PNC", "Financials"), ("TFC", "Financials"),
    ("COF", "Financials"),
    
    # Consumer Discretionary
    ("HD", "Consumer Discretionary"), ("MCD", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("SBUX", "Consumer Discretionary"), ("TJX", "Consumer Discretionary"),
    ("BKNG", "Consumer Discretionary"), ("ABNB", "Consumer Discretionary"),
    ("GM", "Consumer Discretionary"), ("F", "Consumer Discretionary"),
    ("MAR", "Consumer Discretionary"), ("HLT", "Consumer Discretionary"),
    
    # Consumer Staples
    ("PG", "Consumer Staples"), ("KO", "Consumer Staples"),
    ("PEP", "Consumer Staples"), ("COST", "Consumer Staples"),
    ("WMT", "Consumer Staples"), ("PM", "Consumer Staples"),
    ("MO", "Consumer Staples"), ("MDLZ", "Consumer Staples"),
    ("CL", "Consumer Staples"), ("KMB", "Consumer Staples"),
    ("GIS", "Consumer Staples"), ("KHC", "Consumer Staples"),
    
    # Industrials
    ("GE", "Industrials"), ("CAT", "Industrials"), ("RTX", "Industrials"),
    ("HON", "Industrials"), ("UPS", "Industrials"), ("BA", "Industrials"),
    ("DE", "Industrials"), ("LMT", "Industrials"), ("NOC", "Industrials"),
    ("GD", "Industrials"), ("MMM", "Industrials"), ("ETN", "Industrials"),
    ("EMR", "Industrials"), ("ITW", "Industrials"),
    
    # Energy
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    ("SLB", "Energy"), ("EOG", "Energy"), ("PSX", "Energy"),
    ("MPC", "Energy"), ("VLO", "Energy"), ("OXY", "Energy"),
    ("HAL", "Energy"), ("KMI", "Energy"), ("WMB", "Energy"),
    
    # Utilities
    ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"),
    ("D", "Utilities"), ("AEP", "Utilities"), ("EXC", "Utilities"),
    ("SRE", "Utilities"), ("XEL", "Utilities"), ("WEC", "Utilities"),
    
    # Real Estate
    ("AMT", "Real Estate"), ("PLD", "Real Estate"), ("CCI", "Real Estate"),
    ("EQIX", "Real Estate"), ("PSA", "Real Estate"), ("SPG", "Real Estate"),
    ("O", "Real Estate"), ("WELL", "Real Estate"), ("DLR", "Real Estate"),
    
    # Materials
    ("LIN", "Materials"), ("APD", "Materials"), ("SHW", "Materials"),
    ("FCX", "Materials"), ("NEM", "Materials"), ("DOW", "Materials"),
    ("NUE", "Materials"), ("ECL", "Materials"), ("DD", "Materials"),
]


def get_sp500_tickers():
    """Obtiene los tickers del S&P 500 desde la lista hardcodeada."""
    try:
        df = pd.DataFrame(SP500_TICKERS, columns=['ticker', 'sector'])
        df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
        df = df.drop_duplicates(subset=['ticker'])
        logger.info(f"✅ Obtenidos {len(df)} tickers del S&P 500")
        return df
    except Exception as e:
        logger.error(f"❌ Error obteniendo tickers: {e}")
        return None


def calculate_rsi(series, period=14):
    """Calcula el RSI de una serie de precios."""
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
    except Exception as e:
        logger.error(f"Error calculando RSI: {e}")
        return None


def get_rsi_for_tickers(tickers_data, batch_size=20):
    """Obtiene el RSI de 14 días para todos los tickers."""
    tickers = tickers_data['ticker'].tolist()
    results = []
    total = len(tickers)
    
    logger.info(f"🔄 Iniciando descarga de {total} tickers...")

    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"📊 Procesando lote {i//batch_size + 1}: {i+1}-{min(i+batch_size, total)}/{total}")

        try:
            # Descargar datos con manejo de errores
            data = yf.download(
                batch, 
                period='2mo', 
                progress=False, 
                group_by='ticker',
                threads=True,
                ignore_tz=True
            )

            for ticker in batch:
                try:
                    # Manejar tanto descarga individual como múltiple
                    if len(batch) == 1:
                        close_prices = data['Close']
                    else:
                        if ticker in data.columns.levels[0]:
                            close_prices = data[ticker]['Close']
                        else:
                            logger.warning(f"⚠️ {ticker} no encontrado")
                            continue

                    # Verificar que tenemos suficientes datos
                    if close_prices is not None and len(close_prices.dropna()) >= 20:
                        rsi = calculate_rsi(close_prices)
                        
                        if rsi is not None and not pd.isna(rsi):
                            sector = tickers_data[tickers_data['ticker'] == ticker]['sector'].values[0]
                            
                            results.append({
                                'ticker': ticker,
                                'sector': sector,
                                'rsi': round(float(rsi), 2),
                                'oversold': float(rsi) < 30
                            })
                            logger.info(f"✓ {ticker}: RSI {round(float(rsi), 2)}")
                    else:
                        logger.warning(f"⚠️ {ticker}: Datos insuficientes")
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando {ticker}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error en lote {i//batch_size + 1}: {e}")
            continue

    logger.info(f"✅ Procesados {len(results)} tickers exitosamente")
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

    def get_bar(pct, width=8):
        filled = int(pct / 100 * width)
        return '█' * filled + '░' * (width - filled)

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
        sector_name = row['sector'][:20]
        bar = get_bar(row['pct_oversold'])
        message += f"<code>{sector_name:20}</code> {bar} {row['pct_oversold']}% ({int(row['oversold'])}/{int(row['total'])})\n"

    oversold_stocks = df[df['oversold']].nsmallest(min(10, total_oversold), 'rsi')[['ticker', 'rsi']]

    if not oversold_stocks.empty:
        message += f"""
🔴 <b>TOP SOBREVENDIDAS</b>
"""
        for i, (_, row) in enumerate(oversold_stocks.iterrows(), 1):
            message += f"  {i}. {row['ticker']}: RSI <b>{row['rsi']:.1f}</b>\n"

    message += """
━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 <i>RSI &lt; 30 indica sobreventa.</i>"""

    return message


@app.route('/')
def index():
    """Endpoint raíz."""
    return jsonify({
        "service": "S&P 500 RSI Monitor API",
        "status": "online",
        "endpoints": {
            "/rsi-report": "GET - Genera informe RSI completo",
            "/health": "GET - Health check"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    })


@app.route('/rsi-report')
def rsi_report():
    """Genera el informe completo de RSI del S&P 500."""
    try:
        logger.info("🚀 Iniciando generación de informe RSI...")

        # 1. Obtener tickers
        tickers_data = get_sp500_tickers()
        if tickers_data is None or tickers_data.empty:
            raise Exception("No se pudieron obtener los tickers")

        # 2. Obtener RSI
        df = get_rsi_for_tickers(tickers_data)
        
        if df.empty:
            raise Exception("No se obtuvieron datos de RSI")

        logger.info(f"✅ Datos obtenidos para {len(df)} acciones")

        # 3. Analizar por sector
        analysis = analyze_by_sector(df)
        
        if analysis is None or analysis.empty:
            raise Exception("Error analizando datos por sector")

        # 4. Generar informe
        message = generate_report(analysis, df)

        # 5. Preparar respuesta
        response = {
            "success": True,
            "message": message,
            "stats": {
                "total_tickers": int(len(df)),
                "total_oversold": int(df['oversold'].sum()),
                "pct_oversold": round(float(df['oversold'].sum() / len(df) * 100), 1),
                "sectors_affected": int(len(analysis[analysis['oversold'] > 0])),
                "total_sectors": int(len(analysis))
            },
            "sectors": analysis.to_dict('records'),
            "oversold_stocks": df[df['oversold']].nsmallest(20, 'rsi')[['ticker', 'sector', 'rsi']].to_dict('records'),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M")
        }

        logger.info("✅ Informe generado correctamente")
        return jsonify(response), 200

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error generando informe: {error_msg}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "message": f"⚠️ Error: {error_msg}",
            "traceback": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
