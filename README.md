# S&P 500 RSI Monitor API

API Flask que genera informes de RSI del S&P 500 para su envío por Telegram.

## Despliegue en Render

### 1. Subir a GitHub

```bash
# Crear nuevo repositorio en GitHub (ej: sp500-rsi-api)
# Desde la carpeta rsi-api:

git init
git add .
git commit -m "Initial commit - S&P 500 RSI Monitor API"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/sp500-rsi-api.git
git push -u origin main
```

### 2. Crear servicio en Render

1. Ve a [Render Dashboard](https://dashboard.render.com/)
2. Click en **New** → **Web Service**
3. Conecta tu cuenta de GitHub
4. Selecciona el repositorio `sp500-rsi-api`
5. Configuración:
   - **Name**: `sp500-rsi-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free

6. Click en **Deploy Web Service**

### 3. Verificar despliegue

Una vez desplegado, tendrás una URL como:
```
https://sp500-rsi-api.onrender.com
```

Puedes probar el endpoint:
```
https://sp500-rsi-api.onrender.com/rsi-report
```

### 4. Configurar n8n

1. En n8n, importa el workflow desde `n8n_workflow.json`
2. Edita el nodo "Obtener Informe RSI" y cambia `TU-SERVICIO` por tu URL real
3. Crea una credencial de Telegram:
   - Ve a Settings → Credentials → Add Credential
   - Selecciona "Telegram API"
   - Pega el token de tu bot (de @BotFather)
4. Configura la variable de entorno `TELEGRAM_CHAT_ID` con tu Chat ID
5. Activa el workflow

## Endpoints

| Endpoint | Descripción |
|----------|-------------|
| `GET /` | Info del servicio |
| `GET /health` | Health check |
| `GET /rsi-report` | Genera informe completo |

## Formato de respuesta

```json
{
  "success": true,
  "message": "📊 S&P 500 RSI MONITOR...",
  "stats": {
    "total_tickers": 503,
    "total_oversold": 45,
    "pct_oversold": 8.9,
    "sectors_affected": 8,
    "total_sectors": 11
  },
  "sectors": [...],
  "oversold_stocks": [...]
}
```