from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from  predict_trend import predict_trend   
import yfinance as yf

app = FastAPI() 
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
def predict(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return JSONResponse(status_code=404, content={"error": "No data found."})
        result = predict_trend(hist)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/data")
def stock_data(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return JSONResponse(status_code=404, content={"error": "No data found."})

        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)

        info = ticker.info
        return {
            "history": hist.tail(10).to_dict(orient="records"),
            "chartData": {
                "dates": hist["Date"].tolist(),
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist(),
                "volume": hist["Volume"].tolist()
            },
            "metrics": {
                "current": round(hist["Close"].iloc[-1], 2),
                "high": round(hist["High"].iloc[-1], 2),
                "low": round(hist["Low"].iloc[-1], 2),
                "marketCap": round(info.get("marketCap", 0) / 1e12, 2),
                "peRatio": info.get("trailingPE", "N/A"),
                "dividendYield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else "N/A"
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

