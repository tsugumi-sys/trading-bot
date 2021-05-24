from typing import Optional
from fastapi import FastAPI, Query, Path, Request 
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from lstm import lstm

app = FastAPI()

templates = Jinja2Templates(directory='templates')
@app.get('/')
async def index_page(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/api/predict')
async def predict(
    market: Optional[str] = Query(
        'binance',
        title='The name of exchange',
        description='The name of crypto exchange such as binance, bitflyer ...'
    ),
    symbol: Optional[str] = Query(
        'btcusdt',
        title='The name of crypto pairs',
        description='The name of crypto pairs such as btcusdt, ethusdt ...'
    ),
    freq: Optional[int] = Query(
        7200,
        title='The frequency of cryptowatch data',
        description='Time frequency (second) of cryptowatch time series data. (e.g. 7200)'
    )
):
    res = {
        'market': market,
        'symbol': symbol,
        'frequency': freq,
        'predict': lstm(market, symbol, freq)
    }
    return res

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))