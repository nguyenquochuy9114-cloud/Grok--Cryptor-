import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COINS = {name: id for id, name in [("bitcoin", "Bitcoin"), ("ethereum", "Ethereum"), ("binancecoin", "BNB"), ("solana", "Solana"), ("ripple", "XRP")]}

@st.cache_data(ttl=300)
def get_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["time", "price"]).merge(
            pd.DataFrame(data["total_volumes"], columns=["time", "volume"]),
            on="time"
        ).merge(pd.DataFrame(data["market_caps"], columns=["time", "market_cap"]), on="time")
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["volume_percent_mc"] = (df["volume"] / df["market_cap"]) * 100
        df["price_change"] = df["price"].pct_change()
        df["inflow"] = np.where(df["price_change"] > 0, df["volume"] * df["price_change"], 0)
        df["outflow"] = np.where(df["price_change"] < 0, df["volume"] * abs(df["price_change"]), 0)
        if len(df) > 500: df = df.resample("D", on="time").agg({"price": "last", "volume": "sum", "market_cap": "last", "volume_percent_mc": "mean", "inflow": "sum", "outflow": "sum"}).reset_index()
        return df
    except Exception as e:
        st.error(f"Lỗi API: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_rsi(prices): return 100 - (100 / (1 + prices.diff().where(lambda x: x > 0, 0).rolling(14).mean() / prices.diff().where(lambda x: x < 0, -x).rolling(14).mean()))

@st.cache_data
def find_signals(df):
    df["RSI"] = calculate_rsi(df["price"])
    df["MACD"] = df["price"].ewm(span=12).mean() - df["price"].ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    buy = df[(df["RSI"] < 30) & (df["MACD"] > df["MACD_signal"]) & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))]
    sell = df[(df["RSI"] > 70) & (df["MACD"] < df["MACD_signal"]) & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))]
    return pd.concat([buy, sell])[["time", "price", "RSI", "Signal"]].assign(Signal=lambda x: np.where(x.index.isin(buy.index), "Buy", "Sell"))

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Crypto Analyzer")
    coin = st.selectbox("Coin", list(COINS.keys()), format_func=lambda x: COINS[x])
    days = st.slider("Ngày", 1, 365, 30)

    with st.spinner("Đang tải..."):
        df = get_data(COINS[coin], days)
        if df.empty: return

    st.subheader(f"📌 {coin.upper()} ({datetime.today().strftime('%Y-%m-%d')})")
    st.write(f"💰 Giá: ${df['price'].iloc[-1]:,.2f}")
    st.write(f"🏦 Market Cap: ${df['market_cap'].iloc[-1]:,.0f}")
    st.write(f"Khối lượng (% MC): {df['volume_percent_mc'].iloc[-1]:.2f}%")
    st.write(f"Dòng tiền vào: ${df['inflow'].sum():,.0f}")
    st.write(f"Dòng tiền ra: ${df['outflow'].sum():,.0f}")
    if days >= 30: st.write(f"Volume 7/30 ngày: {(df.tail(7)['volume'].mean() / df['volume'].mean()):.2f}")

    fig = make_subplots(4, 1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Giá", "Market Cap", "RSI & MACD", "Volume % MC"),
                        row_heights=[0.3, 0.3, 0.2, 0.2])
    fig.add_trace(go.Scattergl(x=df["time"], y=df["price"], line=dict(color="#00ff00")), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["market_cap"], line=dict(color="#ffa500")), row=2, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["RSI"], line=dict(color="#ff00ff")), row=3, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["MACD"], line=dict(color="#0000ff")), row=3, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["MACD_signal"], line=dict(color="#ff0000")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["time"], y=df["volume_percent_mc"], marker_color="#00b7eb"), row=4, col=1)
    fig.update_layout(height=800, template="plotly_dark", title=f"{coin} ({days} ngày)", hovermode="x unified")
    st.plotly_chart(fig)

    signals = find_signals(df)
    if not signals.empty: st.dataframe(signals, height=200)

if __name__ == "__main__":
    main()
