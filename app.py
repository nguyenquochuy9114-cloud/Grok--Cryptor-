import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Danh sách 500 coin (mẫu 50 coin, có thể mở rộng qua API CoinGecko)
COINS = {
    "Bitcoin": "bitcoin", "Ethereum": "ethereum", "BNB": "binancecoin", "Solana": "solana", "XRP": "ripple",
    "Tether": "tether", "USD Coin": "usd-coin", "Dogecoin": "dogecoin", "Cardano": "cardano", "TRON": "tron",
    "Avalanche": "avalanche-2", "Chainlink": "chainlink", "Polygon": "matic-network", "Polkadot": "polkadot",
    "Litecoin": "litecoin", "Shiba Inu": "shiba-inu", "Dai": "dai", "Uniswap": "uniswap", "Cosmos": "cosmos",
    "Stellar": "stellar", "VeChain": "vechain", "Hedera": "hedera-hashgraph", "Tezos": "tezos", "Algorand": "algorand",
    "Filecoin": "filecoin", "Internet Computer": "internet-computer", "Aptos": "aptos", "NEAR Protocol": "near",
    "Monero": "monero", "VeChain": "vechain", "Theta Network": "theta-token", "Arweave": "arweave",
    "Fantom": "fantom", "Elrond": "elrond-erd-2", "Algorand": "algorand", "Hedera": "hedera-hashgraph",
    "Cosmos Hub": "cosmos", "Cronos": "crypto-com-chain", "VeChain": "vechain", "Tezos": "tezos",
    "Algorand": "algorand", "Filecoin": "filecoin", "Internet Computer": "internet-computer", "Aptos": "aptos",
    "NEAR Protocol": "near", "Monero": "monero", "VeChain": "vechain", "Theta Network": "theta-token",
    "Arweave": "arweave", "Fantom": "fantom", "Elrond": "elrond-erd-2"
}

@st.cache_data(ttl=600)  # Cache 10 phút để giảm tải API
def fetch_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = (pd.DataFrame(data["prices"], columns=["time", "price"])
              .merge(pd.DataFrame(data["total_volumes"], columns=["time", "volume"]), on="time")
              .merge(pd.DataFrame(data["market_caps"], columns=["time", "market_cap"]), on="time"))
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["volume_percent_mc"] = (df["volume"] / df["market_cap"]) * 100
        df["price_change"] = df["price"].pct_change()
        df["inflow"] = np.where(df["price_change"] > 0, df["volume"] * df["price_change"], 0)
        df["outflow"] = np.where(df["price_change"] < 0, df["volume"] * abs(df["price_change"]), 0)
        if len(df) > 300:  # Downsample để tối ưu
            df = df.resample("D", on="time").agg({
                "price": "last", "volume": "sum", "market_cap": "last",
                "volume_percent_mc": "mean", "inflow": "sum", "outflow": "sum"
            }).reset_index()
        return df
    except Exception as e:
        st.error(f"Lỗi API: {e}. Kiểm tra https://docs.coingecko.com/")
        return pd.DataFrame()

@st.cache_data
def calculate_indicators(df):
    df["RSI"] = 100 - (100 / (1 + df["price"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                             df["price"].diff().where(lambda x: x < 0, -x).rolling(14).mean()))
    df["MACD"] = df["price"].ewm(span=12).mean() - df["price"].ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    return df

def get_signals(df):
    buy = df[(df["RSI"] < 30) & (df["MACD"] > df["MACD_signal"]) & 
             (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))]
    sell = df[(df["RSI"] > 70) & (df["MACD"] < df["MACD_signal"]) & 
              (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))]
    signals = pd.concat([buy, sell])
    return signals[["time", "price", "RSI"]].assign(Signal=lambda x: np.where(x.index.isin(buy.index), "Buy", "Sell"))

def main():
    st.set_page_config(page_title="Crypto Analyzer", layout="wide")
    st.title("📊 Crypto Analyzer")
    st.write("Phân tích giá, khối lượng, dòng tiền, và tín hiệu giao dịch")

    coin = st.selectbox("Chọn coin", options=list(COINS.keys()))
    days = st.slider("Chọn khoảng thời gian (ngày)", 1, 365, 30)

    with st.spinner("Đang tải dữ liệu..."):
        df = fetch_data(COINS[coin], days)
        if df.empty:
            st.error("Không có dữ liệu để hiển thị.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)

    st.subheader(f"📌 {coin} ({datetime.today().strftime('%Y-%m-%d %H:%M')})")
    st.write(f"💰 Giá: ${df['price'].iloc[-1]:,.2f}")
    st.write(f"🏦 Market Cap: ${df['market_cap'].iloc[-1]:,.0f}")
    st.write(f"Khối lượng (% Market Cap): {df['volume_percent_mc'].iloc[-1]:.2f}%")
    st.write(f"Dòng tiền vào: ${df['inflow'].sum():,.0f}")
    st.write(f"Dòng tiền ra: ${df['outflow'].sum():,.0f}")
    if days >= 30:
        vol_7d = df.tail(7)["volume"].mean() if len(df) >= 7 else 0
        vol_30d = df["volume"].mean()
        st.write(f"Tỷ lệ Volume 7/30 ngày: {vol_7d / vol_30d if vol_30d > 0 else 0:.2f}")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Giá", "Market Cap", "RSI & MACD", "Volume % MC"),
                        row_heights=[0.3, 0.3, 0.2, 0.2])
    fig.add_trace(go.Scattergl(x=df["time"], y=df["price"], name="Giá", line=dict(color="#00ff00")), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["market_cap"], name="Market Cap", line=dict(color="#ffa500")), row=2, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["RSI"], name="RSI", line=dict(color="#ff00ff")), row=3, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["MACD"], name="MACD", line=dict(color="#0000ff")), row=3, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["MACD_signal"], name="MACD Signal", line=dict(color="#ff0000")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["time"], y=df["volume_percent_mc"], name="Volume % MC", marker_color="#00b7eb"), row=4, col=1)
    fig.update_layout(height=800, template="plotly_dark", title=f"{coin} ({days} ngày)", hovermode="x unified")
    st.plotly_chart(fig)

    st.subheader("📌 Tín hiệu Buy/Sell")
    if not signals.empty:
        st.dataframe(signals, height=200, use_container_width=True)
    else:
        st.info("Không có tín hiệu trong khoảng thời gian này.")

if __name__ == "__main__":
    main()
