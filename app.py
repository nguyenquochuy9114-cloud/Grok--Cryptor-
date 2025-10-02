import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Cache dữ liệu để tăng hiệu suất
@st.cache_data(ttl=300)  # Lưu cache trong 5 phút
def get_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false&sparkline=false"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Lỗi API: {data['error']}")
        market_data = data.get("market_data", {})
        return {
            "name": data.get("name", "N/A"),
            "symbol": data.get("symbol", "N/A").upper(),
            "price": market_data.get("current_price", {}).get("usd", None),
            "market_cap": market_data.get("market_cap", {}).get("usd", None),
            "volume": market_data.get("total_volume", {}).get("usd", None),
            "market_cap_change_percentage_30d": market_data.get("market_cap_change_percentage_30d", None),
            "rsi": None
        }
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu coin: {e}. Kiểm tra https://docs.coingecko.com/")
        return None

@st.cache_data(ttl=300)
def get_historical_data(coin_id, days):
    interval = "hourly" if days <= 7 else "daily"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval={interval}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Lỗi API: {data['error']}")
        
        prices = pd.DataFrame(data["prices"], columns=["time", "price"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["time", "volume"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["time", "market_cap"])
        
        df = prices.merge(volumes, on="time").merge(market_caps, on="time")
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["volume_percent_mc"] = (df["volume"] / df["market_cap"]) * 100
        
        # Giảm số lượng dữ liệu nếu quá lớn để tối ưu biểu đồ
        if len(df) > 1000:
            if days > 7:
                df = df.resample("D", on="time").agg({
                    "price": "mean",
                    "volume": "sum",
                    "market_cap": "mean",
                    "volume_percent_mc": "mean"
                }).reset_index()
            else:
                n = len(df) // 1000 + 1
                df = df.iloc[::n]
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu lịch sử: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    st.set_page_config(page_title="Crypto Analyzer App", layout="wide")
    st.title("📊 Crypto Analyzer App")
    st.write("Phân tích giá, khối lượng, và tín hiệu giao dịch bằng RSI")

    selected_coin = st.selectbox("Chọn coin:", ["bitcoin", "ethereum", "binancecoin", "solana", "ripple"])
    days = st.slider("⏳ Chọn khoảng thời gian (ngày):", min_value=1, max_value=365, value=30, step=1)

    coin_data = get_coin_data(selected_coin)
    if not coin_data:
        return

    st.subheader(f"📌 Phân tích {coin_data['name'].upper()} ({datetime.today().strftime('%Y-%m-%d')})")
    if coin_data["price"]:
        st.write(f"💰 Giá hiện tại: **${coin_data['price']:,.2f}**")
    if coin_data["market_cap"]:
        st.write(f"🏦 Market Cap hiện tại: **${coin_data['market_cap']:,.0f}**")
    if coin_data["market_cap_change_percentage_30d"]:
        st.write(f"% Thay đổi Market Cap (30 ngày): **{round(coin_data['market_cap_change_percentage_30d'], 2)}%**")
    if coin_data["volume"] and coin_data["market_cap"]:
        vol_ratio = (coin_data["volume"] / coin_data["market_cap"]) * 100
        st.write(f"Khối lượng giao dịch (% Market Cap): **{round(vol_ratio, 2)}%**")
    if coin_data["volume"]:
        st.write(f"Dòng tiền (Inflow proxy): **${coin_data['volume']:,.0f}**")

    df = get_historical_data(selected_coin, days)
    if df.empty:
        st.error("Không có dữ liệu lịch sử để hiển thị.")
        return

    df["RSI"] = calculate_rsi(df["price"])
    coin_data["rsi"] = df["RSI"].iloc[-1] if not df["RSI"].empty else None

    if coin_data["rsi"] is not None:
        st.write(f"RSI (14): **{round(coin_data['rsi'], 2)}**")
        if coin_data["rsi"] < 30:
            st.success("📈 Tín hiệu Trade: Mua")
        elif coin_data["rsi"] > 70:
            st.error("📉 Tín hiệu Trade: Bán")
        else:
            st.info("⏸️ Tín hiệu Trade: Giữ")

    st.subheader("📊 Biểu đồ kỹ thuật")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Giá", "RSI", "Khối lượng (% Market Cap)"),
                        row_heights=[0.5, 0.25, 0.25])

    fig.add_trace(go.Scattergl(x=df["time"], y=df["price"], mode="lines", name="Giá", line=dict(color="#00ff00")), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["RSI"], mode="lines", name="RSI", line=dict(color="#ff00ff")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=df["time"], y=df["volume_percent_mc"], name="Volume (% MC)", marker_color="#00b7eb"), row=3, col=1)

    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False, template="plotly_dark",
                      title=f"Biểu đồ {coin_data['name']} ({days} ngày)", hovermode="x unified", dragmode="pan",
                      uirevision="static")
    fig.update_xaxes(title_text="Thời gian", row=3, col=1)
    fig.update_yaxes(title_text="Giá (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume (% MC)", row=3, col=1)

    config = {"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.subheader("📌 Gợi ý tín hiệu (RSI)")
    order = st.radio("Sắp xếp dữ liệu:", ["Mới → Cũ", "Cũ → Mới"])

    df_table = df[["time", "price", "RSI", "volume_percent_mc"]].copy()
    df_table["Signal"] = df_table["RSI"].apply(lambda x: "Mua" if x < 30 else ("Bán" if x > 70 else "Giữ"))
    df_table["price"] = df_table["price"].round(2)
    df_table["RSI"] = df_table["RSI"].round(2)
    df_table["volume_percent_mc"] = df_table["volume_percent_mc"].round(2)

    if order == "Mới → Cũ":
        df_table = df_table.sort_values("time", ascending=False)
    else:
        df_table = df_table.sort_values("time", ascending=True)

    st.dataframe(df_table.head(50), height=300, use_container_width=True)

if __name__ == "__main__":
    main()
