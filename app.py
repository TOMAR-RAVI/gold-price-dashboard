import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
# Page configuration
st.set_page_config(
    
    page_title="GoldPulse Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Inject Google Fonts and custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora&family=Montserrat:wght@600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lora', serif;
        color: #333;
    }

    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif;
        color: #DAA520;
        font-weight: 600;
    }

    .stButton>button {
        font-family: 'Montserrat', sans-serif;
        font-weight: bold;
    }

    .stTabs [role="tab"] {
        font-family: 'Montserrat', sans-serif;
    }

    .markdown-text-container {
        font-family: 'Lora', serif;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Base layout */
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #DAA520; /* Goldenrod */
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fffbe6;
        border-right: 1px solid #e0e0e0;
    }

    /* Buttons */
    .stButton>button {
        background-color: #DAA520;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }

    /* Tabs */
    .stTabs [role="tab"] {
        background-color: #f0f0f0;
        color: #DAA520;
        border-radius: 5px;
        margin-right: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DAA520;
        color: white;
    }

    /* Metrics */
    .stMetric {
        background-color: #fffbe6;
        border-radius: 10px;
        padding: 10px;
        color: #DAA520;
    }

    /* DataFrames */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
    }

    /* Markdown blocks */
    .markdown-text-container {
        color: #333;
    }

    /* Sliders */
    .stSlider .css-1cpxqw2 {
        color: #DAA520;
    }
    </style>
""", unsafe_allow_html=True)



# Constants
GRAMS_PER_OUNCE = 31.1
GRAMS_PER_TAEL = 37.7994
GRAMS_PER_PENNYWEIGHT = GRAMS_PER_OUNCE / 20

# Sidebar: Currency and Unit Selection
st.sidebar.header("🌐 Currency & Unit Settings")
currency = st.sidebar.selectbox("Select Currency", ["INR", "USD", "AUD", "EUR"])
unit = st.sidebar.selectbox("Select Unit", [ "Ounce", "Gram", "Pennyweight", "Tael"])

# Currency conversion tickers
currency_map = {
    "INR": "USDINR=X",
    "AUD": "USDAUD=X",
    "EUR": "USDEUR=X",
    "USD": None
}

# Fetch gold price data
live_data = yf.download('GC=F', period='1d', interval='1m')
if not live_data.empty and 'Close' in live_data.columns and not live_data['Close'].dropna().empty:
    latest_usd = float(live_data['Close'].dropna().iloc[-1])
    open_usd = float(live_data['Open'].dropna().iloc[-1])
    low_usd = float(live_data['Low'].dropna().min())
    high_usd = float(live_data['High'].dropna().max())
else:
    fallback_data = yf.download('GC=F', period='2d', interval='1d')
    if not fallback_data.empty and 'Close' in fallback_data.columns and not fallback_data['Close'].dropna().empty:
        latest_usd = float(fallback_data['Close'].dropna().iloc[-1])
        open_usd = float(fallback_data['Open'].dropna().iloc[-1])
        low_usd = float(fallback_data['Low'].dropna().min())
        high_usd = float(fallback_data['High'].dropna().max())
    else:
        st.error("⚠️ Gold price data is currently unavailable. Please try again later.")
        st.stop()

# Fetch exchange rate
conversion_rate = 1.0
if currency != "USD":
    fx_data = yf.download(currency_map[currency], period='1d', interval='1m')
    if fx_data.empty or 'Close' not in fx_data.columns or fx_data['Close'].dropna().empty:
        st.error(f"⚠️ Exchange rate for {currency} is currently unavailable.")
        st.stop()
    conversion_rate = float(fx_data['Close'].dropna().iloc[-1])

# Unit conversion map
unit_map = {
    "Ounce": 1,
    "Gram": 1 / GRAMS_PER_OUNCE,
    "Pennyweight": 1 / 20,
    "Tael": GRAMS_PER_TAEL / GRAMS_PER_OUNCE
}

# Convert gold price to selected currency and unit
price_per_ounce = latest_usd * conversion_rate
price_in_selected_unit = price_per_ounce * unit_map[unit]
change = (latest_usd - open_usd) * conversion_rate * unit_map[unit]
change_percent = (change / (open_usd * conversion_rate * unit_map[unit])) * 100
low = low_usd * conversion_rate * unit_map[unit]
high = high_usd * conversion_rate * unit_map[unit]
delta_icon = "🟢" if change > 0 else "🔴"

# 🔔 Price Alert Input
st.sidebar.header("🔔 Price Alert")
target_price = st.sidebar.number_input(f"Set alert for 1 {unit} in {currency}", min_value=0.0, value=price_in_selected_unit)
alert_triggered = price_in_selected_unit >= target_price

if alert_triggered:
    st.sidebar.success(f"✅ Alert: Gold price has reached {currency} {price_in_selected_unit:,.2f}")
else:
    st.sidebar.info(f"ℹ️ Current price is below your alert of {currency} {target_price:,.2f}")

# Page layout
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")
st.title(f"🌍 Gold Price Dashboard ({currency} per {unit})")

# Tabs
tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["📊 Live Price", "📈 Chart", "📅 Summary", "🔍 Trends", "🔮 Forecast","📊 Technical Analysis"])

# Tab 1: Live Price
with tab1:
    # 💰 Live Price Section
    with st.container():
        st.markdown("### 💰 Live Gold Price")
        st.metric(
            label=f"💰 Price (1 {unit})",
            value=f"{currency} {price_in_selected_unit:,.2f}",
            delta=f"{delta_icon} {currency} {change:,.2f} ({change_percent:.2f}%)"
        )
        st.divider()

    # 📉 Day's Range Section
    with st.container():
        st.markdown("### 📉 Day's Range")
        col1, col2 = st.columns(2)
        col1.metric("🔻 Low", f"{currency} {low:,.2f}")
        col2.metric("🔺 High", f"{currency} {high:,.2f}")
        st.divider()

    # 🔔 Price Alert Section
    with st.container():
        st.markdown("### 🔔 Price Alert Status")
        if alert_triggered:
            st.success(f"✅ Alert triggered: Gold price is {currency} {price_in_selected_unit:,.2f}")
        else:
            st.info(f"ℹ️ Current price is below your alert of {currency} {target_price:,.2f}")


# Tab 2: Chart
with tab2:
    st.subheader(f"📈 Gold Price Chart ({currency} per {unit})")

    # Download and prepare long-term data
    df = yf.download('GC=F', start='2020-01-01', end='2025-09-20')
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    df['Converted_Price'] = (df['Close'] * conversion_rate) * unit_map[unit]

    # Sidebar date filter
    st.sidebar.header("📅 Select Date Range")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # 📍 User-selected highlight date
    selected_date = st.date_input("📍 Highlight Date", value=pd.to_datetime("2020-04-25"))
    key_date = pd.to_datetime(selected_date)
    key_value = filtered_df.loc[filtered_df['Date'] == key_date, 'Converted_Price'].values

    # Plot chart
    fig = px.line(
        filtered_df,
        x='Date',
        y='Converted_Price',
        title=f"Gold Price Trend ({currency} per {unit})",
        labels={'Date': 'Date', 'Converted_Price': f'Price ({currency})'}
    )

    # Add vertical marker and annotation
    if len(key_value) > 0:
        fig.add_shape(
            type="line",
            x0=key_date,
            x1=key_date,
            y0=filtered_df['Converted_Price'].min(),
            y1=filtered_df['Converted_Price'].max(),
            line=dict(color="gray", dash="dash"),
        )
        fig.add_annotation(
            x=key_date,
            y=filtered_df['Converted_Price'].max(),
            text=f"{key_date.date()} → {currency} {key_value[0]:,.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="white",
            bordercolor="gray",
            font=dict(size=12)
        )

    # Style enhancements
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        xaxis=dict(tickformat='%b %Y', tickangle=45),
        yaxis=dict(title=f"{currency} per {unit}"),
        hovermode='x unified'
    )
    fig.update_traces(hovertemplate=f"{currency} %{{y:.2f}} on %{{x|%Y-%m-%d}}")
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Performance Metrics
    st.markdown("### 📊 Performance Summary")
    latest_price = filtered_df['Converted_Price'].iloc[-1]

    def get_change(days):
        past_date = filtered_df['Date'].iloc[-1] - pd.Timedelta(days=days)
        past_df = filtered_df[filtered_df['Date'] <= past_date]
        if not past_df.empty:
            past_price = past_df['Converted_Price'].iloc[-1]
            change_pct = ((latest_price - past_price) / past_price) * 100
            return f"{change_pct:.2f}%"
        return "N/A"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Today", get_change(1))
    col2.metric("This Week", get_change(7))
    col3.metric("1 Month", get_change(30))
    col4.metric("3 Months", get_change(90))
    col5.metric("YTD", get_change(270))
    col6.metric("1 Year", get_change(365))


# Tab 3: Summary
with tab3:
    st.subheader(f"📅 Summary Statistics ({currency} per {unit})")
    highest = filtered_df['Converted_Price'].max()
    lowest = filtered_df['Converted_Price'].min()
    average = filtered_df['Converted_Price'].mean()
    days = filtered_df['Date'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Highest Price", f"{currency} {highest:,.2f}")
    col2.metric("📉 Lowest Price", f"{currency} {lowest:,.2f}")
    col3.metric("📊 Average Price", f"{currency} {average:,.2f}")
    col4.metric("📅 Trading Days", f"{days}")

# Tab 4: Trends
with tab4:
    st.subheader("📊 Historical Comparison")
    timeframes = {
        "7 days ago": "7d",
        "30 days ago": "30d",
        "365 days ago": "365d"
    }

    comparison_data = []
    for label, period in timeframes.items():
        past_data = yf.download('GC=F', period=period, interval='1d')
        if not past_data.empty and 'Close' in past_data.columns:
            past_price_usd = float(past_data['Close'].dropna().iloc[0])
            past_price_converted = past_price_usd * conversion_rate * unit_map[unit]
            price_change = price_in_selected_unit - past_price_converted
            percent_change = (price_change / past_price_converted) * 100
            comparison_data.append((label, past_price_converted, price_change, percent_change))

    col1, col2, col3 = st.columns(3)
    for i, (label, past_price, change_amt, change_pct) in enumerate(comparison_data):
        col = [col1, col2, col3][i]
        icon = "🟢" if change_amt > 0 else "🔴"
        col.metric(label=label, value=f"{currency} {past_price:,.2f}", delta=f"{icon} {currency} {change_amt:,.2f} ({change_pct:.2f}%)")

from prophet import Prophet
from datetime import datetime
with tab5:
    st.subheader("🔮 Gold Price Forecast")

    # Forecast horizon control
    forecast_days = st.slider(
        "Forecast Days",
        min_value=7,
        max_value=60,
        value=30,
        key="forecast_slider"
    )

    # Today's date
    today = datetime.today().date()

    # Fetch FX data safely
    fx_data = yf.download('USDINR=X', start='2023-01-01', end=today)
    try:
        conversion_rate = float(fx_data['Close'].dropna().iloc[-1])
    except (IndexError, KeyError, TypeError, ValueError):
        st.error("⚠️ Currency data unavailable. Using default conversion rate.")
        conversion_rate = 1.0

    # Fetch gold data
    gold_data = yf.download('GC=F', start='2023-01-01', end=today)
    gold_data.reset_index(inplace=True)
    gold_data['Converted_Price'] = gold_data['Close'] * conversion_rate * unit_map[unit]
    gold_data = gold_data[['Date', 'Converted_Price']].dropna()
    gold_data.rename(columns={'Date': 'ds', 'Converted_Price': 'y'}, inplace=True)

    # Check if gold data is sufficient
    if len(gold_data) < 30:
        st.error("⚠️ Not enough gold price data to train forecast model.")
    else:
        # Train Prophet model
        model = Prophet()
        model.fit(gold_data)

        # Forecast future
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Check forecast output
        if forecast.empty:
            st.error("⚠️ Forecast data is empty. Unable to generate prediction.")
        else:
            final_day = forecast.iloc[-1]
            predicted_price = final_day['yhat']
            predicted_date = final_day['ds'].date()

            # Display forecasted price
            st.metric(
                label=f"📅 Predicted Gold Price on {predicted_date}",
                value=f"{predicted_price:,.2f} {currency} per {unit}"
            )

    # Fetch live gold price using Ticker.info
    gold_info = yf.Ticker("GC=F").info
    live_price_raw = gold_info.get("regularMarketPrice", None)

    if live_price_raw:
        live_price = live_price_raw * conversion_rate * unit_map[unit]
        st.metric("📌 Latest Gold Price", f"{live_price:,.2f} {currency} per {unit}")
    else:
        st.warning("Live gold price not available.")

    # Optional: Debug preview
    # st.write("Gold data preview:", gold_data.tail())
    # st.write("Forecast preview:", forecast.tail())




with tab6:
    import pandas_ta as ta

    st.subheader("📊 Multi-Timeframe Technical Analysis")

    # Download gold data
    df = yf.download('GC=F', start='2023-01-01', end='2025-09-20', interval='1d')
    df.reset_index(inplace=True)
    df['Converted_Price'] = (df['Close'] * conversion_rate) * unit_map[unit]

    # Add indicators
    df['RSI'] = ta.rsi(df['Converted_Price'], length=14)
    macd = ta.macd(df['Converted_Price'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']
    df['EMA_20'] = ta.ema(df['Converted_Price'], length=20)
    df['EMA_50'] = ta.ema(df['Converted_Price'], length=50)

    # Extract latest values
    rsi = df['RSI'].iloc[-1]
    macd_val = df['MACD'].iloc[-1]
    signal_val = df['Signal'].iloc[-1]
    ema20 = df['EMA_20'].iloc[-1]
    ema50 = df['EMA_50'].iloc[-1]
    price = df['Converted_Price'].iloc[-1]

    # Signal logic
    rsi_signal = "Bullish" if rsi > 50 else "Bearish"
    macd_signal = "Bullish" if macd_val > signal_val else "Bearish"
    ema_signal = "Bullish" if ema20 > ema50 else "Bearish"

    # Summary sentiment
    signals = [rsi_signal, macd_signal, ema_signal]
    bullish_count = signals.count("Bullish")
    sentiment = (
        "Strong Buy" if bullish_count == 3 else
        "Buy" if bullish_count == 2 else
        "Neutral" if bullish_count == 1 else
        "Bearish"
    )

    # Display gauges
    st.markdown("### 📈 Summary")
    st.success(f"🟢 Overall Signal: **{sentiment}**")

    st.markdown("### 📊 Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{rsi:.2f}", rsi_signal)
    col2.metric("MACD", f"{macd_val:.2f}", macd_signal)
    col3.metric("EMA Trend", f"{ema20:.2f} vs {ema50:.2f}", ema_signal)

    # Timeframe signals (simulated for now)
    st.markdown("### ⏱️ Timeframe Signals")
    timeframes = {
        "Hourly": sentiment,
        "Daily": sentiment,
        "Weekly": sentiment,
        "Monthly": sentiment
    }
    for tf, sig in timeframes.items():
        st.markdown(f"**{tf}:** 🟢 {sig}")
