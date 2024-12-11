
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.graph_objects as go
import traceback

# Define top 50 stocks
top_50_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "V", "JPM",
    "JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX",
    "ADBE", "KO", "PEP", "NKE", "T", "PFE", "MRK", "INTC", "XOM", "CSCO",
    "ORCL", "CMCSA", "ABT", "CRM", "COST", "VZ", "ACN", "AMD", "UPS", "QCOM",
    "TXN", "AVGO", "LIN", "LOW", "HON", "AMGN", "CVX", "MDT", "PM", "NEE"
]

# Initialize session state
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        adjusted_ticker = ticker.replace(".", "-")  # Adjust special tickers like BRK.B
        stock_data = yf.download(adjusted_ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.warning(f"No data found for ticker '{ticker}', skipping.")
            return None
        return stock_data
    except Exception as e:
        st.warning(f"Error fetching data for ticker '{ticker}': {e}")
        return None

# Function to predict price movement
def predict_price_movement(data, days=7):
    if data is None or len(data) < 2:
        return None, None
    try:
        data['Days'] = np.arange(len(data))
        X = data[['Days']].values
        y = data['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
        predictions = model.predict(future_days)
        return data['Close'].iloc[-1], predictions[-1]
    except Exception as e:
        st.warning(f"Error during price prediction: {e}")
        return None, None

# Function to analyze options for a single stock
def analyze_single_stock(stock, current_price, predicted_price):
    st.subheader(f"Options Recommendations for {stock.info.get('shortName', 'Unknown Stock')}")
    try:
        if not stock.options:
            st.warning(f"No options data available for {stock.info.get('shortName', 'Unknown Stock')}.")
            return

        expiry_dates = stock.options
        expiry = expiry_dates[0]  # Use the nearest expiry date
        options_chain = stock.option_chain(expiry)
        calls = options_chain.calls
        puts = options_chain.puts

        relevant_calls = calls[(calls['strike'] >= current_price * 0.95) & (calls['strike'] <= current_price * 1.05)]
        relevant_puts = puts[(puts['strike'] >= current_price * 0.95) & (puts['strike'] <= current_price * 1.05)]

        st.write("Relevant Call Options")
        st.dataframe(relevant_calls)
        st.write("Relevant Put Options")
        st.dataframe(relevant_puts)

        if predicted_price > current_price * 1.05:
            st.success(
                f"Recommendation: BUY CALL OPTIONS - Predicted price is ${predicted_price:.2f}, up from ${current_price:.2f}."
            )
        elif predicted_price < current_price * 0.95:
            st.success(
                f"Recommendation: BUY PUT OPTIONS - Predicted price is ${predicted_price:.2f}, down from ${current_price:.2f}."
            )
        else:
            st.info("Recommendation: HOLD - Minimal movement predicted.")
    except Exception as e:
        st.warning(f"Error analyzing options: {e}")

# Function to automatically analyze top recommendations
def find_top_recommendations():
    st.subheader("Top Recommendations Across Top 50 Stocks")
    results = []

    for ticker in top_50_stocks:
        try:
            stock_data = fetch_stock_data(ticker, pd.to_datetime("2022-01-01"), date.today())
            if stock_data is None:
                continue

            current_price, predicted_price = predict_price_movement(stock_data)
            if current_price is None or predicted_price is None:
                continue

            movement = abs(predicted_price - current_price)
            results.append((ticker, current_price, predicted_price, movement))

        except Exception as e:
            st.warning(f"Error processing ticker '{ticker}': {e}")

    valid_results = [entry for entry in results if len(entry) == 4 and all(isinstance(x, (int, float)) for x in entry[1:])]
    if not valid_results:
        st.warning("No valid recommendations could be generated.")
        return

    sorted_results = sorted(valid_results, key=lambda x: x[3], reverse=True)[:5]

    for ticker, current_price, predicted_price, movement in sorted_results:
        try:
            st.write(f"**{ticker}**: Predicted Price: ${predicted_price:.2f}, Current Price: ${current_price:.2f}, Movement: ${movement:.2f}")
            stock = yf.Ticker(ticker.replace(".", "-"))
            analyze_single_stock(stock, current_price, predicted_price)
        except Exception as e:
            st.warning(f"Error analyzing options for ticker '{ticker}': {e}")

# Global Error Handler
try:
    # App layout with tabs
    tabs = st.tabs(["Stock Analysis", "Options Analysis", "Top Recommendations"])

    # Tab 1: Stock Analysis
    with tabs[0]:
        st.header("Stock Analysis")
        selected_stock = st.selectbox("Select a Stock", ["Analyze All"] + top_50_stocks)
        start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
        end_date = st.date_input("End Date", value=date.today())

        if st.button("Analyze Stock"):
            if selected_stock == "Analyze All":
                find_top_recommendations()
            else:
                st.session_state.selected_stock = selected_stock
                stock_data = fetch_stock_data(selected_stock, start_date, end_date)
                if stock_data is not None:
                    st.session_state.stock_data = stock_data
                    current_price, predicted_price = predict_price_movement(stock_data)
                    if current_price is not None and predicted_price is not None:
                        st.write(f"Current Price: ${current_price:.2f}")
                        st.write(f"Predicted Price: ${predicted_price:.2f}")
                        stock = yf.Ticker(selected_stock)
                        analyze_single_stock(stock, current_price, predicted_price)
                    else:
                        st.error("Unable to calculate predictions for the selected stock.")

    # Tab 2: Options Analysis
    with tabs[1]:
        st.header("Options Analysis")
        if st.session_state.selected_stock:
            stock = yf.Ticker(st.session_state.selected_stock.replace(".", "-"))
            stock_data = st.session_state.stock_data
            current_price, predicted_price = predict_price_movement(stock_data)
            if current_price is not None and predicted_price is not None:
                analyze_single_stock(stock, current_price, predicted_price)
            else:
                st.error("Unable to calculate predictions for the selected stock.")
        else:
            st.info("Select a stock in the Stock Analysis tab first.")

    # Tab 3: Top Recommendations
    with tabs[2]:
        if st.button("Top Recommendations"):
            find_top_recommendations()

except Exception as e:
    st.error("An unexpected error occurred. See details below:")
    st.text(traceback.format_exc())
