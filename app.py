import streamlit as st
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from utils import StockPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure page
st.set_page_config(
    page_title="FinSight - Finacial Forecasting Agent",
    page_icon="📈",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    try:
        with open('config.json') as f:
            config = json.load(f)
        return config
    except Exception as e:
        st.error(f"Failed to load config: {str(e)}")
        st.stop()

config = load_config()

# Initialize predictor
@st.cache_resource
def init_predictor(config):
    try:
        return StockPredictor(config)
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

predictor = init_predictor(config)

# Sidebar controls
st.sidebar.title("Configuration")
with st.sidebar:
    config['stock_symbol'] = st.text_input("Stock Symbol", value=config['stock_symbol'])
    config['days'] = st.number_input("Days of History", min_value=5, max_value=365, value=config['days'])
    
    today = datetime.now().date()
    start_date = st.date_input("Backtest Start", value=today - timedelta(days=30))
    end_date = st.date_input("Backtest End", value=today)
    
    predict_next = st.checkbox("Predict Next Day", value=True)
    run_backtest = st.checkbox("Run Backtest", value=False)
    show_raw = st.checkbox("Show Raw Data", value=False)

# Main content
st.title("FinSight : An LLM Powered Financial Forecasting Agent")
st.write("Using Gemini AI for stock price predictions")

# Prediction section
if predict_next:
    st.header("Next Trading Day Prediction")
    
    with st.spinner("Generating prediction..."):
        try:
            prediction = predictor.predict_next_day(verbose=True)
            
            # Create compact layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Predicted Closing Price",
                    value=f"${prediction:.2f}",
                    delta_color="off"
                )
                
                # Compact history table
                hist_data = predictor._get_historical_data(datetime.now())
                st.dataframe(
                    hist_data.tail(config['history_days_to_show'])[['Close']]
                    .style.format("{:.2f}"),
                    height=300,
                    use_container_width=True
                )
            
            with col2:
                st.pyplot(
                    predictor.plot_history(hist_data),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Backtesting section
if run_backtest and start_date < end_date:
    st.header("Backtesting Performance")
    
    with st.spinner(f"Running backtest from {start_date} to {end_date}..."):
        try:
            results = predictor.backtest(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                verbose=True
            )
            
            if not results.empty:
                # Metrics in compact cards
                st.subheader("Model Performance Metrics")
                cols = st.columns(4)
                metrics = {
                    'RMSE': np.sqrt(mean_squared_error(results['actual'], results['predicted'])),
                    'MAE': mean_absolute_error(results['actual'], results['predicted']),
                    'R²': r2_score(results['actual'], results['predicted']),
                    'MAPE': f"{np.mean(np.abs(results['pct_error'])):.2f}%"
                }
                
                for (name, value), col in zip(metrics.items(), cols):
                    with col:
                        st.metric(
                            label=name,
                            value=f"{float(value):.2f}" if name != 'MAPE' else value,
                            help=f"{name} metric"
                        )
                
                # Results chart
                st.subheader("Prediction Performance")
                st.pyplot(
                    predictor.plot_results(results),
                    use_container_width=True
                )
                
                # Compact results table
                with st.expander("View Detailed Results", expanded=False):
                    st.dataframe(
                        results.sort_values('date', ascending=False)
                        .style.format({
                            'actual': '{:.2f}',
                            'predicted': '{:.2f}',
                            'error': '{:.2f}',
                            'pct_error': '{:.2f}%'
                        }),
                        height=300,
                        use_container_width=True
                    )
                
            else:
                st.warning("No valid predictions were generated during backtest")
                
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
elif run_backtest:
    st.warning("End date must be after start date")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """ℹ️ Note: 
    - Predictions are AI-generated estimates
    - Not financial advice
    - Results may vary based on market conditions"""
)