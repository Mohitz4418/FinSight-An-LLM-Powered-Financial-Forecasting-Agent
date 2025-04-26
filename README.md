# FinSight An LLM Powered Financial Forecasting Agent

FinSight is an advanced financial forecasting agent that harnesses the power of Large Language Models (LLMs) to deliver automated, data-driven insights for stock market analysis. An intelligent agent utilizing Large Language Models (LLMs) for automated financial news retrieval and stock price prediction.

LLM based Finance Agent is a powerful tool that leverages large language models (LLMs) to automatically fetch news and predict historical stock prices to forecast future prices. This repository is designed to provide financial insights using state-of-the-art natural language processing (NLP) and machine learning techniques.

This project is a stock price prediction web app powered by Gemini AI that forecasts future stock values using historical market data from Yahoo Finance and financial news headlines from NewsAPI. The Streamlit-based application provides next-day price predictions and allows backtesting with performance metrics including RMSE, MAE, and R-squared. It features clean, professional visualizations built with Matplotlib to compare predicted versus actual prices. The tool leverages Google's gemini-2.0-flash-lite model for AI analysis while maintaining a simple, user-friendly interface. Designed for quick market insights, it requires API keys for Gemini and NewsAPI to function. Built with Python, the application combines financial data analysis with generative AI capabilities in an accessible web format. Note that this is an analytical tool only and not intended as financial advice.

## Configuration

Configure the agent by editing the `config.json` file with your API keys and desired settings:
```json
{
    "news_api_key": "your_news_api_key",
    "genai_api_key": "your_genai_api_key",
    "model_name": "gemini-2.0-flash-lite",
    "stock_symbol": "2330.tw",
    "days": 30
}
```

- `news_api_key`: Your API key for the news data provider (Apply [here](https://newsapi.org/)).
- `genai_api_key`: Your API key for Google Generative AI (Apply [here](https://aistudio.google.com/app/u/1/apikey?hl=zh-tw)).
- `model_name`: The name of the Google Generative AI model to be used.
- `stock_symbol`: The stock symbol to analyze.
- `days`: The number of days to consider for the analysis.

## Usage

1. Ensure that you have configured the config.json file as described in the [Configuration](#configuration) section.

2. Run the project using the following command:
    ```python
    python app.py
    ```
