# Overview

Advanced Forex Analysis Platform is a sophisticated financial trading application built with Streamlit that provides real-time forex market analysis, predictive modeling, and technical indicators. The platform combines machine learning capabilities with traditional technical analysis to offer comprehensive trading insights across multiple currency pairs. It features an LSTM-based neural network for price prediction, sentiment analysis of market news, and advanced visualization tools for traders and analysts.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with custom CSS styling
- **UI Design**: Professional gradient styling with metric cards and responsive layout
- **Visualization**: Plotly for interactive charts and Matplotlib for static plots
- **State Management**: Streamlit session state for caching and user interactions

## Backend Architecture
- **Core Framework**: Python-based modular architecture with service-oriented design
- **Data Processing**: Pandas and NumPy for data manipulation and analysis
- **Machine Learning**: PyTorch-based LSTM neural network with attention mechanisms
- **API Integration**: RESTful API consumption from Alpha Vantage for real-time forex data

## Data Storage Solutions
- **Caching Strategy**: In-memory caching using Streamlit session state
- **Cache Management**: TTL-based cache invalidation with configurable expiration times
- **Data Sources**: Alpha Vantage API for forex prices and news sentiment data

## Authentication and Authorization
- **API Security**: Environment variable-based API key management
- **Access Control**: No user authentication system implemented (single-user application)

## Key Components

### Machine Learning Pipeline
- **Model**: Enhanced LSTM with bidirectional layers and attention mechanism
- **Features**: Multi-feature input including OHLC prices and technical indicators
- **Training**: Configurable batch size, learning rate, and epoch parameters
- **Prediction**: Realistic short-term forecasting (maximum 7 days) with confluent confidence system
- **Confidence System**: Combined LSTM (40%), AI (30%), sentiment (20%), and consistency (10%) weights
- **Unified Analysis**: Position, Swing, and Intraday trading use same realistic probability calculations and risk management approach
- **Advanced Money Management**: Complete lot size calculation, drawdown prediction in USD, extension probability values, and ROI analysis

### Technical Analysis Engine
- **Indicators**: Comprehensive set including SMA, EMA, RSI, MACD, Bollinger Bands, and Stochastic
- **Signal Generation**: Automated buy/sell signal generation based on indicator crossovers
- **Risk Management**: Configurable risk levels (Conservative, Moderate, Aggressive)
- **Position Trading**: Unified realistic conditions aligned with Swing and Intraday analyses

### Sentiment Analysis System
- **News Processing**: Real-time news sentiment analysis using VADER sentiment analyzer
- **Data Sources**: Financial news feeds from Alpha Vantage
- **Scoring**: Compound sentiment scores with weighted averaging

### Visualization Framework
- **Interactive Charts**: Candlestick charts with technical indicator overlays
- **Multi-panel Displays**: Subplots for price, RSI, and MACD indicators
- **Responsive Design**: Dynamic chart sizing and color coding

## Configuration Management
- **Settings**: Centralized configuration in `config/settings.py`
- **Parameters**: Configurable trading pairs, intervals, model parameters, and cache settings
- **Trader Profiles**: Real market-based parameters for Conservative, Moderate, and Aggressive profiles
- **Temporal Strategy**: Market-realistic parameters for each timeframe (5min to 1 month) based on actual trading statistics
- **Environment**: Environment variable support for sensitive data

# External Dependencies

## APIs and Web Services
- **Alpha Vantage API**: Primary data source for real-time forex prices and financial news
- **Rate Limiting**: Built-in handling for API quotas and rate limits

## Python Libraries
- **Web Framework**: Streamlit for web application interface
- **Data Science**: Pandas, NumPy for data manipulation
- **Machine Learning**: PyTorch for deep learning models, scikit-learn for preprocessing
- **Visualization**: Plotly for interactive charts, Matplotlib for static plots
- **Technical Analysis**: Custom implementation of technical indicators
- **Sentiment Analysis**: VADER Sentiment Analyzer for news sentiment processing
- **HTTP Requests**: Requests library for API communication

## Model Dependencies
- **PyTorch**: Deep learning framework for LSTM neural networks
- **scikit-learn**: Data preprocessing and model evaluation metrics
- **MinMaxScaler**: Data normalization for neural network training

## Development Tools
- **Environment Management**: Support for environment variables
- **Error Handling**: Comprehensive exception handling for API failures and data issues
- **Logging**: Built-in error reporting and status messaging