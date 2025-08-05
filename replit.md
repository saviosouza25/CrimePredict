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
- **Confidence System**: Balanced 25% weights for each component (Technical, Trend, Volume, Sentiment)
- **Unified Analysis**: Consensus-based decision making with transparent component breakdown
- **Normalized Components**: All signals normalized to prevent single-component dominance
- **Consensus Logic**: Decision based on majority of components (3+ for strong signals)
- **Simplified Money Management**: Manual bank value and lot selection with dollar-based drawdown and extension calculations

### Technical Analysis Engine
- **Indicators**: Comprehensive set including SMA, EMA, RSI, MACD, Bollinger Bands, and Stochastic
- **Signal Generation**: Automated buy/sell signal generation based on indicator crossovers
- **Manual Trading Setup**: Simple bank value and lot size selection in sidebar
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
- **Trading Configuration**: Manual bank value (USD) and lot size selection in sidebar
- **Risk Calculation**: Simple dollar-based drawdown and maximum extension calculations
- **Temporal Strategy**: Market-realistic parameters for each timeframe (5min to 1 month) based on actual trading statistics
- **Environment**: Environment variable support for sensitive data

## Recent Bug Fixes (August 2025)
- **Volume Analysis Consistency**: Fixed critical inconsistency where individual volume analysis showed "COMPRA" but unified analysis showed "VENDA" for same data
- **Root Cause**: Different calculation methods between analyses (volatility-based vs range-based)
- **Solution**: Standardized both to use identical volatility-based calculation with threshold 0.020
- **Prevention**: Added debug_volume_consistency.py tool for future validation

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