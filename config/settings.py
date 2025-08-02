import os
from typing import Dict, List

# API Configuration
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'VZ6XL34A1G4VCKP3')

# Trading Pairs
PAIRS: List[str] = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'GBP/JPY', 'EUR/GBP']

# Time Intervals
INTERVALS: Dict[str, str] = {
    '1min': '1min',
    '5min': '5min', 
    '15min': '15min',
    '30min': '30min',
    '60min': '60min',
    'daily': 'daily'
}

# Prediction Horizons
HORIZONS: List[str] = ['5 Minutos', '15 Minutos', '1 Hora', '4 Horas', '1 Dia', '1 Semana', '1 Mês']
HORIZON_STEPS: Dict[str, int] = {
    '5 Minutos': 1,    # 1 step ahead for 5min intervals
    '15 Minutos': 1,   # 1 step ahead for 15min intervals  
    '1 Hora': 1,
    '4 Horas': 4, 
    '1 Dia': 24,
    '1 Semana': 168,
    '1 Mês': 720
}

# Model Parameters
LOOKBACK_PERIOD = 60
MC_SAMPLES = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# Cache Settings
CACHE_TTL = 300  # 5 minutes
DATA_CACHE_TTL = 900  # 15 minutes
NEWS_CACHE_TTL = 1800  # 30 minutes

# Risk Management
RISK_LEVELS = {
    'Conservative': 0.02,
    'Moderate': 0.05,
    'Aggressive': 0.10
}
