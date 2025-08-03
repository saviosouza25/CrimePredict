import os
from typing import Dict, List

# API Configuration
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Trading Pairs
PAIRS: List[str] = [
    # Major Currency Pairs (100% Alpha Vantage Compatible)
    'EUR/USD',  # Euro/US Dollar - Most liquid pair
    'USD/JPY',  # US Dollar/Japanese Yen - Second most liquid
    'GBP/USD',  # British Pound/US Dollar - Cable
    'AUD/USD',  # Australian Dollar/US Dollar - Aussie
    'USD/CAD',  # US Dollar/Canadian Dollar - Loonie
    'USD/CHF',  # US Dollar/Swiss Franc - Swissy
    'NZD/USD',  # New Zealand Dollar/US Dollar - Kiwi
    
    # Minor Cross Currency Pairs (Alpha Vantage Compatible)
    'EUR/GBP',  # Euro/British Pound
    'EUR/JPY',  # Euro/Japanese Yen
    'EUR/CHF',  # Euro/Swiss Franc
    'EUR/AUD',  # Euro/Australian Dollar
    'EUR/CAD',  # Euro/Canadian Dollar
    'EUR/NZD',  # Euro/New Zealand Dollar
    'GBP/JPY',  # British Pound/Japanese Yen
    'GBP/CHF',  # British Pound/Swiss Franc
    'GBP/AUD',  # British Pound/Australian Dollar
    'GBP/CAD',  # British Pound/Canadian Dollar
    'GBP/NZD',  # British Pound/New Zealand Dollar
    'CHF/JPY',  # Swiss Franc/Japanese Yen
    'AUD/JPY',  # Australian Dollar/Japanese Yen
    'AUD/CHF',  # Australian Dollar/Swiss Franc
    'AUD/CAD',  # Australian Dollar/Canadian Dollar
    'AUD/NZD',  # Australian Dollar/New Zealand Dollar
    'CAD/JPY',  # Canadian Dollar/Japanese Yen
    'CAD/CHF',  # Canadian Dollar/Swiss Franc
    'NZD/JPY',  # New Zealand Dollar/Japanese Yen
    'NZD/CHF',  # New Zealand Dollar/Swiss Franc
    'NZD/CAD',  # New Zealand Dollar/Canadian Dollar
    
    # Select Exotic Pairs (Alpha Vantage Supported)
    'USD/SEK',  # US Dollar/Swedish Krona
    'USD/NOK',  # US Dollar/Norwegian Krone
    'USD/DKK',  # US Dollar/Danish Krone  
    'USD/PLN',  # US Dollar/Polish Zloty
    'USD/TRY',  # US Dollar/Turkish Lira
    'USD/ZAR',  # US Dollar/South African Rand
    'USD/MXN',  # US Dollar/Mexican Peso
    'EUR/SEK',  # Euro/Swedish Krona
    'EUR/NOK',  # Euro/Norwegian Krone
    'EUR/DKK',  # Euro/Danish Krone
    'EUR/PLN',  # Euro/Polish Zloty
    'EUR/TRY',  # Euro/Turkish Lira
    'GBP/SEK',  # British Pound/Swedish Krona
    'GBP/NOK',  # British Pound/Norwegian Krone
    'GBP/PLN',  # British Pound/Polish Zloty
]

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
