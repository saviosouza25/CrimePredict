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
HORIZONS: List[str] = ['15 Minutos', '1 Hora', '4 Horas', '1 Dia', '1 Mês']
HORIZON_STEPS: Dict[str, int] = {
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

# Risk Management with AI Parameters
RISK_LEVELS = {
    'Conservative': 0.02,
    'Moderate': 0.05,
    'Aggressive': 0.10
}

# Advanced Risk Profiles with AI Integration
RISK_PROFILES = {
    'Conservative': {
        'atr_multiplier_stop': 1.2,      # Múltiplo ATR para stop loss
        'atr_multiplier_tp': 1.8,        # Múltiplo ATR para take profit
        'daily_range_factor': 0.6,       # Fator do range diário
        'volatility_buffer': 0.0001,     # Buffer mínimo de volatilidade
        'volatility_threshold': 0.015,   # Limiar de volatilidade (1.5%)
        'max_risk_pips': 25,             # Máximo risco em pips
        'min_risk_reward': 1.8,          # Mínima razão risco/retorno
        'confidence_adjustment': 0.15,   # Ajuste de confiança (15%)
        'banca_risk': 1.0,               # Máximo 1% da banca em risco
        'extension_factor': 1.4,         # Fator de extensão para cenários otimistas
        'reversal_sensitivity': 0.5,     # Sensibilidade para alertas de reversão
        
        # PARÂMETROS IA SEPARADOS
        'ai_historical_weight': 0.4,     # Peso da análise histórica (40%)
        'ai_sentiment_weight': 0.3,      # Peso da análise de sentimento (30%)
        'ai_probability_weight': 0.3,    # Peso das probabilidades (30%)
        'ai_confidence_threshold': 0.7,  # Limiar mínimo de confiança IA
        'ai_historical_periods': 20,     # Períodos históricos para análise
        'ai_sentiment_sensitivity': 0.6, # Sensibilidade ao sentimento (conservador)
        'ai_volatility_adjustment': 0.8, # Ajuste por volatilidade
        'ai_trend_strength_min': 0.6     # Força mínima de tendência
    },
    'Moderate': {
        'atr_multiplier_stop': 1.8,
        'atr_multiplier_tp': 2.5,
        'daily_range_factor': 0.8,
        'volatility_buffer': 0.00015,
        'volatility_threshold': 0.025,
        'max_risk_pips': 45,
        'min_risk_reward': 1.5,
        'confidence_adjustment': 0.25,
        'banca_risk': 2.0,
        'extension_factor': 1.8,
        'reversal_sensitivity': 0.4,
        
        # PARÂMETROS IA SEPARADOS
        'ai_historical_weight': 0.35,    # Peso da análise histórica (35%)
        'ai_sentiment_weight': 0.35,     # Peso da análise de sentimento (35%)
        'ai_probability_weight': 0.30,   # Peso das probabilidades (30%)
        'ai_confidence_threshold': 0.6,  # Limiar mínimo de confiança IA
        'ai_historical_periods': 30,     # Períodos históricos para análise
        'ai_sentiment_sensitivity': 0.7, # Sensibilidade ao sentimento (moderado)
        'ai_volatility_adjustment': 1.0, # Ajuste por volatilidade
        'ai_trend_strength_min': 0.5     # Força mínima de tendência
    },
    'Aggressive': {
        'atr_multiplier_stop': 2.5,
        'atr_multiplier_tp': 3.5,
        'daily_range_factor': 1.2,
        'volatility_buffer': 0.0002,
        'volatility_threshold': 0.04,
        'max_risk_pips': 80,
        'min_risk_reward': 1.2,
        'confidence_adjustment': 0.4,
        'banca_risk': 3.0,
        'extension_factor': 2.5,
        'reversal_sensitivity': 0.3,
        
        # PARÂMETROS IA SEPARADOS
        'ai_historical_weight': 0.30,    # Peso da análise histórica (30%)
        'ai_sentiment_weight': 0.40,     # Peso da análise de sentimento (40%)
        'ai_probability_weight': 0.30,   # Peso das probabilidades (30%)
        'ai_confidence_threshold': 0.5,  # Limiar mínimo de confiança IA
        'ai_historical_periods': 50,     # Períodos históricos para análise
        'ai_sentiment_sensitivity': 0.9, # Sensibilidade ao sentimento (agressivo)
        'ai_volatility_adjustment': 1.3, # Ajuste por volatilidade
        'ai_trend_strength_min': 0.4     # Força mínima de tendência
    }
}

# Parâmetros de IA específicos por estratégia temporal
TEMPORAL_AI_PARAMETERS = {

    '15 Minutos': {  # Scalping Avançado - Baseado em dados de day traders
        'ai_historical_periods': 32,       # 8 horas de dados
        'ai_volatility_sensitivity': 1.6,
        'ai_news_impact_weight': 0.75,
        'ai_technical_weight': 0.75,
        'ai_sentiment_decay': 0.90,
        'ai_probability_threshold': 0.62,
        'ai_trend_confirmation': 3,
        'ai_support_resistance_range': 0.0006,  # 6 pips
        'ai_momentum_periods': 6,          # 1.5 horas
        'ai_reversal_sensitivity': 1.3,
        'ai_success_rate_target': 0.65,    # 65% taxa de sucesso
        'ai_avg_movement_pips': 15,        # 15 pips movimento médio
        'ai_max_holding_periods': 8        # Máximo 2 horas
    },
    '1 Hora': {  # Intraday - Estatísticas reais de day trading
        'ai_historical_periods': 48,      # 2 dias completos de trading
        'ai_volatility_sensitivity': 1.2, # Sensibilidade baseada em ATR horário real
        'ai_news_impact_weight': 0.65,    # Notícias importantes movem 20-40 pips
        'ai_technical_weight': 0.70,      # 70% técnico, 30% fundamental
        'ai_sentiment_decay': 0.80,       # Sentimento persiste por algumas horas
        'ai_probability_threshold': 0.68,
        'ai_trend_confirmation': 4,        # 4 horas para confirmar tendência
        'ai_support_resistance_range': 0.0012, # 12 pips
        'ai_momentum_periods': 8,          # 8 horas de momentum
        'ai_reversal_sensitivity': 1.0,
        'ai_success_rate_target': 0.70,    # 70% taxa de sucesso intraday
        'ai_avg_movement_pips': 35,        # 35 pips movimento médio
        'ai_max_holding_periods': 16       # Máximo 16 horas (1 sessão)
    },
    '4 Horas': {  # Swing Trading - Dados reais de swing traders
        'ai_historical_periods': 60,      # 10 dias de dados (realístico)
        'ai_volatility_sensitivity': 0.9, # Menor sensibilidade, foco em tendências
        'ai_news_impact_weight': 0.55,    # Fundamentals começam a importar mais
        'ai_technical_weight': 0.65,
        'ai_sentiment_decay': 0.70,       # Sentimento dura 1-2 dias
        'ai_probability_threshold': 0.72,
        'ai_trend_confirmation': 3,        # 12 horas para confirmar
        'ai_support_resistance_range': 0.0025, # 25 pips
        'ai_momentum_periods': 12,         # 2 dias de momentum
        'ai_reversal_sensitivity': 0.8,
        'ai_success_rate_target': 0.75,    # 75% taxa de sucesso swing
        'ai_avg_movement_pips': 80,        # 80 pips movimento médio
        'ai_max_holding_periods': 30       # Máximo 5 dias
    },
    '1 Dia': {  # Position Trading - Estatísticas de position traders
        'ai_historical_periods': 90,      # 3 meses de histórico
        'ai_volatility_sensitivity': 0.7,
        'ai_news_impact_weight': 0.45,    # Fundamentals mais importantes
        'ai_technical_weight': 0.60,      # Mais fundamental que técnico
        'ai_sentiment_decay': 0.60,       # Sentimento semanal
        'ai_probability_threshold': 0.75,
        'ai_trend_confirmation': 3,        # 3 dias para confirmar
        'ai_support_resistance_range': 0.0040, # 40 pips
        'ai_momentum_periods': 20,         # 1 mês de momentum
        'ai_reversal_sensitivity': 0.6,
        'ai_success_rate_target': 0.80,    # 80% taxa de sucesso position
        'ai_avg_movement_pips': 150,       # 150 pips movimento médio
        'ai_max_holding_periods': 60       # Máximo 2 meses
    },
    '1 Semana': {  # Análise Semanal
        'ai_historical_periods': 52,      # 1 ano de dados semanais
        'ai_volatility_sensitivity': 0.4,
        'ai_news_impact_weight': 0.3,
        'ai_technical_weight': 0.85,
        'ai_sentiment_decay': 0.3,
        'ai_probability_threshold': 0.85,
        'ai_trend_confirmation': 4,
        'ai_support_resistance_range': 0.02,
        'ai_momentum_periods': 12,
        'ai_reversal_sensitivity': 0.4
    },
    '1 Mês': {  # Análise Mensal - Macro
        'ai_historical_periods': 24,      # 2 anos de dados mensais
        'ai_volatility_sensitivity': 0.3,
        'ai_news_impact_weight': 0.2,     # Foco macro/fundamental
        'ai_technical_weight': 0.9,       # Altamente técnico
        'ai_sentiment_decay': 0.2,        # Tendências macro duradouras
        'ai_probability_threshold': 0.9,   # Muito alto threshold
        'ai_trend_confirmation': 3,        # Poucas confirmações necessárias
        'ai_support_resistance_range': 0.05, # Range muito amplo
        'ai_momentum_periods': 6,          # Momentum macro
        'ai_reversal_sensitivity': 0.3     # Muito baixa sensibilidade
    }
}

# Parâmetros específicos para pares da Alpha Vantage por volatilidade
PAIR_AI_ADJUSTMENTS = {
    # Majors - Baixa volatilidade, alta previsibilidade
    'EUR/USD': {'volatility_multiplier': 0.8, 'prediction_confidence_boost': 1.2},
    'USD/JPY': {'volatility_multiplier': 0.9, 'prediction_confidence_boost': 1.1},
    'GBP/USD': {'volatility_multiplier': 1.1, 'prediction_confidence_boost': 1.0},
    'USD/CHF': {'volatility_multiplier': 0.85, 'prediction_confidence_boost': 1.15},
    
    # Commodities - Volatilidade média
    'AUD/USD': {'volatility_multiplier': 1.2, 'prediction_confidence_boost': 0.95},
    'USD/CAD': {'volatility_multiplier': 1.1, 'prediction_confidence_boost': 1.0},
    'NZD/USD': {'volatility_multiplier': 1.3, 'prediction_confidence_boost': 0.9},
    
    # Crosses - Volatilidade variável
    'EUR/GBP': {'volatility_multiplier': 0.9, 'prediction_confidence_boost': 1.05},
    'EUR/JPY': {'volatility_multiplier': 1.2, 'prediction_confidence_boost': 0.95},
    'GBP/JPY': {'volatility_multiplier': 1.5, 'prediction_confidence_boost': 0.85},
    
    # Exotics - Alta volatilidade, menor previsibilidade
    'USD/TRY': {'volatility_multiplier': 2.5, 'prediction_confidence_boost': 0.6},
    'USD/ZAR': {'volatility_multiplier': 2.0, 'prediction_confidence_boost': 0.7},
    'USD/MXN': {'volatility_multiplier': 1.8, 'prediction_confidence_boost': 0.75}
}
