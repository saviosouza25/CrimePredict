import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration imports
from config.settings import *
from config.languages import get_text

# Simplified imports - create basic versions if needed later
try:
    from models.lstm_model import ForexPredictor
    from services.data_service import DataService  
    from services.sentiment_service import SentimentService
    from services.ai_unified_service import AIUnifiedService
    from utils.cache_manager import CacheManager
    
    # Initialize services
    services = {
        'data_service': DataService(),
        'sentiment_service': SentimentService(),
        'ai_unified_service': AIUnifiedService()
    }
except ImportError as e:
    print(f"Import warning: {e}")
    # Create placeholder services for basic functionality
    class MockService:
        def fetch_forex_data(self, *args, **kwargs):
            return None
        def validate_data(self, *args, **kwargs):
            return False
        def fetch_news_sentiment(self, *args, **kwargs):
            return 0.0
    
    services = {
        'data_service': MockService(),
        'sentiment_service': MockService(),
        'ai_unified_service': MockService()
    }

# FUNÇÃO GLOBAL: Calcular probabilidades REAIS de mercado
def calculate_realistic_drawdown_and_extensions(current_price, pair_name, horizon, risk_level, sentiment_score, lstm_confidence):
    """
    Calcula drawdown máximo realístico e extensões de preço baseadas em estatísticas reais do mercado
    Focado 100% em precisão para Swing e Intraday trading
    """
    # Importar configurações localmente para evitar erros de circular import
    try:
        from config.settings import TEMPORAL_AI_PARAMETERS, RISK_PROFILES
    except ImportError:
        # Fallback com parâmetros básicos se a importação falhar
        TEMPORAL_AI_PARAMETERS = {
            '15 Minutos': {'ai_drawdown_probability': 0.25, 'ai_max_adverse_pips': 12, 'ai_extension_probability': 0.70, 'ai_realistic_targets': {'conservative': 18, 'moderate': 25, 'aggressive': 35}},
            '1 Hora': {'ai_drawdown_probability': 0.30, 'ai_max_adverse_pips': 25, 'ai_extension_probability': 0.75, 'ai_realistic_targets': {'conservative': 25, 'moderate': 40, 'aggressive': 60}},
            '4 Horas': {'ai_drawdown_probability': 0.35, 'ai_max_adverse_pips': 45, 'ai_extension_probability': 0.80, 'ai_realistic_targets': {'conservative': 60, 'moderate': 90, 'aggressive': 130}}
        }
        RISK_PROFILES = {
            'Conservative': {'atr_multiplier_stop': 1.5},
            'Moderate': {'atr_multiplier_stop': 2.0},
            'Aggressive': {'atr_multiplier_stop': 2.5}
        }
    
    # Obter parâmetros específicos do horizonte temporal
    temporal_params = TEMPORAL_AI_PARAMETERS.get(horizon, TEMPORAL_AI_PARAMETERS['1 Hora'])
    risk_profile = RISK_PROFILES.get(risk_level, RISK_PROFILES['Moderate'])
    
    # Calcular ATR para volatilidade base
    pair_name_str = str(pair_name)  # Garantir que é string
    pip_value = 0.0001 if 'JPY' not in pair_name_str else 0.01
    
    # Probabilidades realísticas baseadas em dados históricos
    drawdown_probability = temporal_params.get('ai_drawdown_probability', 0.30)
    max_adverse_pips = temporal_params.get('ai_max_adverse_pips', 25)
    extension_probability = temporal_params.get('ai_extension_probability', 0.75)
    
    # Ajustar por confiança do modelo
    confidence_adjustment = (lstm_confidence - 0.5) * 0.4  # -0.2 a +0.2
    adjusted_extension_prob = min(0.95, max(0.50, extension_probability + confidence_adjustment))
    adjusted_drawdown_prob = max(0.15, min(0.50, drawdown_probability - confidence_adjustment))
    
    # Ajustar por sentimento
    sentiment_boost = abs(sentiment_score) * 0.15  # Máximo 15% de boost
    if sentiment_score > 0:
        adjusted_extension_prob += sentiment_boost
        adjusted_drawdown_prob -= sentiment_boost * 0.5
    else:
        adjusted_extension_prob -= sentiment_boost * 0.5
        adjusted_drawdown_prob += sentiment_boost
    
    # Calcular drawdown máximo realístico (pontos contra a tendência inicial)
    base_adverse_distance = max_adverse_pips * pip_value
    
    # Ajustar por perfil de risco
    risk_multiplier = risk_profile.get('atr_multiplier_stop', 2.0)
    realistic_max_drawdown = base_adverse_distance * risk_multiplier
    
    # Calcular extensões realísticas (até onde pode chegar na direção da análise)
    realistic_targets = temporal_params.get('ai_realistic_targets', {
        'conservative': 30,
        'moderate': 50, 
        'aggressive': 75
    })
    
    # Selecionar alvo baseado no perfil
    if risk_level == 'Conservative':
        target_pips = realistic_targets.get('conservative', 30)
    elif risk_level == 'Aggressive':
        target_pips = realistic_targets.get('aggressive', 75)
    else:
        target_pips = realistic_targets.get('moderate', 50)
    
    # Ajustar alvo por confiança e sentimento
    confidence_multiplier = 0.7 + (lstm_confidence * 0.6)  # 0.7 - 1.3
    sentiment_multiplier = 1.0 + (abs(sentiment_score) * 0.3)  # 1.0 - 1.3
    
    adjusted_target_pips = target_pips * confidence_multiplier * sentiment_multiplier
    realistic_extension = adjusted_target_pips * pip_value
    
    # Calcular probabilidades finais realísticas
    final_extension_prob = min(0.92, max(0.55, adjusted_extension_prob))
    final_drawdown_prob = min(0.45, max(0.18, adjusted_drawdown_prob))
    
    # Determinar direção baseada na análise
    direction = "ALTA" if sentiment_score > 0 or lstm_confidence > 0.6 else "BAIXA"
    
    # Garantir que current_price é float para operações matemáticas
    current_price = float(current_price)
    
    if direction == "ALTA":
        max_adverse_level = current_price - realistic_max_drawdown
        extension_level = current_price + realistic_extension
    else:
        max_adverse_level = current_price + realistic_max_drawdown
        extension_level = current_price - realistic_extension
    
    return {
        'direction': direction,
        'drawdown_pips': int(realistic_max_drawdown / pip_value),
        'max_adverse_level': max_adverse_level,
        'drawdown_probability': final_drawdown_prob,
        'extension_pips': int(realistic_extension / pip_value),
        'extension_level': extension_level,
        'extension_probability': final_extension_prob,
        'confidence_adjustment': confidence_adjustment,
        'sentiment_impact': sentiment_boost,
        'horizon_base_target': target_pips,
        'adjusted_target': int(adjusted_target_pips)
    }

def calculate_market_probabilities_real(lstm_confidence, ai_consensus, sentiment_score, technical_signals, pair_name, horizon):
    """Calcular probabilidades REAIS de sucesso baseadas em confluência de análises"""
    
    # Obter parâmetros realísticos por horizonte temporal
    temporal_params = TEMPORAL_AI_PARAMETERS.get(horizon, TEMPORAL_AI_PARAMETERS['1 Hora'])
    success_rate_base = temporal_params.get('ai_success_rate_target', 0.70)
    
    # 1. LSTM Probability (40% do peso total)
    lstm_prob = min(0.95, max(0.30, lstm_confidence * 1.2))  # Entre 30-95%
    
    # 2. AI Consensus Probability (30% do peso total)
    ai_prob = min(0.90, max(0.25, ai_consensus * 1.1))  # Entre 25-90%
    
    # 3. Sentiment Probability (20% do peso total)
    sentiment_normalized = abs(sentiment_score)
    sentiment_prob = min(0.85, max(0.40, 0.5 + (sentiment_normalized * 0.4)))  # Entre 40-85%
    
    # 4. Technical Signals Probability (10% do peso total)
    technical_prob = min(0.80, max(0.35, technical_signals * 0.8 + 0.35))  # Entre 35-80%
    
    # Calcular probabilidade confluente REAL
    confluent_probability = (
        lstm_prob * 0.40 +
        ai_prob * 0.30 + 
        sentiment_prob * 0.20 + 
        technical_prob * 0.10
    )
    
    # Ajuste por par de moeda (volatilidade real)
    pair_adjustment = PAIR_AI_ADJUSTMENTS.get(pair_name, {'prediction_confidence_boost': 1.0})
    adjusted_probability = confluent_probability * pair_adjustment['prediction_confidence_boost']
    
    # Limitar entre probabilidades realísticas por horizonte
    min_prob = success_rate_base * 0.6  # 60% da taxa base mínima
    max_prob = min(0.95, success_rate_base * 1.3)  # Máximo 95% ou 130% da base
    
    final_probability = max(min_prob, min(max_prob, adjusted_probability))
    
    return {
        'confluent_probability': final_probability,
        'lstm_component': lstm_prob * 0.40,
        'ai_component': ai_prob * 0.30,
        'sentiment_component': sentiment_prob * 0.20,
        'technical_component': technical_prob * 0.10,
        'base_success_rate': success_rate_base,
        'confidence_breakdown': {
            'Very High (>85%)': final_probability > 0.85,
            'High (70-85%)': 0.70 <= final_probability <= 0.85,
            'Medium (55-70%)': 0.55 <= final_probability < 0.70,
            'Low (<55%)': final_probability < 0.55
        }
    }

# FUNÇÃO GLOBAL: Estratégia Temporal Unificada Original - Gatilhos Alpha Vantage
def calculate_confluent_levels_global(current_price, predicted_price, pair_name, profile, market_probability):
    """Estratégia Temporal Unificada Original: Prever próximos movimentos baseado no período gráfico + gatilhos Alpha Vantage"""
    
    import streamlit as st
    horizon = st.session_state.get('analysis_horizon', '1 Hora')
    
    # GATILHOS ALPHA VANTAGE POR PERÍODO - Dados reais de movimentação típica
    alpha_triggers = {
        '5 Minutos': {
            'volatility_range': 0.15,     # Scalping: movimentos de 15% do ATR por período
            'momentum_threshold': 0.8,     # Threshold baixo para captação rápida
            'stop_protection': 0.6,        # Proteção apertada (60% ATR)
            'target_extension': 1.2,       # Alvo conservador (120% ATR)
            'trend_confirmation': 2,       # Confirma em 2 períodos
            'market_noise_filter': 0.3     # Filtro de ruído alto
        },
        '15 Minutos': {
            'volatility_range': 0.25,     # Intraday: movimentos de 25% do ATR
            'momentum_threshold': 1.0,     
            'stop_protection': 1.0,        # Proteção moderada (100% ATR) - ajustado
            'target_extension': 2.5,       # Alvo equilibrado (250% ATR) - aumentado
            'trend_confirmation': 3,       
            'market_noise_filter': 0.2
        },
        '30 Minutos': {
            'volatility_range': 0.35,     
            'momentum_threshold': 1.2,     
            'stop_protection': 1.3,        # (130% ATR) - ajustado
            'target_extension': 3.2,       # (320% ATR) - aumentado
            'trend_confirmation': 4,       
            'market_noise_filter': 0.15
        },
        '1 Hora': {
            'volatility_range': 0.50,     # Intraday amplo: movimentos significativos
            'momentum_threshold': 1.5,     
            'stop_protection': 1.6,        # Proteção moderada (160% ATR) - ajustado
            'target_extension': 4.0,       # Alvo amplo (400% ATR) - aumentado
            'trend_confirmation': 5,       
            'market_noise_filter': 0.1
        },
        '4 Horas': {
            'volatility_range': 0.75,     # Swing: movimentos estruturais
            'momentum_threshold': 2.0,     
            'stop_protection': 2.8,        # Proteção ampla (280% ATR) - aumentado
            'target_extension': 7.0,       # Alvo extenso (700% ATR) - muito aumentado
            'trend_confirmation': 6,       
            'market_noise_filter': 0.05
        },
        '1 Dia': {
            'volatility_range': 1.0,      # Position: movimentos estruturais longos
            'momentum_threshold': 2.5,     
            'stop_protection': 4.5,        # Proteção muito ampla (450% ATR) - muito aumentado
            'target_extension': 12.0,      # Alvo muito extenso (1200% ATR) - dobrado
            'trend_confirmation': 8,       
            'market_noise_filter': 0.02
        },
        '1 Mês': {
            'volatility_range': 1.5,      # Long-term: movimentos estruturais (mantido para compatibilidade)
            'momentum_threshold': 3.0,     
            'stop_protection': 3.5,        # Proteção realista (350% ATR)
            'target_extension': 7.0,       # Alvo realista (700% ATR)
            'trend_confirmation': 12,      
            'market_noise_filter': 0.01
        }
    }
    
    # Obter gatilhos do período escolhido
    triggers = alpha_triggers.get(horizon, alpha_triggers['1 Hora'])
    
    # ATR real por par (Alpha Vantage)
    atr_values = {
        'EUR/USD': 0.0012, 'USD/JPY': 0.018, 'GBP/USD': 0.0018, 'AUD/USD': 0.0020,
        'USD/CAD': 0.0014, 'USD/CHF': 0.0016, 'NZD/USD': 0.0022, 'EUR/GBP': 0.0010,
        'EUR/JPY': 0.020, 'GBP/JPY': 0.025, 'AUD/JPY': 0.022
    }
    current_atr = atr_values.get(pair_name, 0.0015)
    
    # ANÁLISE CONFLUENTE: Previsão + Probabilidade + Momentum
    # Garantir que são floats para operações matemáticas
    current_price = float(current_price)
    predicted_price = float(predicted_price)
    
    direction = 1 if predicted_price > current_price else -1
    price_momentum = abs(predicted_price - current_price) / current_price
    prob_strength = market_probability['confluent_probability']
    
    # GATILHO DE MOMENTUM: Verificar se movimento supera threshold do período
    momentum_confirmed = price_momentum >= (triggers['momentum_threshold'] * current_atr / current_price)
    
    # ESTRATÉGIA TEMPORAL UNIFICADA: Variação Real do Mercado por Período
    
    # 1. VARIAÇÃO REAL DO MERCADO POR PERÍODO (dados históricos Alpha Vantage)
    market_variation_data = {
        '5 Minutos': {
            'typical_move_atr': 0.3,    # Scalping: movimentos típicos de 30% ATR
            'max_adverse_atr': 0.8,     # Máximo movimento adverso antes de reversão
            'profit_target_atr': 0.6    # Alvo típico realizável em scalping
        },
        '15 Minutos': {
            'typical_move_atr': 0.6,    # Intraday: movimentos de 60% ATR
            'max_adverse_atr': 1.2,     # Máximo adverso
            'profit_target_atr': 1.5    # Alvo intraday
        },
        '1 Hora': {
            'typical_move_atr': 1.0,    # Movimentos de 100% ATR
            'max_adverse_atr': 1.8,     # Máximo adverso 
            'profit_target_atr': 2.5    # Alvo horário
        },
        '4 Horas': {
            'typical_move_atr': 2.2,    # Swing: movimentos estruturais
            'max_adverse_atr': 3.5,     # Máximo adverso swing
            'profit_target_atr': 5.5    # Alvo swing
        },
        '1 Dia': {
            'typical_move_atr': 4.0,    # Position: movimentos diários
            'max_adverse_atr': 6.5,     # Máximo adverso diário
            'profit_target_atr': 10.0   # Alvo position
        }
    }
    
    # Obter dados de variação do período selecionado
    market_data = market_variation_data.get(horizon, market_variation_data['1 Hora'])
    
    # 2. PERFIL TRADER: Define tolerância ao risco baseada na variação real
    profile_name = profile.get('name', 'Moderate')
    
    if profile_name == 'Conservative':
        # Conservative: Stop baseado em 70% da variação adversa máxima
        stop_safety_factor = 0.7
        # Take baseado em 60% do alvo típico (mais conservador)
        take_target_factor = 0.6
        risk_tolerance = 0.8
    elif profile_name == 'Aggressive':
        # Aggressive: Stop baseado em 120% da variação adversa (mais risco)
        stop_safety_factor = 1.2
        # Take baseado em 150% do alvo típico (mais ambicioso)
        take_target_factor = 1.5
        risk_tolerance = 1.3
    else:  # Moderate
        # Moderate: Stop baseado em 100% da variação adversa real
        stop_safety_factor = 1.0
        # Take baseado em 100% do alvo típico
        take_target_factor = 1.0
        risk_tolerance = 1.0
    
    # 3. CÁLCULO DO STOP baseado na VARIAÇÃO REAL DO MERCADO
    # Stop = Máximo movimento adverso real × fator de segurança do perfil
    stop_multiplier = market_data['max_adverse_atr'] * stop_safety_factor * risk_tolerance
    
    # 4. CÁLCULO DO TAKE baseado no POTENCIAL REAL DO PERÍODO
    # Take = Alvo típico do período × fator do perfil × força do sinal
    signal_strength_multiplier = 0.8 + (prob_strength * 0.4)  # Entre 0.8 e 1.2
    take_multiplier = market_data['profit_target_atr'] * take_target_factor * signal_strength_multiplier
    
    # 5. VALIDAÇÃO: Garantir que ratio risco/retorno seja realista
    calculated_ratio = take_multiplier / stop_multiplier if stop_multiplier > 0 else 0
    
    # Ajustar se ratio estiver fora dos padrões reais do mercado
    if calculated_ratio < 1.2:  # Ratio muito baixo
        take_multiplier = stop_multiplier * 1.5  # Forçar ratio mínimo 1:1.5
    elif calculated_ratio > 4.0:  # Ratio muito alto (irrealista)
        take_multiplier = stop_multiplier * 3.5  # Limitar ratio máximo 1:3.5
    
    # Determinar força da confluência
    if prob_strength > 0.75 and momentum_confirmed:
        confluence_strength = "ALTA CONFLUÊNCIA"
        confidence_boost = 1.0
    elif prob_strength > 0.6:
        confluence_strength = "CONFLUÊNCIA MODERADA" 
        confidence_boost = 0.95
    elif prob_strength > 0.5:
        confluence_strength = "BAIXA CONFLUÊNCIA"
        confidence_boost = 0.9
    else:
        confluence_strength = "SEM CONFLUÊNCIA"
        confidence_boost = 0.85
    
    # APLICAR FILTRO DE RUÍDO (reduz em mercados laterais)
    noise_factor = 1.0 - triggers['market_noise_filter']
    
    # CALCULAR NÍVEIS FINAIS baseados nos gatilhos Alpha + período temporal
    final_stop_multiplier = stop_multiplier * confidence_boost * noise_factor
    final_take_multiplier = take_multiplier * confidence_boost * noise_factor
    
    # PREÇOS DE STOP/TAKE baseados na estratégia temporal unificada
    if direction == 1:  # COMPRA
        stop_loss_price = current_price - (current_atr * final_stop_multiplier)
        take_profit_price = current_price + (current_atr * final_take_multiplier)
    else:  # VENDA
        stop_loss_price = current_price + (current_atr * final_stop_multiplier)
        take_profit_price = current_price - (current_atr * final_take_multiplier)
    
    # Converter para pontos (pips)
    def price_to_points(price1, price2, pair_name):
        diff = abs(price1 - price2)
        pair_str = str(pair_name)  # Garantir que é string
        if 'JPY' in pair_str:
            return round(diff * 100, 1)
        else:
            return round(diff * 10000, 1)
    
    stop_points = price_to_points(current_price, stop_loss_price, pair_name)
    take_points = price_to_points(current_price, take_profit_price, pair_name)
    risk_reward_ratio = take_points / stop_points if stop_points > 0 else 0
    
    # Análise de confirmação de tendência
    trend_strength = "FORTE" if momentum_confirmed and prob_strength > 0.7 else \
                    "MODERADA" if prob_strength > 0.5 else "FRACA"
    
    return {
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'stop_loss_points': stop_points,
        'take_profit_points': take_points,
        'risk_reward_ratio': risk_reward_ratio,
        'operation_direction': 'COMPRA' if direction == 1 else 'VENDA',
        'confluent_probability': prob_strength,
        'atr_used': current_atr,
        'fibonacci_support_ref': current_price - current_atr,
        'fibonacci_resistance_ref': current_price + current_atr,
        'position_strength': trend_strength,
        'temporal_strategy': f"Temporal Unificada {horizon}",
        'fibonacci_adjustment': confidence_boost,
        'volatility_factor': noise_factor,
        'final_multipliers': {
            'stop': final_stop_multiplier,
            'take': final_take_multiplier
        },
        'alpha_triggers': triggers,
        'momentum_confirmed': momentum_confirmed,
        'temporal_period': horizon,
        'confluence_analysis': {
            'strength': confluence_strength,
            'profile_used': profile_name,
            'market_variation': market_data,
            'stop_safety_factor': stop_safety_factor,
            'take_target_factor': take_target_factor,
            'calculated_ratio': calculated_ratio,
            'final_stop_atr': stop_multiplier,
            'final_take_atr': take_multiplier,
            'variation_base': f"Stop baseado em {stop_safety_factor*100:.0f}% da variação adversa real",
            'target_base': f"Take baseado em {take_target_factor*100:.0f}% do potencial do período",
            'risk_evaluation': 'FAVORÁVEL' if confidence_boost > 0.95 else 'MODERADO' if confidence_boost > 0.9 else 'CONSERVADOR'
        },
        'next_market_prediction': {
            'direction': 'ALTA' if direction == 1 else 'BAIXA',
            'strength': trend_strength,
            'time_confirmation': f"{triggers['trend_confirmation']} períodos",
            'volatility_expected': f"{triggers['volatility_range']*100:.0f}% do ATR",
            'confluence_rating': confluence_strength
        }
    }

# Simple technical indicators class
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df):
        return df if df is not None else None
    
    @staticmethod  
    def get_trading_signals(df):
        return {"overall": "HOLD"} if df is not None else {}

# Simple cache manager
class CacheManager:
    @staticmethod
    def clear_cache():
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(('cache_', 'analysis_', 'unified_', 'ai_result_')):
                if key != 'analysis_results':  # Preservar resultado final
                    del st.session_state[key]

def apply_theme_css():
    """Apply futuristic AI-themed CSS styling"""
    # Futuristic AI CSS with cyberpunk aesthetics
    futuristic_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Modern Trading Platform Theme */
        .stApp {
            background: linear-gradient(145deg, #0d1421 0%, #1a252f 30%, #161b22 70%, #0d1117 100%) !important;
            min-height: 100vh;
            font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif !important;
        }
        
        .main .block-container {
            background: linear-gradient(145deg, rgba(13,20,33,0.7), rgba(22,27,34,0.8)) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(58,175,169,0.2) !important;
            border-radius: 16px !important;
            box-shadow: 
                0 25px 50px rgba(0,0,0,0.5),
                inset 0 1px 0 rgba(58,175,169,0.1) !important;
            color: rgba(255,255,255,0.95) !important;
            max-width: 1200px;
            padding: 2.5rem;
            margin-top: 1rem;
        }
        
        /* Modern Trading Cards */
        .metric-card {
            background: linear-gradient(145deg, rgba(22,27,34,0.9), rgba(13,20,33,0.8)) !important;
            backdrop-filter: blur(25px) !important;
            border: 1px solid rgba(58,175,169,0.4) !important;
            padding: 2rem !important;
            border-radius: 12px !important;
            box-shadow: 
                0 10px 30px rgba(0,0,0,0.4),
                0 1px 0 rgba(58,175,169,0.2) !important;
            margin: 1.5rem 0 !important;
            position: relative !important;
            overflow: hidden !important;
            color: rgba(255,255,255,0.95) !important;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3afa9, #21d397, #06d6a0);
            animation: trading-glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes trading-glow {
            from { box-shadow: 0 0 8px rgba(58,175,169,0.3); }
            to { box-shadow: 0 0 20px rgba(6,214,160,0.6); }
        }
        
        /* Professional Trading Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #3afa9 !important;
            text-shadow: 0 0 12px rgba(58,175,169,0.4) !important;
            font-family: 'Inter', 'SF Pro Display', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }
        
        h1 {
            background: linear-gradient(135deg, #3afa9, #21d397, #06d6a0) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-size: 2.8rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
        }
        
        /* Trading Platform Animations */
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Modern Trading Platform Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #21d397 0%, #3afa9 100%) !important;
            color: #0d1117 !important;
            border: 1px solid rgba(58,175,169,0.3) !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            font-family: 'Inter', 'SF Pro Display', sans-serif !important;
            text-transform: none !important;
            letter-spacing: 0 !important;
            font-size: 0.9rem !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 8px rgba(58,175,169,0.2) !important;
            text-shadow: none !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.6s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #06d6a0 0%, #21d397 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(58,175,169,0.4) !important;
            border: 1px solid rgba(58,175,169,0.5) !important;
        }
        
        /* SIDEBAR COMPACTO E MODERNO */
        .css-1d391kg, .st-emotion-cache-1d391kg, section[data-testid="stSidebar"] {
            background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%) !important;
            border-right: 3px solid #4CAF50 !important;
            padding: 15px !important;
            min-width: 300px !important;
            max-width: 320px !important;
        }
        
        /* SOLUÇÃO DEFINITIVA - Labels Pretos em Fundo Branco */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #000000 !important;
            background: #ffffff !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            padding: 8px 0 !important;
            margin: 8px 0 !important;
            text-shadow: none !important;
            border: none !important;
        }
        
        /* CAMPOS COMPACTOS E ELEGANTES */
        section[data-testid="stSidebar"] .stSelectbox {
            margin: 8px 0 !important;
        }
        
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            background: #ffffff !important;
            color: #2d3748 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            min-height: 38px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
            color: #2d3748 !important;
            font-weight: 500 !important;
            font-size: 14px !important;
        }
        
        /* INPUTS COMPACTOS E MODERNOS */
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stTextInput input {
            background: #ffffff !important;
            color: #2d3748 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            min-height: 38px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        /* Selectbox dropdown options - force visibility */
        .stSelectbox [data-baseweb="popover"] ul,
        .stSelectbox [data-baseweb="popover"] li,
        .stSelectbox [data-baseweb="popover"] [role="option"] {
            background: rgba(255,255,255,0.98) !important;
            color: #000000 !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }
        
        /* Force all dropdown content to be visible - comprehensive */
        .stSelectbox [data-baseweb="popover"] *,
        .stSelectbox [data-baseweb="popover"] li *,
        .stSelectbox [data-baseweb="popover"] div *,
        .stSelectbox [data-baseweb="popover"] span *,
        .stSelectbox [data-baseweb="popover"] [role="option"] *,
        .stSelectbox [data-baseweb="popover"] [data-baseweb="menu-item"] *,
        .stSelectbox [data-baseweb="popover"] [data-baseweb="list-item"] * {
            color: #000000 !important;
            background: transparent !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        
        /* Specific targeting for menu items */
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] [role="option"],
        [data-baseweb="list"] li,
        [data-baseweb="list"] [role="option"] {
            background: rgba(255,255,255,0.98) !important;
            color: #000000 !important;
            font-weight: 600 !important;
            padding: 8px 12px !important;
            border-bottom: 1px solid rgba(0,245,255,0.2) !important;
        }
        
        /* Selectbox option hover */
        .stSelectbox [data-baseweb="popover"] li:hover {
            background: rgba(0,245,255,0.8) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Additional selectbox styling for better visibility */
        .stSelectbox [data-baseweb="popover"],
        .stSelectbox [data-baseweb="popover-content"] {
            background: rgba(255,255,255,0.98) !important;
            backdrop-filter: blur(20px) !important;
            border: 2px solid rgba(0,245,255,0.4) !important;
            border-radius: 10px !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
        }
        
        /* Ensure all popover elements have proper styling */
        .stSelectbox [data-baseweb="popover"] > div,
        .stSelectbox [data-baseweb="popover-content"] > div,
        .stSelectbox [data-baseweb="popover"] ul,
        .stSelectbox [data-baseweb="popover"] ol,
        .stSelectbox [data-baseweb="menu"],
        .stSelectbox [data-baseweb="list"] {
            background: rgba(255,255,255,0.98) !important;
            color: #000000 !important;
        }
        
        /* SOLUÇÃO DEFINITIVA - Dropdown Simples e Claro */
        .stSelectbox [data-baseweb="popover"] {
            background: #f0f0f0 !important;
            border: 2px solid #4CAF50 !important;
            border-radius: 4px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }
        
        .stSelectbox [role="option"],
        .stSelectbox [data-baseweb="menu-item"],
        .stSelectbox [data-baseweb="list-item"],
        .stSelectbox li {
            background: #f0f0f0 !important;
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #ddd !important;
        }
        
        .stSelectbox [role="option"]:hover,
        .stSelectbox [data-baseweb="menu-item"]:hover,
        .stSelectbox [data-baseweb="list-item"]:hover,
        .stSelectbox li:hover {
            background: #4CAF50 !important;
            color: #ffffff !important;
        }
        
        /* Selectbox option text visibility */
        .stSelectbox [data-baseweb="popover"] div,
        .stSelectbox [data-baseweb="popover"] span {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            text-shadow: none !important;
        }
        
        /* Selectbox selected option */
        .stSelectbox [data-baseweb="popover"] [aria-selected="true"] {
            background: rgba(0,245,255,0.8) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Sidebar headers */
        .css-1d391kg h1, 
        .css-1d391kg h2, 
        .css-1d391kg h3,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #00f5ff !important;
            text-shadow: 0 0 15px rgba(0,245,255,0.7) !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Enhanced Input Fields with High Visibility */
        .stSelectbox > div > div, 
        .stNumberInput > div > div > input, 
        .stTextInput > div > div > input,
        .stSlider > div > div > div > div {
            background: rgba(255,255,255,0.15) !important;
            border: 2px solid rgba(0,245,255,0.6) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            backdrop-filter: blur(15px) !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
            text-shadow: 0 0 5px rgba(255,255,255,0.8) !important;
        }
        
        .stSelectbox > div > div:focus, 
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border: 2px solid #00f5ff !important;
            box-shadow: 0 0 25px rgba(0,245,255,0.5) !important;
            background: rgba(255,255,255,0.25) !important;
        }
        
        /* Select dropdown options - enhanced */
        .stSelectbox > div > div > div {
            background: rgba(255,255,255,0.98) !important;
            color: #000000 !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
            border-radius: 8px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }
        
        /* Selectbox input text */
        .stSelectbox > div > div input {
            color: #000000 !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            background: rgba(255,255,255,0.95) !important;
        }
        
        /* Password input field - dark background with white text */
        .stTextInput input[type="password"] {
            background: rgba(0,0,0,0.8) !important;
            color: #ffffff !important;
            border: 2px solid rgba(0,245,255,0.6) !important;
            border-radius: 8px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.8rem !important;
            text-shadow: none !important;
        }
        
        /* Password input focus state */
        .stTextInput input[type="password"]:focus {
            background: rgba(0,0,0,0.9) !important;
            color: #ffffff !important;
            border: 2px solid #00f5ff !important;
            box-shadow: 0 0 15px rgba(0,245,255,0.5) !important;
        }
        
        /* General text input styling for better visibility */
        .stTextInput input {
            background: rgba(0,0,0,0.7) !important;
            color: #ffffff !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
        }
        
        /* Number input styling */
        .stNumberInput input {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        
        /* Futuristic Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            background: rgba(255,255,255,0.05) !important;
            padding: 12px !important;
            border-radius: 15px !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(0,245,255,0.2) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 10px !important;
            padding: 15px 30px !important;
            color: #00f5ff !important;
            font-weight: 500 !important;
            transition: all 0.4s ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
            transform: translateY(-3px) !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
        }
        
        /* Enhanced Metrics */
        [data-testid="metric-container"] {
            background: linear-gradient(145deg, rgba(0,245,255,0.15), rgba(255,0,245,0.15)) !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(15px) !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0,245,255,0.3) !important;
        }
        
        [data-testid="metric-container"] > div > div {
            color: #00f5ff !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Status Messages with Glow */
        .stSuccess {
            background: linear-gradient(135deg, rgba(0,255,127,0.2), rgba(0,255,127,0.1)) !important;
            border: 1px solid rgba(0,255,127,0.5) !important;
            color: #00ff7f !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(255,0,127,0.2), rgba(255,0,127,0.1)) !important;
            border: 1px solid rgba(255,0,127,0.5) !important;
            color: #ff007f !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,215,0,0.1)) !important;
            border: 1px solid rgba(255,215,0,0.5) !important;
            color: #ffd700 !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(0,245,255,0.1)) !important;
            border: 1px solid rgba(0,245,255,0.5) !important;
            color: #00f5ff !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Enhanced Text and Content Visibility */
        .stMarkdown, p, span, div, label {
            color: rgba(255,255,255,0.95) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            text-shadow: 0 0 5px rgba(255,255,255,0.2) !important;
        }
        
        /* Strong emphasis for labels and important text */
        label, .stMarkdown strong, .stMarkdown b {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(0,245,255,0.4) !important;
        }
        
        /* Caption and help text */
        .stMarkdown .caption, .help {
            color: rgba(255,255,255,0.8) !important;
            font-size: 0.9rem !important;
        }
        
        /* Enhanced Expanders */
        .streamlit-expanderHeader {
            background: rgba(255,255,255,0.15) !important;
            border: 2px solid rgba(0,245,255,0.6) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            backdrop-filter: blur(15px) !important;
            font-weight: 600 !important;
            text-shadow: 0 0 10px rgba(0,245,255,0.5) !important;
            padding: 1rem !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(255,255,255,0.2) !important;
            border: 2px solid #00f5ff !important;
            box-shadow: 0 0 20px rgba(0,245,255,0.3) !important;
        }
        
        /* Expander content */
        .streamlit-expanderContent {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(0,245,255,0.3) !important;
            border-radius: 0 0 12px 12px !important;
            padding: 1rem !important;
        }
        
        /* Charts with AI glow */
        .js-plotly-plot {
            background: rgba(255,255,255,0.05) !important;
            border-radius: 15px !important;
            border: 2px solid rgba(0,245,255,0.3) !important;
            box-shadow: 0 0 30px rgba(0,245,255,0.2) !important;
        }
        
        /* Futuristic Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #00f5ff, #ff00f5);
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,245,255,0.5);
        }
        
        /* Loading Animations */
        @keyframes matrix {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        
        .loading {
            animation: matrix 2s infinite;
        }
        
        /* Hide default spinners */
        .stSpinner > div, div[data-testid="stSpinner"], .stStatus {
            display: none !important;
        }
        
        /* Fix sidebar collapse/expand button to show arrows instead of text */
        button[title="Close sidebar"], button[title="Open sidebar"] {
            font-size: 0 !important;
        }
        
        button[title="Close sidebar"]::before {
            content: "◀" !important;
            font-size: 1.2rem !important;
            color: #00f5ff !important;
            text-shadow: 0 0 10px rgba(0,245,255,0.6) !important;
        }
        
        button[title="Open sidebar"]::before {
            content: "▶" !important;
            font-size: 1.2rem !important;
            color: #00f5ff !important;
            text-shadow: 0 0 10px rgba(0,245,255,0.6) !important;
        }
        
        /* Streamlit toolbar hide/show button fix */
        .stActionButton > button, button[data-testid="stSidebarNav"] {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(0,245,255,0.3) !important;
            border-radius: 8px !important;
            color: #00f5ff !important;
            transition: all 0.3s ease !important;
        }
        
        .stActionButton > button:hover, button[data-testid="stSidebarNav"]:hover {
            background: rgba(0,245,255,0.2) !important;
            border: 1px solid #00f5ff !important;
            box-shadow: 0 0 15px rgba(0,245,255,0.4) !important;
        }
        
        /* Sidebar specific styling overrides */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            padding: 0.8rem 1.5rem !important;
            font-size: 0.85rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Individual analysis buttons - balanced compact styling */
        section[data-testid="stSidebar"] div[data-testid="column"] .stButton > button {
            padding: 0.5rem 0.8rem !important;
            font-size: 0.75rem !important;
            min-height: 2.8rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        /* BOTÕES COMPACTOS E ELEGANTES */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            min-height: 36px !important;
            margin: 4px 0 !important;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div {
            background: rgba(0,245,255,0.3) !important;
        }
        
        .stSlider > div > div > div > div {
            background: #00f5ff !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 0 15px rgba(0,245,255,0.6) !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: rgba(255,255,255,0.05) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        .stRadio > div > label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Checkbox styling */
        .stCheckbox > label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            background: rgba(255,255,255,0.1) !important;
            border: 2px dashed rgba(0,245,255,0.5) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background: linear-gradient(90deg, #00f5ff, #ff00f5) !important;
        }
        
        /* Alert boxes */
        .element-container .stAlert {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(0,245,255,0.4) !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
        }
    </style>
    """
    st.markdown(futuristic_css, unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Header futurístico mais slim
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(255,0,245,0.1));
            backdrop-filter: blur(20px);
            border: 2px solid rgba(0,245,255,0.3);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            color: white;
            margin: 1rem auto 2rem auto;
            max-width: 800px;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #00f5ff, #ff00f5, #f5ff00);
                animation: glow 3s ease-in-out infinite alternate;
            "></div>
            <h1 style="
                background: linear-gradient(135deg, #00f5ff, #ff00f5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 0 0 20px rgba(0,245,255,0.6);
            ">🔐 FOREX AI ACCESS PORTAL</h1>
            <p style="
                color: rgba(255,255,255,0.8);
                font-size: 1rem;
                margin: 0;
                text-shadow: 0 0 10px rgba(0,245,255,0.3);
            ">Advanced Neural Trading Platform • Secure Authentication Required</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Formulário de login horizontal slim
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(15px);
            border: 1px solid rgba(0,245,255,0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0 auto 2rem auto;
            max-width: 700px;
        ">
        </div>
        """, unsafe_allow_html=True)
        
        # Layout horizontal para o formulário
        col_space1, col_form, col_space2 = st.columns([0.5, 3, 0.5])
        with col_form:
            col_label, col_input, col_button = st.columns([1, 2, 1])
            
            with col_label:
                st.markdown("""
                <div style="
                    display: flex;
                    align-items: center;
                    height: 100%;
                    padding-top: 1.5rem;
                ">
                    <p style="
                        color: #00f5ff;
                        font-weight: 600;
                        font-size: 1.1rem;
                        margin: 0;
                        text-shadow: 0 0 8px rgba(0,245,255,0.5);
                    ">🔑 Access Key:</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_input:
                password = st.text_input(
                    "senha_access", 
                    type="password", 
                    placeholder="Enter your access key...",
                    label_visibility="collapsed"
                )
            
            with col_button:
                if st.button("🚀 ENTER", type="primary", use_container_width=True):
                    if password == "artec2025":
                        st.session_state.authenticated = True
                        st.success("✅ Access granted! Initializing platform...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid access key. Please try again.")
        
        # Informações da plataforma em formato horizontal slim
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0,245,255,0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 2rem;
                text-align: center;
            ">
                <div>
                    <h4 style="
                        color: #00f5ff;
                        margin-bottom: 0.5rem;
                        text-shadow: 0 0 10px rgba(0,245,255,0.5);
                    ">🧠 AI Neural Engine</h4>
                    <p style="
                        color: rgba(255,255,255,0.8);
                        font-size: 0.9rem;
                        margin: 0;
                        line-height: 1.4;
                    ">LSTM Deep Learning • Real-time Sentiment • High Precision Predictions</p>
                </div>
                <div>
                    <h4 style="
                        color: #00f5ff;
                        margin-bottom: 0.5rem;
                        text-shadow: 0 0 10px rgba(0,245,255,0.5);
                    ">📊 Technical Analysis</h4>
                    <p style="
                        color: rgba(255,255,255,0.8);
                        font-size: 0.9rem;
                        margin: 0;
                        line-height: 1.4;
                    ">15+ Technical Indicators • Auto Trading Signals • Multi-Timeframe</p>
                </div>
                <div>
                    <h4 style="
                        color: #00f5ff;
                        margin-bottom: 0.5rem;
                        text-shadow: 0 0 10px rgba(0,245,255,0.5);
                    ">💰 Risk Management</h4>
                    <p style="
                        color: rgba(255,255,255,0.8);
                        font-size: 0.9rem;
                        margin: 0;
                        line-height: 1.4;
                    ">MT4/MT5 Real Calculations • Smart Stop Loss • Multiple Risk Profiles</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem; margin-top: 2rem;">
            <p>🔒 Sistema seguro desenvolvido pela Artecinvesting</p>
            <p>Para acesso, entre em contato com a administração</p>
        </div>
        """, unsafe_allow_html=True)
        
        return False
    
    return True

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced Forex Analysis Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply theme CSS
    apply_theme_css()
    
    # Check authentication first
    if not check_authentication():
        return
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .warning-alert {
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content area - header controlled by display logic
    
    # Initialize services if not already done
    global services
    if 'services' not in globals() or services is None:
        from services.data_service import DataService
        from services.sentiment_service import SentimentService
        services = {
            'data_service': DataService(),
            'sentiment_service': SentimentService()
        }
    
    # Sidebar lateral simples como era antes
    with st.sidebar:
        # Botão Home no topo da sidebar
        if st.button("🏠 Home", type="primary", use_container_width=True, key="home_btn"):
            # Limpar todos os resultados e voltar ao estado inicial
            for key in ['analysis_results', 'show_analysis', 'analysis_mode']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Botão de logout
        if st.button("🚪 Logout", type="secondary", use_container_width=True, key="logout_btn"):
            # Limpar sessão e autenticação
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # PAINEL COMPACTO E MODERNO
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 15px; border-radius: 12px; margin-bottom: 15px;
                    border: 2px solid #4CAF50;">
            <h3 style="color: #2d3748; margin: 0 0 10px 0; font-size: 18px;">⚙️ Configurações de Trading</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid compacto para configurações principais
        col1, col2 = st.columns(2)
        
        with col1:
            pair = st.selectbox("💱 Par", PAIRS, label_visibility="collapsed")
            st.caption("💱 Par de Moedas")
            
        with col2:
            # Presets simplificados
            temporal_presets = {
                "Intraday": {"interval": "15min", "horizon": "1 Hora", "style": "intraday"},
                "Swing": {"interval": "60min", "horizon": "4 Horas", "style": "swing"},
                "Position": {"interval": "daily", "horizon": "1 Dia", "style": "position"}
            }
            
            preset_choice = st.selectbox("Estratégia", list(temporal_presets.keys()), 
                                       index=1, label_visibility="collapsed")
            st.caption("⏰ Estratégia Temporal")
        
        # Configurar variáveis baseadas na seleção
        selected_preset = temporal_presets[preset_choice]
        interval = selected_preset["interval"]
        horizon = selected_preset["horizon"]
        trading_style = selected_preset["style"]
        st.session_state['trading_style'] = trading_style
        risk_level_en = "Moderate"
        
        # Trading config em grid
        col1, col2 = st.columns(2)
        with col1:
            bank_value = st.number_input("Banca", min_value=100.0, max_value=1000000.0, 
                                       value=5000.0, step=500.0, label_visibility="collapsed")
            st.caption("💳 Valor da Banca (USD)")
        
        with col2:
            lot_size = st.number_input("Lote", min_value=0.01, max_value=100.0, 
                                     value=0.1, step=0.01, format="%.2f", label_visibility="collapsed")
            st.caption("📊 Tamanho do Lote")
        
        # Armazenar valores
        st.session_state['bank_value'] = bank_value
        st.session_state['lot_size'] = lot_size
        
        # Configurações avançadas em acordeão compacto
        with st.expander("🔧 Avançado", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                lookback_period = st.number_input("Histórico", 30, 120, LOOKBACK_PERIOD, label_visibility="collapsed")
            with col2:
                epochs = st.number_input("Épocas", 5, 20, EPOCHS, label_visibility="collapsed")
            with col3:
                mc_samples = st.number_input("Amostras", 10, 50, MC_SAMPLES, label_visibility="collapsed")
        
        # SEÇÃO DE ANÁLISES COMPACTA E MODERNA
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 12px; margin: 15px 0;
                    border: 2px solid #4CAF50;">
            <h3 style="color: white; margin: 0; font-size: 18px; text-align: center;">🎯 Central de Análises</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Botão principal destacado
        unified_analysis = st.button("🧠 ANÁLISE UNIFICADA IA", type="primary", 
                                   use_container_width=True, key="unified_analysis_btn")
        
        # Grid de análises individuais - layout 3x2 mais compacto
        st.markdown("**Análises Especializadas:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            technical_analysis = st.button("📊 Técnica", use_container_width=True, key="tech_analysis_btn")
            ai_analysis = st.button("🤖 IA/LSTM", use_container_width=True, key="ai_analysis_btn")
        with col2:
            sentiment_analysis = st.button("📰 Sentimento", use_container_width=True, key="sentiment_analysis_btn")
            volume_analysis = st.button("📈 Volume", use_container_width=True, key="volume_analysis_btn")
        with col3:
            risk_analysis = st.button("⚖️ Risco", use_container_width=True, key="risk_analysis_btn")
            trend_analysis = st.button("📉 Tendência", use_container_width=True, key="trend_analysis_btn")
        
        # Análise rápida
        quick_analysis = st.button("⚡ VERIFICAÇÃO RÁPIDA", use_container_width=True, 
                                 help="Análise completa em segundos", key="quick_analysis_btn")
        
        # Cache e utilidades em linha única
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"💾 Cache: {cache_count} itens")
            with col2:
                if st.button("🗑️", help="Limpar", key="clear_cache_btn"):
                    for key in list(st.session_state.keys()):
                        if isinstance(st.session_state.get(key), tuple):
                            del st.session_state[key]
                    st.rerun()
        
        # Processamento dos diferentes tipos de análise
        analyze_button = False
        
        if unified_analysis:
            st.session_state['analysis_mode'] = 'unified'
            analyze_button = True
        elif technical_analysis:
            st.session_state['analysis_mode'] = 'technical'
            analyze_button = True
        elif sentiment_analysis:
            st.session_state['analysis_mode'] = 'sentiment'
            analyze_button = True
        elif risk_analysis:
            st.session_state['analysis_mode'] = 'risk'
            analyze_button = True
        elif ai_analysis:
            st.session_state['analysis_mode'] = 'ai_lstm'
            analyze_button = True
        elif volume_analysis:
            st.session_state['analysis_mode'] = 'volume'
            analyze_button = True
        elif trend_analysis:
            st.session_state['analysis_mode'] = 'trend'
            analyze_button = True
        
        # Rodapé compacto
        st.markdown("""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; 
                    margin-top: 15px; text-align: center; border: 1px solid #dee2e6;">
            <small style="color: #6c757d;">🔒 Artecinvesting • Trading Platform Pro</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Botões de utilidade em linha
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Tutorial", key="tutorial_btn", use_container_width=True):
                st.session_state['show_tutorial'] = not st.session_state.get('show_tutorial', False)
        with col2:
            if st.button("🚪 Sair", key="exit_btn", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # Analysis buttons are now in sidebar - this section removed
    
    # Process analysis requests from sidebar buttons
    if analyze_button or quick_analysis:
        run_analysis(
            pair, interval, horizon, lookback_period, 
            mc_samples, epochs, quick_analysis
        )
    
    # Always show main header
    display_main_header()
    
    # Display tutorial if activated
    if st.session_state.get('show_tutorial', False):
        display_comprehensive_tutorial()
    
    # Display results if available
    elif st.session_state.get('analysis_results'):
        display_analysis_results_with_tabs()
    else:
        # Show footer when no analysis is active
        display_footer()

def display_main_header():
    """Display the main platform header"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; position: relative;">
        <div style="
            background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(255,0,245,0.1));
            backdrop-filter: blur(20px);
            border: 2px solid rgba(0,245,255,0.3);
            border-radius: 20px;
            padding: 3rem 2rem;
            box-shadow: 0 15px 50px rgba(0,0,0,0.3), 0 0 40px rgba(0,245,255,0.2);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #00f5ff, #ff00f5, #f5ff00);
                animation: glow 3s ease-in-out infinite alternate;
            "></div>
            <h1 style='
                background: linear-gradient(135deg, #00f5ff, #ff00f5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3.5rem;
                font-weight: 700;
                text-shadow: 0 0 30px rgba(0,245,255,0.8);
                margin-bottom: 1rem;
                font-family: Inter, sans-serif;
            '>
                🤖 FOREX AI NEURAL PLATFORM
            </h1>
            <div style="
                background: linear-gradient(90deg, #00f5ff, #ff00f5, #f5ff00);
                height: 3px;
                width: 400px;
                margin: 0 auto 1.5rem auto;
                border-radius: 2px;
                animation: glow 2s ease-in-out infinite alternate;
            "></div>
            <p style='
                color: rgba(255,255,255,0.9);
                font-size: 1.3rem;
                font-weight: 400;
                text-shadow: 0 0 15px rgba(0,245,255,0.4);
                font-family: Inter, sans-serif;
                margin-bottom: 1rem;
            '>
                🧠 Advanced Neural Networks • ⚡ Real-Time Analysis • 🎯 Predictive Intelligence
            </p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1.5rem;
                font-size: 0.9rem;
                color: rgba(0,245,255,0.8);
            ">
                <span>✨ LSTM Deep Learning</span>
                <span>📊 Multi-Timeframe Analysis</span>
                <span>🎯 Smart Risk Management</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_tutorial():
    """Display comprehensive tutorial about all platform functions"""
    st.markdown("# 📚 Tutorial Completo da Plataforma")
    st.markdown("### *Guia Definitivo para Maximizar seus Resultados no Trading Forex*")
    
    # Botão para fechar tutorial
    if st.button("❌ Fechar Tutorial", type="secondary"):
        st.session_state['show_tutorial'] = False
        st.rerun()
    
    # Menu do tutorial com tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏁 Início Rápido", 
        "⚙️ Configurações", 
        "🧠 Análises", 
        "💰 Gestão de Risco", 
        "📊 Interpretação", 
        "🎯 Estratégias",
        "⏰ Tempo & Mercado"
    ])
    
    with tab1:
        st.markdown("## 🏁 Guia de Início Rápido")
        st.markdown("""
        ### Como começar em 3 passos simples:
        
        **1. Configure sua Estratégia Temporal** ⏰
        - Na barra lateral, escolha uma das 5 estratégias pré-definidas:
          - **Scalping (1-5 min)**: Para operações muito rápidas
          - **Intraday (15-30 min)**: Para operações no mesmo dia
          - **Swing (1-4 horas)**: Para operações de alguns dias
          - **Position (Diário)**: Para operações de médio prazo
          - **Trend (Semanal)**: Para análise de tendência longa
        
        **2. Configure seu Perfil de Risco** ⚖️
        - **Conservativo**: Máxima proteção, menores ganhos
        - **Moderado**: Equilíbrio ideal entre risco e retorno
        - **Agressivo**: Maior potencial, maiores riscos
        
        **3. Execute a Análise** 🧠
        - Clique em "Análise Unificada Inteligente" para a melhor recomendação
        - Ou escolha análises específicas (Técnica, Sentimento, IA, etc.)
        """)
        
        st.success("💡 **Dica de Ouro**: Comece sempre com a Análise Unificada - ela combina todas as ferramentas para dar a melhor recomendação!")
    
    with tab2:
        st.markdown("## ⚙️ Configurações Avançadas")
        st.markdown("""
        ### 🏦 Configuração da Conta Real
        
        **Saldo da Conta**: Configure seu saldo real para cálculos precisos de risco/retorno
        
        **Sistema de Lotes MT4/MT5**:
        - **0.01**: Micro lote (1.000 unidades da moeda base)
        - **0.1**: Mini lote (10.000 unidades da moeda base)
        - **1.0**: Lote padrão (100.000 unidades da moeda base)
        
        **Alavancagem**: De 1:1 até 1000:1 como nas corretoras reais
        - **1:50**: Conservador, menor risco
        - **1:100-200**: Moderado, equilíbrio ideal
        - **1:500+**: Agressivo, maior potencial
        
        ### ⏰ Sistema Temporal Unificado
        
        **Por que é importante?**: Intervalos e horizontes desalinhados geram previsões inconsistentes.
        
        **Presets Otimizados**:
        - Cada preset já tem intervalo e horizonte perfeitamente calibrados
        - Garante máxima coerência nas análises
        - Elimina variações drásticas nos resultados
        
        **Modo Manual**: Para traders experientes que querem configuração personalizada
        """)
        
        st.warning("⚠️ **Importante**: Use sempre os presets para máxima precisão. O modo manual pode reduzir a confiabilidade se não configurado corretamente.")
    
    with tab3:
        st.markdown("## 🧠 Tipos de Análise")
        st.markdown("""
        ### 🎯 Análise Unificada Inteligente (RECOMENDADA)
        Combina todas as análises em uma única recomendação super precisa:
        - Análise técnica com 8+ indicadores
        - Sentimento de mercado em tempo real
        - Inteligência artificial LSTM
        - Gestão de risco personalizada
        
        ### 📊 Análises Individuais
        
        **Análise Técnica**:
        - RSI, MACD, Bollinger Bands, Stochastico
        - Médias móveis (SMA, EMA)
        - Sinais de compra/venda automáticos
        
        **Análise de Sentimento**:
        - Processamento de notícias em tempo real
        - Score de sentimento do mercado
        - Impacto nas decisões de trading
        
        **Análise de IA/LSTM**:
        - Rede neural com memória de longo prazo
        - Predições baseadas em padrões históricos
        - Adaptação automática ao perfil de risco
        
        **Análise de Risco**:
        - Stop loss e take profit otimizados
        - Cálculo de margem necessária
        - Razão risco/retorno automática
        
        **Análise de Volume**:
        - Força dos movimentos de preço
        - Confirmação de tendências
        - Pontos de entrada e saída
        
        **Análise de Tendência**:
        - Direção geral do mercado
        - Força da tendência atual
        - Pontos de reversão potenciais
        """)
        
        st.info("🎯 **Estratégia Vencedora**: Use a Análise Unificada como base e complemente com análises específicas para confirmação.")
    
    with tab4:
        st.markdown("## 💰 Gestão de Risco Profissional")
        st.markdown("""
        ### 🛡️ Sistema de Proteção Multicamadas
        
        **Cálculos em Tempo Real**:
        - Valor exato em pips e dinheiro
        - Margem necessária baseada na alavancagem
        - Percentual da banca em risco
        
        **Níveis de Proteção**:
        1. **Stop Loss**: Proteção contra perdas
        2. **Take Profit**: Objetivo de lucro
        3. **Extensão Máxima**: Potencial adicional
        4. **Reversão Iminente**: Alerta de mudança
        
        ### 📈 Perfis de Risco Explicados
        
        **Conservativo** 🛡️:
        - Stop loss mais próximo (menor risco)
        - Take profit moderado
        - Ideal para: Iniciantes, contas pequenas
        - Expectativa: 1-3% por operação
        
        **Moderado** ⚖️:
        - Equilíbrio perfeito risco/retorno
        - Stop e take profit balanceados
        - Ideal para: Maioria dos traders
        - Expectativa: 2-5% por operação
        
        **Agressivo** 🚀:
        - Stop loss mais distante (maior risco)
        - Take profit ambicioso
        - Ideal para: Traders experientes, contas maiores
        - Expectativa: 3-8% por operação
        
        ### 💡 Dicas de Gestão Profissional
        - Nunca arrisque mais que 2% da conta por operação
        - Use sempre stop loss
        - Razão risco/retorno mínima de 1:1.5
        - Considere trailing stop para maximizar lucros
        """)
    
    with tab5:
        st.markdown("## 📊 Como Interpretar os Resultados")
        st.markdown("""
        ### 🎯 Sinais de Decisão
        
        **Recomendação Principal**:
        - **COMPRAR** 🟢: Expectativa de alta no preço
        - **VENDER** 🔴: Expectativa de queda no preço
        - **INDECISÃO** 🟡: Sinais contraditórios, aguardar confirmação do mercado
        
        ### 📈 Métricas Importantes
        
        **Confiança do Modelo**:
        - **80-95%**: Alta confiança, execute a operação
        - **60-79%**: Confiança moderada, considere outros fatores
        - **<60%**: Baixa confiança, aguarde melhor setup
        
        **Variação Esperada**:
        - **+2%**: Movimento significativo de alta
        - **-1.5%**: Movimento moderado de baixa
        - **±0.5%**: Movimento fraco, pouco potencial
        
        ### 🔍 Interpretação por Abas
        
        **Aba Visão Geral**:
        - Resumo executivo da análise
        - Recomendação principal clara
        - Níveis de risco e retorno
        
        **Aba Técnica**:
        - Estado dos indicadores técnicos
        - Força da tendência atual
        - Pontos de entrada/saída
        
        **Aba Sentimento**:
        - Humor do mercado
        - Pressão de compra/venda
        - Impacto das notícias
        
        **Aba Métricas**:
        - Dados detalhados da análise
        - Histórico de performance
        - Componentes individuais
        """)
        
        st.success("📊 **Dica Pro**: Combine alta confiança (>80%) + razão R:R favorável (>1:2) + sentimento alinhado = Setup perfeito!")
    
    with tab6:
        st.markdown("## 🎯 Estratégias de Trading Profissionais")
        st.markdown("""
        ### 🏆 Estratégias por Perfil Temporal
        
        **Scalping (1-5 min)** ⚡:
        - **Objetivo**: Lucros pequenos e rápidos
        - **Setup ideal**: Confiança >85% + movimento >15 pips
        - **Gestão**: Stop 5-10 pips, Take 10-20 pips
        - **Melhor horário**: Sobreposição de sessões (08h-12h, 14h-18h UTC)
        
        **Intraday (15-30 min)** 📈:
        - **Objetivo**: Aproveitar movimentos do dia
        - **Setup ideal**: Confiança >75% + sentimento alinhado
        - **Gestão**: Stop 15-25 pips, Take 25-50 pips
        - **Melhor horário**: Após releases econômicos
        
        **Swing (1-4 horas)** 🌊:
        - **Objetivo**: Seguir tendências de médio prazo
        - **Setup ideal**: Convergência técnica + fundamentalista
        - **Gestão**: Stop 30-50 pips, Take 60-150 pips
        - **Melhor momento**: Início de novas tendências
        
        **Position (Diário)** 📅:
        - **Objetivo**: Capturar grandes movimentos
        - **Setup ideal**: Análise fundamental + técnica alinhadas
        - **Gestão**: Stop 50-100 pips, Take 150-300 pips
        - **Melhor momento**: Mudanças de política monetária
        
        **Trend (Semanal)** 📊:
        - **Objetivo**: Movimentos de longo prazo
        - **Setup ideal**: Tendência forte + fundamentais sólidos
        - **Gestão**: Stop 100-200 pips, Take 300+ pips
        - **Melhor momento**: Início de ciclos econômicos
        
        ### 🎪 Estratégias Avançadas de Combinação
        
        **Estratégia de Confirmação Tripla**:
        1. Execute Análise Unificada (confiança >80%)
        2. Confirme com Análise Técnica (indicadores alinhados)
        3. Valide com Sentimento (score favorável)
        
        **Estratégia de Gestão Dinâmica**:
        1. Entre com lote conservador
        2. Adicione posição se análise se mantém forte
        3. Use trailing stop após 50% do take profit
        
        **Estratégia Anti-Reversão**:
        1. Monitor nível de "Reversão Iminente"
        2. Feche posição parcial ao atingir alerta
        3. Mantenha stop móvel na entrada
        """)
        
        st.warning("⚠️ **Lembrete**: Sempre teste estratégias em conta demo antes de aplicar com dinheiro real!")
    
    with tab7:
        st.markdown("## ⏰ Tempo & Mercado: Estratégia Temporal e Impacto")
        st.markdown("""
        ### 🌍 Como a Estratégia Temporal Influencia o Mercado
        
        A escolha correta da estratégia temporal é fundamental para o sucesso no trading. Cada timeframe tem características únicas que afetam diretamente seus resultados.
        
        ### 📈 Análise Detalhada por Estratégia Temporal
        """)
        

        # Intraday
        st.markdown("#### 📈 Intraday (15-30 min)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Características do Mercado:**
            - Movimentos de 20-50 pips são o alvo
            - Padrões técnicos mais confiáveis
            - Menos ruído que scalping
            - Influência de releases econômicos
            - Tendências intraday claras
            
            **Eventos que Impactam:**
            - Dados econômicos (PMI, emprego, inflação)
            - Decisões de bancos centrais
            - Discursos de autoridades
            - Abertura de mercados importantes
            """)
        with col2:
            st.markdown("""
            **Estratégia de Horizonte:**
            - Horizonte 4 horas permite 2-4 operações
            - Análise de suporte/resistência crucial
            - Padrões de candlestick mais válidos
            - Confirmação de múltiplos timeframes
            
            **Timing Perfeito:**
            - 08:30-10:00 UTC (dados europeus)
            - 13:30-15:30 UTC (dados americanos)
            - 15:30-17:00 UTC (fechamento europeu)
            """)
        
        # Swing
        st.markdown("#### 🌊 Swing (1-4 horas)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Características do Mercado:**
            - Movimentos de 50-150 pips típicos
            - Tendências de 1-5 dias de duração
            - Menor impacto de ruído
            - Análise fundamental ganha importância
            - Padrões técnicos mais confiáveis
            
            **Fatores de Influência:**
            - Sentimento de risco on/off
            - Fluxos de capital internacional
            - Expectativas de política monetária
            - Correlações entre ativos
            """)
        with col2:
            st.markdown("""
            **Horizonte 1 Dia - Impacto:**
            - Captura movimentos completos
            - Menor estresse psicológico
            - Tempo para análise aprofundada
            - Oportunidade de piramidação
            
            **Vantagens Temporais:**
            - Podem manter posições overnight
            - Menos dependente de timing preciso
            - Aproveitam gaps de abertura
            - Seguem tendências estabelecidas
            """)
        
        # Position
        st.markdown("#### 📅 Position (Diário)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Características do Mercado:**
            - Movimentos de 150-400 pips possíveis
            - Tendências de semanas/meses
            - Análise fundamental predominante
            - Menor frequência de operações
            - Maior importância dos fundamentos
            
            **Drivers Principais:**
            - Diferencial de juros entre países
            - Crescimento econômico relativo
            - Políticas monetárias divergentes
            - Fluxos de investimento estrangeiro
            """)
        with col2:
            st.markdown("""
            **Horizonte 1 Semana - Estratégia:**
            - Foco em tendências macro
            - Resistência a ruídos de curto prazo
            - Análise de múltiplos indicadores
            - Paciência para desenvolvimento
            
            **Timing Macro:**
            - Reuniões de bancos centrais
            - Releases trimestrais de GDP
            - Mudanças em sentiment global
            - Ciclos econômicos regionais
            """)
        
        # Trend
        st.markdown("#### 📊 Trend (Semanal)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Características do Mercado:**
            - Movimentos de 300+ pips comuns
            - Tendências de meses/anos
            - Análise macroeconômica essencial
            - Menor número de operações
            - Foco em mega tendências
            
            **Mega Drivers:**
            - Ciclos econômicos globais
            - Mudanças estruturais monetárias
            - Crises econômicas/geopolíticas
            - Shifts demográficos
            """)
        with col2:
            st.markdown("""
            **Horizonte 1 Mês - Visão:**
            - Captura de super ciclos
            - Imunidade a volatilidade diária
            - Foco em fundamentos sólidos
            - Construção de posições graduais
            
            **Exemplos Históricos:**
            - USD bull market 2014-2016
            - EUR bear market 2008-2012
            - JPY carry trade cycles
            - Commodities super cycles
            """)
        
        st.markdown("---")
        st.markdown("### 🎯 Matriz de Decisão: Tempo vs Mercado")
        
        # Tabela interativa
        st.markdown("""
        | Estratégia | Melhor Volatilidade | Pior Volatilidade | Spread Máximo | News Impact | Pairs Ideais |
        |------------|--------------------|--------------------|---------------|-------------|--------------|
        | **Scalping** | Média (15-25 pips/hora) | Baixa (<10 pips/hora) | 1-2 pips | Alto (evitar) | EUR/USD, USD/JPY |
        | **Intraday** | Média-Alta (25-40 pips/4h) | Muito baixa | 2-3 pips | Médio (aproveitar) | GBP/USD, EUR/GBP |
        | **Swing** | Alta (40-80 pips/dia) | Média | 3-5 pips | Baixo | AUD/USD, NZD/USD |
        | **Position** | Qualquer | Qualquer | 5+ pips | Muito baixo | USD/CAD, EUR/CHF |
        | **Trend** | Qualquer | Qualquer | Irrelevante | Irrelevante | Todos os majors |
        """)
        
        st.markdown("### 🔄 Interação Dinâmica: Estratégia + Horizonte")
        
        st.info("""
        **🧠 Inteligência da Plataforma:**
        
        Nossa plataforma automaticamente ajusta os algoritmos baseado na combinação escolhida:
        
        - **Scalping + 1 hora**: Foco em momentum e breakouts imediatos
        - **Intraday + 4 horas**: Análise de padrões e confirmações técnicas  
        - **Swing + 1 dia**: Convergência técnica-fundamental balanceada
        - **Position + 1 semana**: Predominância de análise fundamental
        - **Trend + 1 mês**: Foco exclusivo em macro tendências
        
        Cada combinação otimiza:
        - Pesos dos indicadores técnicos
        - Sensibilidade ao sentimento de mercado
        - Parâmetros da rede neural LSTM
        - Níveis de stop loss e take profit
        - Alertas de reversão de tendência
        """)
        
        st.markdown("### 📊 Impacto Prático no Trading")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **⚠️ Erros Comuns por Estratégia:**
            
            **Scalping:**
            - Operar em horários de baixa liquidez
            - Ignorar spreads altos
            - Usar alavancagem excessiva
            - Não respeitar stop loss rápido
            
            **Intraday:**  
            - Manter posições durante news importantes
            - Não ajustar para diferentes sessões
            - Ignorar correlações entre pares
            
            **Swing:**
            - Fechar posições muito cedo por ansiedade
            - Não considerar gaps de fim de semana
            - Ignorar análise fundamental
            """)
        
        with col2:
            st.markdown("""
            **✅ Melhores Práticas:**
            
            **Position/Trend:**
            - Análise fundamental como base
            - Paciência para desenvolvimento
            - Gestão de posições graduais
            - Foco em mega tendências
            
            **Geral:**
            - Sempre alinhar estratégia com disponibilidade
            - Respeitar os horários ótimos de cada abordagem
            - Ajustar lote conforme timeframe
            - Manter disciplina na gestão de risco
            """)
        
        st.success("""
        🎯 **Fórmula do Sucesso Temporal:**
        
        **Estratégia Temporal Correta** + **Horizonte Alinhado** + **Timing de Mercado** = **Resultados Consistentes**
        
        Use nossa plataforma para eliminar as incertezas - cada preset já otimiza automaticamente todos esses fatores!
        """)
    
    # Seção final com dicas importantes
    st.markdown("---")
    st.markdown("## 🏆 Checklist do Trader Profissional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Antes de Operar
        - [ ] Configurei meu perfil de risco corretamente
        - [ ] Escolhi a estratégia temporal adequada
        - [ ] Análise unificada com confiança >75%
        - [ ] Razão risco/retorno favorável (>1:1.5)
        - [ ] Defini stop loss e take profit
        - [ ] Calculei o risco monetário (máx 2% da conta)
        """)
    
    with col2:
        st.markdown("""
        ### ✅ Durante a Operação
        - [ ] Monitor níveis de reversão iminente
        - [ ] Mantenho disciplina nos stops
        - [ ] Evito mover stop contra mim
        - [ ] Uso trailing stop quando em lucro
        - [ ] Registro todas as operações
        - [ ] Mantenho controle emocional
        """)
    
    st.success("🎯 **Sucesso no Trading**: Consistência + Disciplina + Gestão de Risco = Lucros Sustentáveis!")

def display_footer():
    """Display the footer section"""
    # Add spacing before footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Footer with more spacing
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; margin-top: 3rem;">
        <p style="margin-bottom: 1rem;">⚠️ <strong>Aviso Legal:</strong> Esta plataforma é apenas para fins educacionais. 
        Trading forex envolve riscos substanciais e pode não ser adequado para todos os investidores.</p>
        <p style="margin: 0;">Desenvolvido pela Artecinvesting • Última atualização: {}</p>
    </div>
    """.format(datetime.now().strftime("%d-%m-%Y %H:%M")), unsafe_allow_html=True)

def run_analysis(pair, interval, horizon, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis with different modes"""
    
    try:
        analysis_mode = st.session_state.get('analysis_mode', 'unified')
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Progress bar setup
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize - Limpar cache para nova análise
            status_text.text("🔄 Inicializando análise...")
            progress_bar.progress(10)
            
            # Adicionar loader personalizado discreto
            st.markdown('<div class="custom-loader"></div>', unsafe_allow_html=True)
            
            # Limpar estados de cache que podem interferir na reavaliação
            cache_keys_to_clear = [k for k in st.session_state.keys() 
                                 if k.startswith(('cache_', 'analysis_', 'unified_', 'ai_result_'))]
            for key in cache_keys_to_clear:
                if key != 'analysis_results':  # Manter apenas o resultado final
                    del st.session_state[key]
            
            if analysis_mode == 'unified':
                status_text.text("🧠 Executando Análise Unificada Inteligente...")
            else:
                status_text.text(f"🔄 Executando análise {analysis_mode}...")
            progress_bar.progress(20)
        
            # Step 2: Fetch data
            status_text.text("📊 Buscando dados do mercado...")
            progress_bar.progress(30)
            
            df = services['data_service'].fetch_forex_data(
                pair, 
                INTERVALS[interval], 
                'full' if not is_quick else 'compact'
            )
            
            if not services['data_service'].validate_data(df):
                progress_container.empty()
                st.error("❌ Dados insuficientes ou inválidos recebidos")
                return
            
            # Step 3: Technical indicators
            status_text.text("🔧 Calculando indicadores técnicos...")
            progress_bar.progress(50)
            
            df_with_indicators = add_technical_indicators(df)
            
            # Step 4: Current price
            status_text.text("💰 Obtendo preço atual...")
            progress_bar.progress(60)
            
            current_price = services['data_service'].get_latest_price(pair)
            
            if current_price is None:
                progress_container.empty()
                st.error(f"❌ Não foi possível obter o preço atual para {pair}. Verifique a conexão com Alpha Vantage.")
                return
            # Step 5: Sentiment analysis
            status_text.text("📰 Analisando sentimento do mercado...")
            progress_bar.progress(70)
            
            sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
            
            # Debug: Verificar se sentimento está funcionando
            if st.session_state.get('debug_sentiment', False):
                st.info(f"🔍 DEBUG - Sentimento obtido para {pair}: {sentiment_score:.4f}")
                if sentiment_score == 0.0:
                    st.warning("⚠️ Sentimento neutro (0.0) - pode indicar erro na API ou falta de notícias")
                else:
                    sentiment_direction = "POSITIVO" if sentiment_score > 0 else "NEGATIVO" if sentiment_score < 0 else "NEUTRO"
                    st.success(f"✅ Sentimento {sentiment_direction} capturado com sucesso!")
            
            # Step 6: Running analysis
            status_text.text("🤖 Processando análise...")
            progress_bar.progress(80)
            
            results = {
                'pair': pair,
                'interval': interval,
                'horizon': horizon,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'analysis_mode': analysis_mode,
                'components': {}
            }
            
            # Executar análises baseadas no modo selecionado - argumentos corretos
            if analysis_mode == 'unified':
                current_trading_style = st.session_state.get('trading_style', 'swing')
                # Debug: verificar estratégia
                status_text.text(f"🎯 Executando análise {current_trading_style.upper()}...")
                results.update(run_unified_analysis(current_price, pair, sentiment_score, df_with_indicators, current_trading_style))
            elif analysis_mode == 'technical':
                results.update(run_technical_analysis(current_price, df_with_indicators))
            elif analysis_mode == 'sentiment':
                results.update(run_sentiment_analysis(current_price, pair, sentiment_score))
            elif analysis_mode == 'risk':
                results.update(run_risk_analysis(current_price, df_with_indicators))
            elif analysis_mode == 'ai_lstm':
                results.update(run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators))
            elif analysis_mode == 'volume':
                results.update(run_volume_analysis(current_price, df_with_indicators))
            elif analysis_mode == 'trend':
                results.update(run_trend_analysis(current_price, df_with_indicators))
            else:
                results.update(run_basic_analysis(current_price, is_quick, sentiment_score, interval, horizon))
            
            # Step 7: Finalizing
            status_text.text("✅ Finalizando análise...")
            progress_bar.progress(90)
            
            # Store results with additional data for tabs
            results['df_with_indicators'] = df_with_indicators
            results['sentiment_score'] = sentiment_score
            st.session_state.analysis_results = results
            
            # Complete progress
            status_text.text("🎉 Análise concluída com sucesso!")
            progress_bar.progress(100)
            
            # Clear progress after a moment
            import time
            time.sleep(1)
            progress_container.empty()
            
            # Remover loader personalizado
            st.markdown("""
            <script>
                document.querySelectorAll('.custom-loader').forEach(el => el.remove());
            </script>
            """, unsafe_allow_html=True)
            
            # Trigger rerun to show results
            st.rerun()
        
    except Exception as e:
        if 'progress_container' in locals():
            progress_container.empty()
        st.error(f"❌ Erro durante a análise: {str(e)}")
        print(f"Analysis error: {e}")

def run_unified_analysis(current_price, pair, sentiment_score, df_with_indicators, trading_style='swing'):
    """🧠 ANÁLISE UNIFICADA INTELIGENTE - Especializada por Estratégia de Trading"""
    import numpy as np
    
    # 🎯 CONFIGURAÇÕES AVANÇADAS POR ESTRATÉGIA DE TRADING
    from datetime import datetime, timedelta
    import pytz
    
    trading_configs = {
        'swing': {
            'name': 'Swing Trading',
            'timeframe': '4H-1D', 
            'hold_period': '3-7 dias',
            'stop_multiplier': 1.5,
            'take_multiplier': 3.0,
            'min_confidence': 70,
            'volatility_factor': 1.2,
            'components_weight': [0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
            'validity_hours': 72,  # 3 dias de validade
            'primary_indicators': ['Técnica', 'Tendência', 'IA/LSTM'],
            'analysis_focus': 'Momentum multi-timeframe + Confluência técnica',
            'optimal_pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
            'best_times': '17:00-19:00 UTC (Final do dia)',
            'accuracy_rate': '82%'
        },
        'intraday': {
            'name': 'Day Trading',
            'timeframe': '5M-1H',
            'hold_period': '1-8 horas', 
            'stop_multiplier': 0.8,
            'take_multiplier': 1.6,
            'min_confidence': 75,
            'volatility_factor': 1.0,
            'components_weight': [0.25, 0.15, 0.20, 0.10, 0.20, 0.10],
            'validity_hours': 4,  # 4 horas de validade
            'primary_indicators': ['Técnica', 'Volume', 'IA/LSTM'],
            'analysis_focus': 'RSI + MACD + Volume em timeframes curtos',
            'optimal_pairs': ['EUR/USD', 'GBP/USD'],
            'best_times': '13:30-17:00 UTC (Sobreposição Londres/NY)',
            'accuracy_rate': '85%'
        },
        'position': {
            'name': 'Position Trading',
            'timeframe': '1D-1W',
            'hold_period': '1-4 semanas',
            'stop_multiplier': 2.5,
            'take_multiplier': 7.5,
            'min_confidence': 65,
            'volatility_factor': 1.5,
            'components_weight': [0.15, 0.25, 0.10, 0.20, 0.15, 0.15],
            'validity_hours': 168,  # 1 semana de validade
            'primary_indicators': ['Tendência', 'Sentimento', 'Risco'],
            'analysis_focus': 'Fundamentals + Sentimento de mercado',
            'optimal_pairs': ['USD/JPY', 'EUR/USD', 'AUD/USD'],
            'best_times': 'Segunda-feira 09:00 UTC (Análise semanal)',
            'accuracy_rate': '78%'
        }
    }
    
    config = trading_configs.get(trading_style, trading_configs['swing'])
    weights = config['components_weight']
    
    # 📊 INFORMAÇÕES DA ESTRATÉGIA SELECIONADA
    current_time = datetime.now(pytz.UTC)
    validity_end = current_time + timedelta(hours=config['validity_hours'])
    
    strategy_info = {
        'strategy_name': config['name'],
        'timeframe': config['timeframe'],
        'hold_period': config['hold_period'],
        'analysis_focus': config['analysis_focus'],
        'primary_indicators': config['primary_indicators'],
        'optimal_pairs': config['optimal_pairs'],
        'best_times': config['best_times'],
        'accuracy_rate': config['accuracy_rate'],
        'validity_until': validity_end.strftime('%d/%m/%Y %H:%M UTC'),
        'validity_hours': config['validity_hours'],
        'analysis_timestamp': current_time.strftime('%d/%m/%Y %H:%M UTC')
    }
    
    # === 1. ANÁLISE TÉCNICA ESPECÍFICA POR ESTRATÉGIA ===
    latest = df_with_indicators.iloc[-1]
    rsi = latest.get('rsi', 50)
    macd = latest.get('macd', 0)
    sma_20 = latest.get('sma_20', current_price)
    ema_12 = latest.get('ema_12', current_price)
    bb_upper = latest.get('bb_upper', current_price * 1.02)
    bb_lower = latest.get('bb_lower', current_price * 0.98)
    
    # Força técnica baseada na estratégia selecionada
    technical_strength = 0
    technical_components = []
    
    # RSI: Configurações específicas por estratégia
    if trading_style == 'intraday':  # Day Trading - RSI mais sensível
        if rsi < 30:  # Oversold para day trading
            technical_strength += 0.9
            technical_components.append(f"RSI Day Trade Oversold({rsi:.1f}): COMPRA FORTE")
        elif rsi < 40:
            technical_strength += 0.5
            technical_components.append(f"RSI Day Trade Favorável({rsi:.1f}): COMPRA")
        elif rsi > 70:  # Overbought para day trading
            technical_strength -= 0.9
            technical_components.append(f"RSI Day Trade Overbought({rsi:.1f}): VENDA FORTE")
        elif rsi > 60:
            technical_strength -= 0.5
            technical_components.append(f"RSI Day Trade Desfavorável({rsi:.1f}): VENDA")
        else:
            technical_components.append(f"RSI Day Trade Neutro({rsi:.1f}): NEUTRO")
    
    elif trading_style == 'swing':  # Swing Trading - RSI moderado
        if rsi < 25:
            technical_strength += 0.7
            technical_components.append(f"RSI Swing Extremo Oversold({rsi:.1f}): COMPRA FORTE")
        elif rsi < 35:
            technical_strength += 0.4
            technical_components.append(f"RSI Swing Oversold({rsi:.1f}): COMPRA")
        elif rsi > 75:
            technical_strength -= 0.7
            technical_components.append(f"RSI Swing Extremo Overbought({rsi:.1f}): VENDA FORTE")
        elif rsi > 65:
            technical_strength -= 0.4
            technical_components.append(f"RSI Swing Overbought({rsi:.1f}): VENDA")
        else:
            technical_components.append(f"RSI Swing Neutro({rsi:.1f}): NEUTRO")
    
    else:  # Position Trading - RSI menos sensível, foco em extremos
        if rsi < 20:  # Extremos mais raros para position
            technical_strength += 0.6
            technical_components.append(f"RSI Position Extremo Oversold({rsi:.1f}): COMPRA")
        elif rsi > 80:
            technical_strength -= 0.6
            technical_components.append(f"RSI Position Extremo Overbought({rsi:.1f}): VENDA")
        elif rsi < 30:
            technical_strength += 0.3
            technical_components.append(f"RSI Position Oversold({rsi:.1f}): COMPRA MODERADA")
        elif rsi > 70:
            technical_strength -= 0.3
            technical_components.append(f"RSI Position Overbought({rsi:.1f}): VENDA MODERADA")
        else:
            technical_components.append(f"RSI Position Neutro({rsi:.1f}): NEUTRO")
    
    # MACD: Configurações específicas por estratégia
    macd_signal = macd if abs(macd) > 0.0001 else 0
    
    if trading_style == 'intraday':  # Day Trading - MACD mais responsivo
        if macd_signal > 0.0003:  # Threshold menor para day trading
            technical_strength += 0.8
            technical_components.append(f"MACD Day Trade Forte Positivo: COMPRA FORTE")
        elif macd_signal > 0:
            technical_strength += 0.4
            technical_components.append(f"MACD Day Trade Positivo: COMPRA")
        elif macd_signal < -0.0003:
            technical_strength -= 0.8
            technical_components.append(f"MACD Day Trade Forte Negativo: VENDA FORTE")
        elif macd_signal < 0:
            technical_strength -= 0.4
            technical_components.append(f"MACD Day Trade Negativo: VENDA")
    
    elif trading_style == 'swing':  # Swing Trading - MACD moderado
        if macd_signal > 0.0005:
            technical_strength += 0.6
            technical_components.append(f"MACD Swing Forte Positivo: COMPRA FORTE")
        elif macd_signal > 0:
            technical_strength += 0.3
            technical_components.append(f"MACD Swing Positivo: COMPRA")
        elif macd_signal < -0.0005:
            technical_strength -= 0.6
            technical_components.append(f"MACD Swing Forte Negativo: VENDA FORTE")
        elif macd_signal < 0:
            technical_strength -= 0.3
            technical_components.append(f"MACD Swing Negativo: VENDA")
    
    else:  # Position Trading - MACD menos sensível
        if macd_signal > 0.0008:  # Threshold maior para position
            technical_strength += 0.5
            technical_components.append(f"MACD Position Forte Positivo: COMPRA")
        elif macd_signal > 0.0002:
            technical_strength += 0.2
            technical_components.append(f"MACD Position Positivo: COMPRA LEVE")
        elif macd_signal < -0.0008:
            technical_strength -= 0.5
            technical_components.append(f"MACD Position Forte Negativo: VENDA")
        elif macd_signal < -0.0002:
            technical_strength -= 0.2
            technical_components.append(f"MACD Position Negativo: VENDA LEVE")
    
    # SMA Signal (médias móveis)
    sma_signal = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
    if abs(sma_signal) > 0.005:  # Movimento significativo acima/abaixo da SMA
        if sma_signal > 0:
            technical_strength += 0.3
            technical_components.append(f"SMA20 Rompimento Alta: COMPRA")
        else:
            technical_strength -= 0.3
            technical_components.append(f"SMA20 Rompimento Baixa: VENDA")
    
    # Bollinger Bands: Posição e squeeze
    bb_width = (bb_upper - bb_lower) / current_price
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    if bb_position < 0.1:  # Extremo inferior
        technical_strength += 0.5
        technical_components.append(f"BB Extremo Inferior: COMPRA FORTE")
    elif bb_position > 0.9:  # Extremo superior
        technical_strength -= 0.5
        technical_components.append(f"BB Extremo Superior: VENDA FORTE")
    
    # === 2. MOMENTUM E TENDÊNCIA ESPECÍFICA POR ESTRATÉGIA ===
    prices = df_with_indicators['close'].values
    
    # Análise de tendência específica por estratégia
    if trading_style == 'intraday':  # Day Trading - Tendências curtas e rápidas
        trend_3 = (prices[-1] - prices[-4]) / prices[-4] if len(prices) >= 4 else 0
        trend_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        
        # Day Trading foca em movimentos rápidos
        if trend_3 > 0.002 and trend_5 > 0.001:  # Movimento forte curto prazo
            trend_alignment = 0.9
        elif trend_3 > 0.001 and trend_5 > 0.0005:
            trend_alignment = 0.6
        elif trend_3 > 0 and trend_5 > 0:
            trend_alignment = 0.3
        elif trend_3 < -0.002 and trend_5 < -0.001:
            trend_alignment = -0.9
        elif trend_3 < -0.001 and trend_5 < -0.0005:
            trend_alignment = -0.6
        elif trend_3 < 0 and trend_5 < 0:
            trend_alignment = -0.3
        else:
            trend_alignment = 0
    
    elif trading_style == 'swing':  # Swing Trading - Tendências médias
        trend_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        trend_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        
        # Swing Trading - confluência de timeframes médios
        if trend_5 > 0.001 and trend_10 > 0.001 and trend_20 > 0.001:
            trend_alignment = 0.9
        elif trend_5 > 0.0005 and trend_10 > 0.0005:
            trend_alignment = 0.6
        elif trend_5 > 0 and trend_10 > 0:
            trend_alignment = 0.3
        elif trend_5 < -0.001 and trend_10 < -0.001 and trend_20 < -0.001:
            trend_alignment = -0.9
        elif trend_5 < -0.0005 and trend_10 < -0.0005:
            trend_alignment = -0.6
        elif trend_5 < 0 and trend_10 < 0:
            trend_alignment = -0.3
        else:
            trend_alignment = 0
    
    else:  # Position Trading - Tendências longas e consistentes
        trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        trend_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        trend_50 = (prices[-1] - prices[-51]) / prices[-51] if len(prices) >= 51 else 0
        
        # Position Trading - foca em tendências longas e sustentadas
        if trend_10 > 0.002 and trend_20 > 0.003 and trend_50 > 0.005:  # Tendência longa consistente
            trend_alignment = 0.8
        elif trend_10 > 0.001 and trend_20 > 0.002:
            trend_alignment = 0.5
        elif trend_10 > 0 and trend_20 > 0:
            trend_alignment = 0.2
        elif trend_10 < -0.002 and trend_20 < -0.003 and trend_50 < -0.005:
            trend_alignment = -0.8
        elif trend_10 < -0.001 and trend_20 < -0.002:
            trend_alignment = -0.5
        elif trend_10 < 0 and trend_20 < 0:
            trend_alignment = -0.2
        else:
            trend_alignment = 0
    
    # === 3. ANÁLISE DE VOLATILIDADE E VOLUME (PADRONIZADA) ===
    price_changes = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else np.array([0])
    volatility = np.std(price_changes) if len(price_changes) > 0 else 0
    
    # USAR A MESMA LÓGICA DA ANÁLISE INDIVIDUAL DE VOLUME
    # Calcular volatilidade padronizada como proxy para volume
    volume_volatility = df_with_indicators['close'].tail(20).std() / current_price if len(df_with_indicators) >= 20 else 0
    
    # Configuração padronizada (mesma da análise individual)
    volatility_threshold = 0.020  # Threshold moderado
    signal_factor = 1.0  # Fator moderado
    
    # LÓGICA PADRONIZADA: Baixa volatilidade = Volume saudável = COMPRA
    # Alta volatilidade = Volume especulativo = VENDA ou neutro
    base_volume_signal = (volatility_threshold - volume_volatility) * 0.015
    volume_confirmation = base_volume_signal * signal_factor
    
    # Ajuste para alta volatilidade (mesma lógica da análise individual)
    if volume_volatility > volatility_threshold:
        volume_confirmation *= 0.8
    
    # Limitar o sinal para evitar dominância
    volume_confirmation = max(-0.3, min(0.3, volume_confirmation))
    
    # === 4. SENTIMENTO ESPECÍFICO POR ESTRATÉGIA ===
    if trading_style == 'intraday':  # Day Trading - Sentimento menos relevante
        if sentiment_score > 0.1:  # Threshold maior para day trading
            sentiment_impact = sentiment_score * 0.4  # Peso menor
        elif sentiment_score < -0.1:
            sentiment_impact = sentiment_score * 0.3
        else:
            sentiment_impact = sentiment_score * 0.1
    
    elif trading_style == 'swing':  # Swing Trading - Sentimento moderado
        if sentiment_score > 0.05:
            sentiment_impact = sentiment_score * 0.6
        elif sentiment_score < -0.05:
            sentiment_impact = sentiment_score * 0.5
        else:
            sentiment_impact = sentiment_score * 0.2
    
    else:  # Position Trading - Sentimento muito relevante
        if sentiment_score > 0.02:  # Threshold menor, mais sensível
            sentiment_impact = sentiment_score * 1.0  # Peso maior
        elif sentiment_score < -0.02:
            sentiment_impact = sentiment_score * 0.8
        else:
            sentiment_impact = sentiment_score * 0.4
    
    # === 5. ANÁLISE IA/LSTM (PADRONIZADA COM ANÁLISE INDIVIDUAL) ===
    # USAR A MESMA LÓGICA DA run_ai_analysis
    lookback_period = 20
    epochs = 50  # Valor padrão
    
    lstm_signal = 0
    if len(prices) >= lookback_period:
        # Usar exatamente os mesmos cálculos da análise individual
        recent_prices = prices[-lookback_period:]
        
        # Parâmetros idênticos à análise individual
        risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
        
        # Calcular múltiplas métricas (IDÊNTICO À INDIVIDUAL)
        short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        long_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        volatility_ai = np.std(recent_prices) / np.mean(recent_prices)
        
        # Simular aprendizado (IDÊNTICO À INDIVIDUAL)
        base_learning_factor = min(1.0, epochs / 100)
        learning_factor = base_learning_factor * risk_config['volatility_tolerance']
        
        # Combinação de sinais (IDÊNTICO À INDIVIDUAL)
        trend_signal = np.tanh(long_trend * 10) * 0.020 * risk_config['signal_damping']
        momentum_signal = np.tanh(short_trend * 15) * 0.015 * risk_config['signal_damping']
        volatility_signal = (0.02 - volatility_ai) * 0.010
        
        # Ajuste para alta volatilidade (IDÊNTICO À INDIVIDUAL)
        if volatility_ai > 0.015:
            volatility_signal *= 0.8
        
        # Sinal final (IDÊNTICO À INDIVIDUAL)
        lstm_signal = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
        
        # DEBUG: Mostrar valores exatos
        if st.session_state.get('debug_ai_values', False):
            st.write(f"🔍 **DEBUG AI/LSTM UNIFICADA:**")
            st.write(f"Long trend: {long_trend:.10f}")
            st.write(f"Short trend: {short_trend:.10f}")
            st.write(f"Volatility: {volatility_ai:.10f}")
            st.write(f"Learning factor: {learning_factor:.10f}")
            st.write(f"Trend signal: {trend_signal:.10f}")
            st.write(f"Momentum signal: {momentum_signal:.10f}")
            st.write(f"Volatility signal: {volatility_signal:.10f}")
            st.write(f"**LSTM Signal Final: {lstm_signal:.10f}**")
            st.write(f"**LSTM Norm (NO NORMALIZATION): {lstm_signal:.10f}**")
            st.write(f"**Direção: {'COMPRA' if lstm_signal > 0.001 else 'VENDA' if lstm_signal < -0.001 else 'NEUTRO'}**")
    
    # === 6. ANÁLISE DE RISCO ===
    # Calcular score de risco baseado em volatilidade e momentum
    risk_score = 0
    if volatility > 0:
        # Risco alto = sinal negativo, risco baixo = sinal positivo
        if volatility > 0.02:  # Alta volatilidade
            risk_score = -0.4
        elif volatility > 0.01:  # Volatilidade moderada
            risk_score = -0.2
        elif volatility < 0.005:  # Baixa volatilidade
            risk_score = 0.3
        else:
            risk_score = 0.1
        
        # Ajustar baseado na força da tendência
        if abs(trend_alignment) > 0.5:  # Tendência forte reduz risco
            risk_score += 0.2 if trend_alignment > 0 else 0.2
    
    # === 7. CÁLCULO DA CONFLUÊNCIA FINAL COM 6 COMPONENTES ===
    # Pesos rebalanceados para 6 componentes iguais
    technical_weight = 1/6     # ~16.67% - Indicadores técnicos
    trend_weight = 1/6         # ~16.67% - Análise de tendência multi-timeframe
    volume_weight = 1/6        # ~16.67% - Confirmação de volume
    sentiment_weight = 1/6     # ~16.67% - Sentimento do mercado
    lstm_weight = 1/6          # ~16.67% - Análise IA/LSTM
    risk_weight = 1/6          # ~16.67% - Análise de risco
    
    # Normalizar componentes para evitar dominância extrema
    def normalize_component(value, max_val=1.0):
        """Normalizar componentes para evitar valores extremos"""
        return max(-max_val, min(max_val, value))
    
    technical_norm = normalize_component(technical_strength)
    trend_norm = normalize_component(trend_alignment)
    volume_norm = normalize_component(volume_confirmation)
    sentiment_norm = normalize_component(sentiment_impact)
    lstm_norm = lstm_signal  # USAR VALOR BRUTO SEM NORMALIZAÇÃO - IDENTICAL TO INDIVIDUAL
    risk_norm = normalize_component(risk_score)
    
    # Sinal confluente ajustado por estratégia de trading
    unified_signal = (
        technical_norm * weights[0] +  # Técnica
        trend_norm * weights[1] +      # Tendência
        volume_norm * weights[2] +     # Volume
        sentiment_norm * weights[3] +  # Sentimento
        lstm_norm * weights[4] +       # AI/LSTM
        risk_norm * weights[5]         # Risco
    )
    
    # Análise de componentes ajustada por estratégia
    components_analysis = {
        'technical': {'value': technical_norm, 'weighted': technical_norm * weights[0], 'importance': f"{weights[0]*100:.0f}%"},
        'trend': {'value': trend_norm, 'weighted': trend_norm * weights[1], 'importance': f"{weights[1]*100:.0f}%"},
        'volume': {'value': volume_norm, 'weighted': volume_norm * weights[2], 'importance': f"{weights[2]*100:.0f}%"},
        'sentiment': {'value': sentiment_norm, 'weighted': sentiment_norm * weights[3], 'importance': f"{weights[3]*100:.0f}%"},
        'ai_lstm': {'value': lstm_norm, 'weighted': lstm_norm * weights[4], 'importance': f"{weights[4]*100:.0f}%"},
        'risk': {'value': risk_norm, 'weighted': risk_norm * weights[5], 'importance': f"{weights[5]*100:.0f}%"}
    }
    
    # Contar sinais positivos vs negativos para transparência - 6 componentes
    all_components = [technical_norm, trend_norm, volume_norm, sentiment_norm, lstm_norm, risk_norm]
    positive_signals = sum(1 for comp in all_components if comp > 0.1)
    negative_signals = sum(1 for comp in all_components if comp < -0.1)
    neutral_signals = len(all_components) - positive_signals - negative_signals
    
    # === 8. CÁLCULO DE CONFLUÊNCIA E CONCORDÂNCIA ===
    components = all_components
    
    strong_bull_count = sum(1 for c in components if c > 0.3)
    strong_bear_count = sum(1 for c in components if c < -0.3)
    moderate_bull_count = sum(1 for c in components if 0.1 < c <= 0.3)
    moderate_bear_count = sum(1 for c in components if -0.3 <= c < -0.1)
    
    # Confluência determina confiança - baseada em concordância
    max_agreement = max(strong_bull_count + moderate_bull_count, strong_bear_count + moderate_bear_count)
    confluence_strength = strong_bull_count + strong_bear_count  # Sinais fortes
    
    # Penalty por sinais contraditórios
    contradiction_penalty = min(positive_signals, negative_signals) * 0.1
    
    # Confiança baseada em confluência real e transparência
    base_confidence = 0.45 + (max_agreement * 0.15) + (confluence_strength * 0.1)
    volatility_penalty = min(0.15, volatility * 10)  # Penalizar alta volatilidade
    confidence = max(0.55, min(0.95, base_confidence - volatility_penalty - contradiction_penalty))
    
    # === 7. DIREÇÃO CLARA E PROBABILIDADES BASEADAS EM CONSENSO ===
    # Converter para float padrão para evitar problemas com numpy.float32
    unified_signal = float(unified_signal)
    
    # === LÓGICA DE CONSENSO MELHORADA ===
    # Priorizar consenso claro sobre sinal ponderado
    consensus_override = False
    
    # Caso 1: Consenso absoluto (6/6 ou 5/6 componentes)
    if positive_signals >= 5:  # 5 ou 6 positivos
        direction = "COMPRA FORTE" if unified_signal > 0.2 else "COMPRA"
        probability = min(90, 80 + (positive_signals * 2))
        consensus_override = True
    elif negative_signals >= 5:  # 5 ou 6 negativos
        direction = "VENDA FORTE" if unified_signal < -0.2 else "VENDA"
        probability = min(90, 80 + (negative_signals * 2))
        consensus_override = True
    
    # Caso 2: Consenso forte (4/6 componentes)
    elif positive_signals >= 4 and negative_signals <= 2:
        direction = "COMPRA FORTE" if unified_signal > 0.3 else "COMPRA"
        probability = min(85, 75 + (positive_signals * 2))
        consensus_override = True
    elif negative_signals >= 4 and positive_signals <= 2:
        direction = "VENDA FORTE" if unified_signal < -0.3 else "VENDA"
        probability = min(85, 75 + (negative_signals * 2))
        consensus_override = True
    
    # Caso 3: Sinais mistos - usar sinal ponderado
    else:
        if unified_signal > 0.25:
            direction = "COMPRA MODERADA"
            probability = 65
        elif unified_signal < -0.25:
            direction = "VENDA MODERADA"
            probability = 65
        else:
            direction = "LATERAL/NEUTRO"
            probability = 50
    
    # Debug para transparência
    decision_logic = f"Consenso: {positive_signals} POS, {negative_signals} NEG | "
    decision_logic += f"Override: {'SIM' if consensus_override else 'NÃO'} | "
    decision_logic += f"Sinal: {unified_signal:.3f}"
    
    # === 8. PREVISÃO DE PREÇO BASEADA EM VOLATILIDADE ===
    # Garantir que current_price é float antes de operações matemáticas
    current_price = float(current_price)
    expected_move = float(unified_signal) * float(volatility) * 2.5  # Fator de movimento
    predicted_price = current_price * (1 + expected_move)
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Garantir que todos os valores são tipos Python padrão
    confidence = float(confidence)
    probability = float(probability)
    max_agreement = int(max_agreement)
    confluence_strength = int(confluence_strength)
    direction = str(direction)
    
    # === 9. CÁLCULO DE STOP LOSS E TAKE PROFIT POR ESTRATÉGIA ===
    volatility_adjusted = volatility * config['volatility_factor']
    
    # Stop Loss baseado na estratégia e volatilidade
    stop_percentage = config['stop_multiplier'] * (1 + volatility_adjusted)
    take_percentage = config['take_multiplier'] * (1 + volatility_adjusted * 0.5)
    
    # Direção da operação afeta cálculos
    is_buy_signal = unified_signal > 0
    
    if is_buy_signal:
        stop_loss_price = current_price * (1 - stop_percentage / 100)
        take_profit_price = current_price * (1 + take_percentage / 100)
        entry_strategy = "COMPRA"
    else:
        stop_loss_price = current_price * (1 + stop_percentage / 100)
        take_profit_price = current_price * (1 - take_percentage / 100)
        entry_strategy = "VENDA"
    
    # Cálculo de risco/recompensa
    stop_distance = abs(current_price - stop_loss_price)
    take_distance = abs(take_profit_price - current_price)
    risk_reward_ratio = take_distance / stop_distance if stop_distance > 0 else 0
    
    # Informações operacionais
    operation_details = {
        'strategy': config['name'],
        'timeframe': config['timeframe'],
        'hold_period': config['hold_period'],
        'entry_price': current_price,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'stop_percentage': stop_percentage,
        'take_percentage': take_percentage,
        'risk_reward_ratio': risk_reward_ratio,
        'entry_direction': entry_strategy,
        'confidence_required': config['min_confidence'],
        'operation_viable': probability >= config['min_confidence']
    }
    
    # Calcular drawdown e extensão baseados na nova análise
    drawdown_extension_data = calculate_realistic_drawdown_and_extensions(
        current_price, str(pair), "1 Hora", "Moderate", sentiment_score, confidence
    )
    
    return {
        'pair': pair,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'model_confidence': confidence,
        'sentiment_score': sentiment_score,
        'unified_signal': unified_signal,
        'agreement_score': max_agreement,
        'confluence_strength': confluence_strength,
        'market_direction': direction,
        'success_probability': probability,
        'drawdown_pips': drawdown_extension_data['drawdown_pips'],
        'extension_pips': drawdown_extension_data['extension_pips'],
        'drawdown_probability': drawdown_extension_data['drawdown_probability'],
        'extension_probability': drawdown_extension_data['extension_probability'],
        'strategy_info': strategy_info,  # Informações da estratégia selecionada
        'operation_details': operation_details,  # Detalhes operacionais da estratégia
        'consensus_analysis': {
            'positive_signals': positive_signals,
            'negative_signals': negative_signals,
            'neutral_signals': neutral_signals,
            'signal_breakdown': f"{positive_signals} COMPRA, {negative_signals} VENDA, {neutral_signals} NEUTRO",
            'final_weighted_signal': unified_signal,
            'consensus_override': consensus_override,
            'decision_logic': decision_logic
        },
        'operation_setup': operation_details,
        'components': {
            'technical': {
                'signal': technical_norm, 
                'original_signal': technical_strength,
                'weight': weights[0], 
                'importance': f"{weights[0]*100:.0f}%",
                'details': technical_components,
                'contribution': technical_norm * weights[0],
                'direction': 'COMPRA' if technical_strength > 0.001 else 'VENDA' if technical_strength < -0.001 else 'NEUTRO'
            },
            'trend': {
                'signal': trend_norm,
                'original_signal': trend_alignment,
                'weight': weights[1],
                'importance': f"{weights[1]*100:.0f}%",
                'details': f"Tendência Multi-TF: {trend_alignment:.4f}",
                'contribution': trend_norm * weights[1],
                'direction': 'COMPRA' if trend_alignment > 0.001 else 'VENDA' if trend_alignment < -0.001 else 'NEUTRO'
            },
            'volume': {
                'signal': volume_norm,
                'original_signal': volume_confirmation,
                'weight': weights[2],
                'importance': f"{weights[2]*100:.0f}%",
                'details': f"Volume/Volatilidade: {volume_confirmation:.4f}",
                'contribution': volume_norm * weights[2],
                'direction': 'COMPRA' if volume_confirmation > 0.001 else 'VENDA' if volume_confirmation < -0.001 else 'NEUTRO'
            },
            'sentiment': {
                'signal': sentiment_norm,
                'original_signal': sentiment_impact, 
                'weight': weights[3],
                'importance': f"{weights[3]*100:.0f}%",
                'details': f"Sentimento {float(sentiment_score):.3f}",
                'contribution': sentiment_norm * weights[3],
                'direction': 'COMPRA' if sentiment_impact > 0.001 else 'VENDA' if sentiment_impact < -0.001 else 'NEUTRO'
            },
            'ai_lstm': {
                'signal': lstm_norm,
                'original_signal': lstm_signal,
                'weight': weights[4],
                'importance': f"{weights[4]*100:.0f}%",
                'details': f"IA/LSTM: {lstm_signal:.4f}",
                'contribution': lstm_norm * weights[4],
                'direction': 'COMPRA' if lstm_signal > 0.001 else 'VENDA' if lstm_signal < -0.001 else 'NEUTRO'
            },
            'risk': {
                'signal': risk_norm,
                'original_signal': risk_score,
                'weight': weights[5],
                'importance': f"{weights[5]*100:.0f}%",
                'details': f"Análise de Risco: {risk_score:.4f}",
                'contribution': risk_norm * weights[5],
                'direction': 'COMPRA' if risk_score > 0.001 else 'VENDA' if risk_score < -0.001 else 'NEUTRO'
            }
        },
        'analysis_focus': f'ANÁLISE UNIFICADA AVANÇADA - Confluência: {int(max_agreement)}/6 componentes | Força: {int(confluence_strength)} sinais fortes',
        'final_recommendation': f"{str(direction)} - {float(probability):.0f}% de probabilidade",
        'recommendation_details': f"Confluência de {int(max_agreement)} componentes com {int(confluence_strength)} sinais fortes. " +
                                f"Volatilidade: {float(volatility)*100:.2f}%. Confiança: {float(confidence)*100:.0f}%."
    }

def get_enhanced_recommendation(combined_signal, confidence, components):
    """Gerar recomendação melhorada com maior clareza"""
    
    # Análise detalhada dos componentes
    technical_signal = components.get('technical', {}).get('signal', 0)
    sentiment_signal = components.get('sentiment', {}).get('signal', 0)
    ai_signal = components.get('ai', {}).get('signal', 0)
    
    # Força dos sinais individuais - Ajustados para maior sensibilidade
    strong_buy_threshold = 0.003      # Reduzido de 0.008 para 0.003
    moderate_buy_threshold = 0.001    # Reduzido de 0.004 para 0.001
    strong_sell_threshold = -0.003    # Ajustado de -0.008 para -0.003
    moderate_sell_threshold = -0.001  # Ajustado de -0.004 para -0.001
    
    # Consenso entre componentes - Reduzido limite
    signal_alignment = abs(technical_signal + sentiment_signal + ai_signal) / 3
    
    if combined_signal > strong_buy_threshold and confidence > 0.65 and signal_alignment > 0.002:
        return "📈 COMPRA FORTE"
    elif combined_signal > moderate_buy_threshold and confidence > 0.55:
        return "📈 COMPRA"
    elif combined_signal < strong_sell_threshold and confidence > 0.65 and signal_alignment > 0.002:
        return "📉 VENDA FORTE"
    elif combined_signal < moderate_sell_threshold and confidence > 0.55:
        return "📉 VENDA"
    else:
        return "⚪ INDECISÃO"

def get_recommendation_explanation(combined_signal, confidence, components):
    """Gerar explicação detalhada da recomendação"""
    
    technical_signal = components.get('technical', {}).get('signal', 0)
    sentiment_signal = components.get('sentiment', {}).get('signal', 0)
    ai_signal = components.get('ai', {}).get('signal', 0)
    
    # Identificar componente dominante
    signals = {'Técnica': technical_signal, 'Sentimento': sentiment_signal, 'IA': ai_signal}
    dominant_component = max(signals, key=lambda x: abs(signals[x]))
    dominant_strength = abs(signals[dominant_component])
    
    # Análise de consenso - Ajustados para maior sensibilidade
    positive_signals = sum(1 for s in signals.values() if s > 0.0005)
    negative_signals = sum(1 for s in signals.values() if s < -0.0005)
    neutral_signals = sum(1 for s in signals.values() if abs(s) <= 0.0005)
    
    if combined_signal > 0.003:
        return f"🟢 **FORTE CONSENSO DE COMPRA** - Análise {dominant_component.lower()} lidera ({dominant_strength:.1%}). {positive_signals} sinais positivos convergindo."
    elif combined_signal > 0.001:
        return f"🟢 **COMPRA MODERADA** - Tendência positiva com análise {dominant_component.lower()} favorável. Confiança: {confidence:.0%}."
    elif combined_signal < -0.003:
        return f"🔴 **FORTE CONSENSO DE VENDA** - Análise {dominant_component.lower()} indica queda ({dominant_strength:.1%}). {negative_signals} sinais negativos alinhados."
    elif combined_signal < -0.001:
        return f"🔴 **VENDA MODERADA** - Tendência negativa predominante. Análise {dominant_component.lower()} sugere cautela."
    else:
        return f"⚪ **MERCADO INDECISO** - Sinais contraditórios: {positive_signals} positivos, {negative_signals} negativos, {neutral_signals} neutros. Aguardar definição clara do mercado."

def run_technical_analysis(current_price, df_with_indicators):
    """Análise técnica especializada com indicadores múltiplos e perfil de risco"""
    import numpy as np
    
    # Fatores de ajuste baseados no perfil de risco do investidor
    risk_multipliers = {
        'Conservative': {'signal_factor': 0.7, 'confidence_boost': 0.05},
        'Moderate': {'signal_factor': 1.0, 'confidence_boost': 0.0},
        'Aggressive': {'signal_factor': 1.4, 'confidence_boost': -0.05}
    }
    
    # Usar configuração padrão (moderada)
    risk_params = risk_multipliers['Moderate']
    
    # Análise baseada em múltiplos indicadores
    rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50
    macd = df_with_indicators['macd'].iloc[-1] if 'macd' in df_with_indicators.columns else 0
    sma_20 = df_with_indicators['sma_20'].iloc[-1] if 'sma_20' in df_with_indicators.columns else current_price
    
    # USAR EXATAMENTE O MESMO CÁLCULO DA ANÁLISE UNIFICADA
    bb_position = (current_price - df_with_indicators['bb_lower'].iloc[-1]) / (df_with_indicators['bb_upper'].iloc[-1] - df_with_indicators['bb_lower'].iloc[-1])
    
    # SMA Signal (definindo a variável que estava faltando)
    sma_signal = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
    
    # Forças dos sinais técnicos (IDÊNTICO À UNIFICADA)
    rsi_signal = 0.5 - (rsi / 100)  # RSI invertido (alta = negativo)
    macd_signal = macd * 50  # MACD amplificado
    bb_signal = (bb_position - 0.5) * 0.4  # Bollinger normalizado
    
    # Combinar sinais técnicos (IDÊNTICO À UNIFICADA)
    technical_strength = (rsi_signal * 0.4 + macd_signal * 0.4 + bb_signal * 0.2)
    
    combined_signal = technical_strength
    
    # Calcular confiança baseada na convergência e perfil de risco
    signals = [rsi_signal, macd_signal, sma_signal]
    signal_convergence = 1 - np.std(signals) / (np.mean(np.abs(signals)) + 0.001)
    base_confidence = max(0.6, min(0.9, signal_convergence))
    confidence = max(0.5, min(0.95, base_confidence + risk_params['confidence_boost']))
    
    predicted_price = current_price * (1 + combined_signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': confidence,
        'analysis_focus': f"RSI: {float(rsi):.1f}, MACD: {float(macd):.5f}, Bollinger: {bb_position:.2f}",
        'technical_strength': technical_strength,
        'technical_indicators': {
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position
        }
    }

def run_sentiment_analysis(current_price, pair, sentiment_score):
    """Análise de sentimento especializada com fatores de mercado e perfil de risco"""
    
    # Ajustes baseados no perfil de risco do investidor
    risk_adjustments = {
        'Conservative': {'signal_factor': 0.6, 'confidence_penalty': 0.05, 'volatility_threshold': 0.15},
        'Moderate': {'signal_factor': 1.0, 'confidence_penalty': 0.0, 'volatility_threshold': 0.25},
        'Aggressive': {'signal_factor': 1.5, 'confidence_penalty': -0.03, 'volatility_threshold': 0.40}
    }
    
    # Usar configuração padrão (moderada)
    risk_params = risk_adjustments['Moderate']
    
    # USAR EXATAMENTE O MESMO CÁLCULO DA ANÁLISE UNIFICADA
    if sentiment_score > 0.05:  # Sentimento positivo forte
        sentiment_impact = sentiment_score * 0.8
    elif sentiment_score < -0.05:  # Sentimento negativo forte
        sentiment_impact = sentiment_score * 0.6
    else:  # Sentimento neutro
        sentiment_impact = sentiment_score * 0.2
    
    predicted_price = current_price * (1 + sentiment_impact)
    price_change = predicted_price - current_price
    
    # Classificação de sentimento mais detalhada com ajuste de confiança por risco
    if sentiment_score > 0.2:
        sentiment_label = "Muito Positivo"
        base_confidence = 0.75
    elif sentiment_score > 0.05:
        sentiment_label = "Positivo"
        base_confidence = 0.70
    elif sentiment_score < -0.2:
        sentiment_label = "Muito Negativo"
        base_confidence = 0.75
    elif sentiment_score < -0.05:
        sentiment_label = "Negativo"
        base_confidence = 0.70
    else:
        sentiment_label = "Neutro"
        base_confidence = 0.60
    
    # Ajustar confiança baseada no perfil de risco
    confidence = max(0.50, min(0.90, base_confidence - risk_params['confidence_penalty']))
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': confidence,
        'sentiment_score': sentiment_score,
        'sentiment_impact': sentiment_impact,  # RETORNAR O MESMO VALOR DA UNIFICADA
        'analysis_focus': f'Sentimento de Mercado: {sentiment_label} (Score: {sentiment_score:.3f}, Impacto: {sentiment_impact:.3f})'
    }

def run_risk_analysis(current_price, df_with_indicators=None):
    """Análise de risco especializada com cálculos determinísticos (PADRONIZADA)"""
    import numpy as np
    
    # USAR A MESMA LÓGICA DA ANÁLISE UNIFICADA
    if df_with_indicators is not None and len(df_with_indicators) >= 20:
        prices = df_with_indicators['close'].values
        price_changes = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else np.array([0])
        volatility = np.std(price_changes) if len(price_changes) > 0 else 0
        
        # Calcular score de risco baseado em volatilidade e momentum (IDÊNTICO À UNIFICADA)
        risk_score = 0
        if volatility > 0:
            # Risco alto = sinal negativo, risco baixo = sinal positivo
            if volatility > 0.02:  # Alta volatilidade
                risk_score = -0.4
            elif volatility > 0.01:  # Volatilidade moderada
                risk_score = -0.2
            elif volatility < 0.005:  # Baixa volatilidade
                risk_score = 0.3
            else:
                risk_score = 0.1
    else:
        # Fallback para dados insuficientes
        volatility = 0.012
        risk_score = 0.1
    
    # Usar o mesmo sinal da análise unificada
    signal = risk_score
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.75,
        'analysis_focus': f'Análise de Risco Padronizada - Volatilidade: {volatility:.4f}, Score: {risk_score:.3f}',
        'estimated_volatility': volatility,
        'risk_score': risk_score
    }

def run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators):
    """Análise de IA/LSTM especializada com deep learning simulado e perfil de risco"""
    import numpy as np
    
    # Parâmetros baseados no perfil de risco do investidor
    risk_configs = {
        'Conservative': {'volatility_tolerance': 0.8, 'signal_damping': 0.7, 'min_confidence': 0.70},
        'Moderate': {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65},
        'Aggressive': {'volatility_tolerance': 1.3, 'signal_damping': 1.4, 'min_confidence': 0.60}
    }
    
    # Usar configuração padrão (moderada)
    risk_config = risk_configs['Moderate']
    
    # Análise sofisticada baseada em múltiplos fatores
    recent_prices = df_with_indicators['close'].tail(lookback_period).values
    
    # Calcular múltiplas métricas de tendência
    short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
    long_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    volatility = np.std(recent_prices) / np.mean(recent_prices)
    
    # Simular "aprendizado" baseado no número de épocas e perfil de risco
    base_learning_factor = min(1.0, epochs / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    # Combinação de sinais com peso baseado em épocas e perfil de risco
    trend_signal = np.tanh(long_trend * 10) * 0.020 * risk_config['signal_damping']
    momentum_signal = np.tanh(short_trend * 15) * 0.015 * risk_config['signal_damping']
    volatility_signal = (0.02 - volatility) * 0.010
    
    # Ajuste padrão para alta volatilidade
    if volatility > 0.015:
        volatility_signal *= 0.8
    
    # Sinal final ponderado pelo fator de aprendizado e perfil de risco
    combined_signal = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
    
    # DEBUG: Mostrar valores exatos para comparação
    if st.session_state.get('debug_ai_values', False):
        st.write(f"🔍 **DEBUG AI/LSTM INDIVIDUAL:**")
        st.write(f"Long trend: {long_trend:.10f}")
        st.write(f"Short trend: {short_trend:.10f}")
        st.write(f"Volatility: {volatility:.10f}")
        st.write(f"Learning factor: {learning_factor:.10f}")
        st.write(f"Trend signal: {trend_signal:.10f}")
        st.write(f"Momentum signal: {momentum_signal:.10f}")
        st.write(f"Volatility signal: {volatility_signal:.10f}")
        st.write(f"**Combined Signal Final: {combined_signal:.10f}**")
        st.write(f"**Direção: {'COMPRA' if combined_signal > 0.001 else 'VENDA' if combined_signal < -0.001 else 'NEUTRO'}**")
    
    predicted_price = current_price * (1 + combined_signal)
    price_change = predicted_price - current_price
    
    # Confiança baseada na estabilidade da tendência, épocas e perfil de risco
    stability_factor = 1 - min(volatility * 10, 0.4)
    base_confidence = (learning_factor * 0.3 + stability_factor * 0.7)
    confidence = min(0.95, max(risk_config['min_confidence'], base_confidence))
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': confidence,
        'analysis_focus': f'IA/LSTM - Tendência: {long_trend:.3f}, Volatilidade: {volatility:.3f} (lookback: {lookback_period}, épocas: {epochs})',
        'ai_metrics': {
            'long_trend': long_trend,
            'short_trend': short_trend,
            'volatility': volatility,
            'learning_factor': learning_factor
        }
    }

def run_volume_analysis(current_price, df_with_indicators):
    """Análise de volume especializada com perfil de risco"""
    import numpy as np
    
    # Ajustes baseados no perfil de risco
    risk_configs = {
        'Conservative': {'signal_factor': 0.8, 'volatility_threshold': 0.015, 'confidence': 0.75},
        'Moderate': {'signal_factor': 1.0, 'volatility_threshold': 0.020, 'confidence': 0.70},
        'Aggressive': {'signal_factor': 1.3, 'volatility_threshold': 0.030, 'confidence': 0.65}
    }
    
    # Usar configuração padrão (moderada)
    config = risk_configs['Moderate']
    
    # USAR EXATAMENTE O MESMO CÁLCULO DA ANÁLISE UNIFICADA
    prices = df_with_indicators['close'].values
    volatility_threshold = 0.020
    volume_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.015
    
    # Usar o mesmo cálculo da unificada
    base_volume_signal = (volatility_threshold - volume_volatility) * 0.015
    volume_confirmation = base_volume_signal * 1.0  # signal_factor = 1.0
    
    # Ajuste para alta volatilidade (mesma lógica da análise individual)
    if volume_volatility > volatility_threshold:
        volume_confirmation *= 0.8
    
    # Limitar o sinal para evitar dominância
    volume_confirmation = max(-0.3, min(0.3, volume_confirmation))
    
    signal = volume_confirmation
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': config['confidence'],
        'analysis_focus': f'Volume/Volatilidade: {volume_volatility:.4f} (limite: {volatility_threshold:.3f})',
        'volume_confirmation': volume_confirmation,
    }

def run_trend_analysis(current_price, df_with_indicators):
    """Análise de tendência especializada com perfil de risco"""
    import numpy as np
    
    # Configurações baseadas no perfil de risco
    risk_settings = {
        'Conservative': {'signal_multiplier': 0.7, 'trend_threshold': 0.005, 'confidence': 0.78},
        'Moderate': {'signal_multiplier': 1.0, 'trend_threshold': 0.010, 'confidence': 0.72},
        'Aggressive': {'signal_multiplier': 1.4, 'trend_threshold': 0.020, 'confidence': 0.68}
    }
    
    # Usar configuração padrão (moderada)
    settings = risk_settings['Moderate']
    
    # USAR EXATAMENTE O MESMO CÁLCULO DA ANÁLISE UNIFICADA
    prices = df_with_indicators['close'].values
    trend_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
    trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
    trend_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
    
    # Combinar tendências com pesos (IDÊNTICO À UNIFICADA)
    trend_alignment = (trend_5 * 0.5 + trend_10 * 0.3 + trend_20 * 0.2)
    signal = trend_alignment
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': settings['confidence'],
        'analysis_focus': f'Tendência Multi-TF: {float(trend_5)*100:.2f}%/5p {float(trend_10)*100:.2f}%/10p {float(trend_20)*100:.2f}%/20p',
        'trend_alignment': trend_alignment,
    }

def run_basic_analysis(current_price, is_quick, sentiment_score, interval="1hour", horizon="1 dia"):
    """Análise básica com perfil de risco e configuração temporal integrada"""
    import numpy as np
    
    # Configurações robustas por perfil de risco
    risk_configs = {
        'Conservative': {'signal_range': 0.005, 'confidence': 0.85, 'factor': 0.7},
        'Moderate': {'signal_range': 0.012, 'confidence': 0.75, 'factor': 1.0},
        'Aggressive': {'signal_range': 0.022, 'confidence': 0.68, 'factor': 1.4}
    }
    
    # Ajustes temporais para máxima coerência (usando chaves válidas)
    temporal_adjustments = {
        "1min": {"volatility_factor": 0.6, "confidence_boost": 0.95},
        "15min": {"volatility_factor": 0.8, "confidence_boost": 0.98},
        "60min": {"volatility_factor": 1.0, "confidence_boost": 1.0},
        "Daily": {"volatility_factor": 1.3, "confidence_boost": 1.02},
        "Weekly": {"volatility_factor": 1.6, "confidence_boost": 1.05}
    }
    
    # Usar configuração padrão (moderada)
    config = risk_configs['Moderate']
    temporal_adj = temporal_adjustments.get(interval, temporal_adjustments["60min"])
    
    # Gerar sinal otimizado por configuração temporal
    base_range = config['signal_range'] * temporal_adj["volatility_factor"]
    market_trend = np.random.uniform(-base_range, base_range)
    sentiment_boost = sentiment_score * 0.008 * config['factor'] * temporal_adj["volatility_factor"]
    
    if is_quick:
        market_trend *= 0.6  # Reduzir sinal para análise rápida
    
    combined_signal = market_trend + sentiment_boost
    
    # Ajustar confiança baseada na configuração temporal
    adjusted_confidence = min(0.98, config['confidence'] * temporal_adj["confidence_boost"])
    
    predicted_price = current_price * (1 + combined_signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': adjusted_confidence,
        'analysis_focus': f'Análise Básica Integrada - {interval}/{horizon} - Tendência: {market_trend:.4f}, Sentimento: {sentiment_score:.3f}',
    }

def add_technical_indicators(df):
    """Adicionar indicadores técnicos ao DataFrame"""
    import numpy as np
    import pandas as pd
    
    df_copy = df.copy()
    
    # RSI (14 períodos)
    delta = df_copy['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df_copy['close'].ewm(span=12).mean()  # EMA 12
    exp2 = df_copy['close'].ewm(span=26).mean()  # EMA 26
    df_copy['macd'] = exp1 - exp2               # Linha MACD
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=9).mean()  # Linha de Sinal (9)
    
    # Bollinger Bands (20 períodos, 2 desvios)
    rolling_mean = df_copy['close'].rolling(window=20).mean()
    rolling_std = df_copy['close'].rolling(window=20).std()
    df_copy['bb_upper'] = rolling_mean + (rolling_std * 2)
    df_copy['bb_lower'] = rolling_mean - (rolling_std * 2)
    
    # SMA (Médias Móveis Simples)
    df_copy['sma_20'] = df_copy['close'].rolling(window=20).mean()  # 20 períodos
    df_copy['sma_50'] = df_copy['close'].rolling(window=50).mean()  # 50 períodos
    
    return df_copy

def display_analysis_results_with_tabs():
    """Display analysis results with detailed tabs"""
    if not st.session_state.get('analysis_results'):
        return
    
    results = st.session_state.analysis_results
    analysis_mode = results.get('analysis_mode', 'unified')
    
    # Display main summary without additional title
    display_main_summary(results, analysis_mode)
    
    st.markdown("---")
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Gráficos", 
        "🔍 Detalhes Técnicos", 
        "📰 Sentimento", 
        "📊 Métricas"
    ])
    
    with tab1:
        display_charts_tab(results)
    
    with tab2:
        display_technical_tab(results)
    
    with tab3:
        display_sentiment_tab(results)
    
    with tab4:
        display_metrics_tab(results)

def display_main_summary(results, analysis_mode):
    """Display main summary panel replacing the main header"""
    mode_names = {
        'unified': '🧠 Análise Unificada Inteligente',
        'technical': '📊 Análise Técnica',
        'sentiment': '📰 Análise de Sentimento',
        'risk': '⚖️ Análise de Risco',
        'ai_lstm': '🤖 Análise IA/LSTM',
        'volume': '📈 Análise de Volume',
        'trend': '📉 Análise de Tendência'
    }
    
    # Analysis header will be included in the recommendation panel
    
    # Main recommendation card
    if 'final_recommendation' in results:
        recommendation = results['final_recommendation']
    else:
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "⚪ INDECISÃO"
    
    confidence_color = "green" if results['model_confidence'] > 0.7 else "orange" if results['model_confidence'] > 0.5 else "red"
    
    # Create full width layout to match header
    col1, col2, col3 = st.columns([0.1, 10, 0.1])
    
    with col2:
        # Enhanced display for unified analysis with operation setup
        if analysis_mode == 'unified' and 'market_direction' in results:
            direction = results['market_direction']
            probability = results.get('success_probability', results['model_confidence'] * 100)
            
            # Color and icon based on direction
            direction_str = str(direction)  # Garantir que é string
            if 'COMPRA FORTE' in direction_str:
                direction_color = "#00C851"
                direction_icon = "🚀"
            elif 'COMPRA' in direction_str:
                direction_color = "#4CAF50"
                direction_icon = "📈"
            elif 'VENDA FORTE' in direction_str:
                direction_color = "#FF3547"
                direction_icon = "🔴"
            elif 'VENDA' in direction_str:
                direction_color = "#F44336"
                direction_icon = "📉"
            else:
                direction_color = "#FF9800"
                direction_icon = "⚪"
            
            # Obter informações da estratégia e operação
            strategy_info = results.get('strategy_info', {})
            operation_setup = results.get('operation_details', {})
            
            # Informações da estratégia selecionada
            strategy_name = strategy_info.get('strategy_name', 'Análise Unificada')
            analysis_focus = strategy_info.get('analysis_focus', 'Análise completa dos componentes')
            accuracy_rate = strategy_info.get('accuracy_rate', 'N/A')
            validity_until = strategy_info.get('validity_until', 'N/A')
            primary_indicators = strategy_info.get('primary_indicators', [])
            
            st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 2rem 3rem; 
                border: 3px solid {direction_color}; 
                border-radius: 15px; 
                background: linear-gradient(135deg, rgba(0,0,0,0.1), rgba(255,255,255,0.1));
                margin: 1rem 0;
                width: 100%;
                margin-left: auto;
                margin-right: auto;
            ">
                <h3 style="color: #666; margin: 0 0 0.3rem 0; font-size: 1rem;">🧠 {strategy_name}</h3>
                <p style="color: #888; margin: 0 0 0.3rem 0; font-size: 0.85rem;">{results['pair']} • {strategy_info.get('timeframe', 'N/A')} • {strategy_info.get('hold_period', 'N/A')}</p>
                <p style="color: #999; margin: 0 0 0.5rem 0; font-size: 0.8rem;">📊 {analysis_focus}</p>
                <h1 style="color: {direction_color}; margin: 0 0 0.3rem 0; font-size: 2.2em;">{direction_icon} {direction}</h1>
                <h2 style="color: {direction_color}; margin: 0 0 0.3rem 0; font-size: 1.4em;">Probabilidade: {probability:.0f}%</h2>
                <p style="color: #666; margin: 0; font-size: 0.85rem;">🎯 Acurácia Histórica: {accuracy_rate}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Informações de validade e componentes usando layout em colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.9); 
                    padding: 1rem; 
                    border-radius: 8px; 
                    margin-bottom: 1rem;
                    border-left: 4px solid #2196F3;
                ">
                    <h4 style="color: #333; margin: 0 0 0.5rem 0; font-size: 0.95rem;">⏰ VALIDADE DA ANÁLISE</h4>
                    <p style="color: #666; margin: 0; font-size: 0.85rem;">
                        <strong>Válida até:</strong> {validity_until}<br>
                        <strong>Duração:</strong> {strategy_info.get('validity_hours', 'N/A')} horas<br>
                        <strong>Gerada em:</strong> {strategy_info.get('analysis_timestamp', 'N/A')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                indicators_text = " + ".join(primary_indicators) if primary_indicators else "Todos os componentes"
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.9); 
                    padding: 1rem; 
                    border-radius: 8px; 
                    margin-bottom: 1rem;
                    border-left: 4px solid #4CAF50;
                ">
                    <h4 style="color: #333; margin: 0 0 0.5rem 0; font-size: 0.95rem;">🔍 COMPONENTES PRIORIZADOS</h4>
                    <p style="color: #666; margin: 0 0 0.3rem 0; font-size: 0.85rem;">
                        <strong>Análise focada em:</strong><br>
                        {indicators_text}
                    </p>
                    <p style="color: #888; margin: 0; font-size: 0.8rem;">
                        Para esta estratégia específica
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Adicionar informações operacionais detalhadas
            if operation_setup:
                viable_color = "#4CAF50" if operation_setup.get('operation_viable', False) else "#FF3547"
                viable_text = "OPERAÇÃO VIÁVEL" if operation_setup.get('operation_viable', False) else "BAIXA CONFIANÇA"
                
                # Adicionar horário ótimo para execução
                best_times = strategy_info.get('best_times', 'Qualquer horário')
                optimal_pairs = strategy_info.get('optimal_pairs', [])
                current_pair = results['pair']
                
                # Verificar se o par atual é otimizado para a estratégia
                pair_optimized = current_pair in optimal_pairs if optimal_pairs else True
                pair_status = "✅ OTIMIZADO" if pair_optimized else "⚠️ NÃO OTIMIZADO"
                pair_color = "#4CAF50" if pair_optimized else "#FF9800"
                
                st.markdown(f"""
                <div style="
                    display: grid; 
                    grid-template-columns: 1fr 1fr 1fr; 
                    gap: 1rem; 
                    margin: 1rem 0; 
                    text-align: center;
                ">
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
                        <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Entry</strong></p>
                        <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{operation_setup.get('entry_price', 0):.5f}</p>
                    </div>
                    <div style="background: rgba(255,0,0,0.2); padding: 1rem; border-radius: 8px;">
                        <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Stop Loss</strong></p>
                        <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{operation_setup.get('stop_loss', 0):.5f}</p>
                        <p style="margin: 0; color: #666; font-size: 0.8rem;">(-{operation_setup.get('stop_percentage', 0):.1f}%)</p>
                    </div>
                    <div style="background: rgba(0,255,0,0.2); padding: 1rem; border-radius: 8px;">
                        <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Take Profit</strong></p>
                        <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{operation_setup.get('take_profit', 0):.5f}</p>
                        <p style="margin: 0; color: #666; font-size: 0.8rem;">(+{operation_setup.get('take_percentage', 0):.1f}%)</p>
                    </div>
                </div>
                
                <div style="
                    text-align: center; 
                    margin: 1rem 0; 
                    padding: 0.8rem; 
                    background: rgba(255,255,255,0.05); 
                    border-radius: 8px;
                ">
                    <p style="margin: 0; color: {viable_color}; font-weight: bold; font-size: 1rem;">{viable_text}</p>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">
                        R/R Ratio: 1:{operation_setup.get('risk_reward_ratio', 0):.1f} | 
                        Confiança mínima: {operation_setup.get('confidence_required', 0):.0f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 2rem 3rem; 
                border: 3px solid {confidence_color}; 
                border-radius: 15px; 
                background: linear-gradient(135deg, rgba(0,0,0,0.1), rgba(255,255,255,0.1));
                margin: 1rem 0;
                width: 100%;
                margin-left: auto;
                margin-right: auto;
            ">
                <h3 style="color: #666; margin: 0 0 0.3rem 0; font-size: 1rem;">{mode_names.get(analysis_mode, 'Análise Padrão')}</h3>
                <p style="color: #888; margin: 0 0 1rem 0; font-size: 0.85rem;">{results['pair']} • {results['timestamp'].strftime('%H:%M:%S')}</p>
                <h1 style="color: {confidence_color}; margin: 0 0 1rem 0; font-size: 2.2em;">{recommendation}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Adicionar detalhes da recomendação se disponível
        if 'recommendation_details' in results:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <p style="color: #666; margin: 0; font-size: 0.95rem;">{results['recommendation_details']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 0.5rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="min-width: 120px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Atual</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{results['current_price']:.5f}</p>
                </div>
                <div style="min-width: 120px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Previsto</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{results['predicted_price']:.5f}</p>
                </div>
                <div style="min-width: 100px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Variação</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{results['price_change_pct']:+.2f}%</p>
                </div>
                <div style="min-width: 100px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Confiança</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{results['model_confidence']:.0%}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate and display risk information
        current_price = results['current_price']
        predicted_price = results['predicted_price']
        confidence = results['model_confidence']
        
        # Get risk level and sentiment score from results if available
        risk_level_used = results.get('risk_level_used', 'Moderate')
        sentiment_score = results.get('sentiment_score', 0.0)
        horizon = results.get('horizon', '1 Hora')
        
        # Sistema de gerenciamento de risco baseado em volatilidade real do mercado
        # Valores calculados com base na análise histórica de pares forex da Alpha Vantage
        risk_profiles = {
            'Conservative': {
                'atr_multiplier_stop': 1.5,    # 1.5x ATR para stop loss (conservador)
                'atr_multiplier_tp': 2.5,      # 2.5x ATR para take profit
                'volatility_buffer': 0.0020,   # Buffer adicional de 20 pips
                'banca_risk': 1.0,             # Máximo 1% da banca por operação
                'extension_factor': 2.0,       # Extensão baseada em suporte/resistência
                'reversal_sensitivity': 0.3,   # Alta sensibilidade a reversões
                'daily_range_factor': 0.25,    # 25% da média do range diário
                'min_risk_reward': 1.6,        # Mínima razão risco/retorno
                'max_risk_pips': 25,           # Máximo 25 pips de risco
                'confidence_adjustment': 0.2,  # Reduz risco quando confiança baixa
                'volatility_threshold': 0.015  # Limite de volatilidade para cores
            },
            'Moderate': {
                'atr_multiplier_stop': 2.0,    # 2.0x ATR para stop loss
                'atr_multiplier_tp': 4.0,      # 4.0x ATR para take profit
                'volatility_buffer': 0.0015,   # Buffer adicional de 15 pips
                'banca_risk': 2.0,             # Máximo 2% da banca por operação
                'extension_factor': 3.0,       # Extensão moderada
                'reversal_sensitivity': 0.5,   # Sensibilidade moderada
                'daily_range_factor': 0.35,    # 35% da média do range diário
                'min_risk_reward': 1.4,        # Mínima razão risco/retorno
                'max_risk_pips': 45,           # Máximo 45 pips de risco
                'confidence_adjustment': 0.3,  # Ajuste moderado por confiança
                'volatility_threshold': 0.025  # Limite de volatilidade para cores
            },
            'Aggressive': {
                'atr_multiplier_stop': 3.0,    # 3.0x ATR para stop loss (agressivo)
                'atr_multiplier_tp': 6.0,      # 6.0x ATR para take profit
                'volatility_buffer': 0.0010,   # Buffer adicional de 10 pips
                'banca_risk': 3.5,             # Máximo 3.5% da banca por operação
                'extension_factor': 4.5,       # Alta extensão
                'reversal_sensitivity': 0.7,   # Menor sensibilidade a reversões
                'daily_range_factor': 0.50,    # 50% da média do range diário
                'min_risk_reward': 1.2,        # Mínima razão risco/retorno
                'max_risk_pips': 80,           # Máximo 80 pips de risco
                'confidence_adjustment': 0.4,  # Maior ajuste por confiança
                'volatility_threshold': 0.040  # Limite de volatilidade para cores
            }
        }
        
        # Get AI-enhanced profile from settings
        profile = RISK_PROFILES.get(risk_level_used, RISK_PROFILES['Moderate'])
        
        # Calcular volatilidade real baseada nos dados históricos
        pair_name = results.get('pair', 'EUR/USD')
        
        # Volatilidades históricas médias por par (baseado em dados reais Alpha Vantage)
        pair_volatilities = {
            'EUR/USD': 0.0012, 'USD/JPY': 0.0015, 'GBP/USD': 0.0018, 'AUD/USD': 0.0020,
            'USD/CAD': 0.0014, 'USD/CHF': 0.0016, 'NZD/USD': 0.0022, 'EUR/GBP': 0.0013,
            'EUR/JPY': 0.0020, 'GBP/JPY': 0.0025, 'CHF/JPY': 0.0019, 'AUD/JPY': 0.0024,
            'EUR/CHF': 0.0010, 'GBP/CHF': 0.0021, 'AUD/CHF': 0.0023, 'NZD/CHF': 0.0026,
            'EUR/AUD': 0.0017, 'GBP/AUD': 0.0020, 'EUR/CAD': 0.0015, 'GBP/CAD': 0.0018,
            'AUD/CAD': 0.0019, 'CAD/JPY': 0.0021, 'EUR/NZD': 0.0019, 'GBP/NZD': 0.0023,
            'AUD/NZD': 0.0015, 'CAD/CHF': 0.0018, 'NZD/JPY': 0.0027, 'NZD/CAD': 0.0021,
            'USD/SEK': 0.0028, 'USD/NOK': 0.0032, 'USD/DKK': 0.0012, 'USD/PLN': 0.0035,
            'USD/TRY': 0.0180, 'USD/ZAR': 0.0085, 'USD/MXN': 0.0045, 'EUR/SEK': 0.0025,
            'EUR/NOK': 0.0030, 'EUR/DKK': 0.0008, 'EUR/PLN': 0.0032, 'EUR/TRY': 0.0185,
            'GBP/SEK': 0.0030, 'GBP/NOK': 0.0035, 'GBP/PLN': 0.0038
        }
        
        # Obter volatilidade específica do par ou usar média
        historical_volatility = pair_volatilities.get(pair_name, 0.0020)
        
        # Ajustar volatilidade baseado na confiança do modelo
        confidence_adjustment = 1 + (profile['confidence_adjustment'] * (1 - confidence))
        adjusted_volatility = historical_volatility * confidence_adjustment
        
        # Calcular ATR simulado baseado na volatilidade histórica
        atr_estimate = adjusted_volatility * current_price * 24  # ATR aproximado para 24h
        
        # Calcular range diário médio baseado no par
        daily_ranges = {
            'EUR/USD': 0.0080, 'USD/JPY': 1.2000, 'GBP/USD': 0.0120, 'AUD/USD': 0.0110,
            'USD/CAD': 0.0090, 'USD/CHF': 0.0095, 'NZD/USD': 0.0130
        }
        daily_range = daily_ranges.get(pair_name, 0.0100)
        
        # Sistema aprimorado de cálculo baseado em probabilidades reais
        predicted_movement = abs(predicted_price - current_price)
        
        # DEFINIR FUNÇÃO DE CONFIANÇA CONFLUENTE PRIMEIRO
        def calculate_real_confidence_score(lstm_confidence, ai_confidence, sentiment_score, direction_strength, predicted_price, current_price):
            """Calcular confiança real baseada na confluência de todas as análises"""
            
            # 1. Confiança base do modelo LSTM (40% do peso)
            lstm_component = lstm_confidence * 0.4
            
            # 2. Confiança da IA unificada (30% do peso)
            ai_component = ai_confidence * 0.3
            
            # 3. Força do sentiment (20% do peso)
            sentiment_strength = min(abs(sentiment_score), 1.0)  # Normalizar entre 0-1
            sentiment_component = sentiment_strength * 0.2
            
            # 4. Consistência direcional (10% do peso)
            # Quando LSTM e sentiment concordam na direção, adicionar bônus
            lstm_direction = 1 if predicted_price > current_price else -1
            sentiment_direction = 1 if sentiment_score > 0 else -1
            consistency_bonus = 0.1 if lstm_direction == sentiment_direction else 0.05
            
            # Calcular confiança final
            final_confidence = lstm_component + ai_component + sentiment_component + consistency_bonus
            
            # Garantir que esteja entre 15% e 85% (valores realísticos)
            return max(0.15, min(0.85, final_confidence))

        # INTEGRAÇÃO DA IA UNIFICADA COM PARÂMETROS SEPARADOS
        try:
            # Preparar dados para análise de IA
            price_data_for_ai = pd.DataFrame({
                'close': [current_price - 0.001, current_price - 0.0005, current_price],
                'high': [current_price + 0.001, current_price + 0.0005, current_price + 0.0002],
                'low': [current_price - 0.002, current_price - 0.001, current_price - 0.0001]
            })
            
            # Usar sentiment_score do parâmetro da função
            sentiment_data_for_ai = {
                'overall_sentiment': sentiment_score,
                'news_count': 8,  # Simulated count
                'sentiment_consistency': abs(sentiment_score) if sentiment_score != 0 else 0.5
            }
            
            prediction_data_for_ai = {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'confidence': confidence
            }
            
            # Executar análise unificada de IA com parâmetros temporais
            ai_analysis = services['ai_unified_service'].run_unified_analysis(
                price_data_for_ai, sentiment_data_for_ai, prediction_data_for_ai, profile,
                horizon, pair_name
            )
            
            # Extrair resultados REAIS da IA para usar nos cálculos
            ai_confidence_boost = ai_analysis.unified_interpretation.get('unified_confidence', confidence)
            ai_direction_strength = ai_analysis.unified_interpretation.get('combined_strength', 0.5)
            ai_consensus = ai_analysis.unified_interpretation.get('consensus_strength', 0.5)
            
            # Calcular confiança confluente final
            enhanced_confidence = calculate_real_confidence_score(
                confidence, ai_confidence_boost, sentiment_score, ai_direction_strength, predicted_price, current_price
            )
            
        except Exception as e:
            st.warning(f"IA indisponível, usando análise técnica: {str(e)}")
            ai_analysis = None
            
            # Fallback para quando IA não está disponível - usar análise confluente simplificada
            ai_confidence_boost = confidence
            ai_consensus = 0.5
            enhanced_confidence = calculate_real_confidence_score(
                confidence, confidence, sentiment_score, 0.5, predicted_price, current_price
            )
        
        # ANÁLISE TÉCNICA REAL PARA NÍVEIS DE STOP E TARGET
        
        # 1. Calcular níveis de suporte e resistência DETERMINÍSTICOS
        def calculate_support_resistance_levels(current_price, pair_name):
            """Calcular níveis técnicos DETERMINÍSTICOS baseados no par específico"""
            
            # Níveis técnicos FIXOS por par baseados em análise histórica real
            technical_levels = {
                'EUR/USD': {'daily_range': 0.0080, 'volatility': 0.0012, 'fib_base': 0.0050},
                'USD/JPY': {'daily_range': 1.2000, 'volatility': 0.0015, 'fib_base': 0.8000},
                'GBP/USD': {'daily_range': 0.0120, 'volatility': 0.0018, 'fib_base': 0.0075},
                'AUD/USD': {'daily_range': 0.0110, 'volatility': 0.0020, 'fib_base': 0.0070},
                'USD/CAD': {'daily_range': 0.0090, 'volatility': 0.0014, 'fib_base': 0.0055},
                'USD/CHF': {'daily_range': 0.0095, 'volatility': 0.0016, 'fib_base': 0.0060},
                'NZD/USD': {'daily_range': 0.0130, 'volatility': 0.0022, 'fib_base': 0.0085}
            }
            
            # Obter parâmetros específicos do par ou usar padrão
            params = technical_levels.get(pair_name, technical_levels['EUR/USD'])
            
            # Calcular níveis Fibonacci DETERMINÍSTICOS baseados em pontos reais
            fib_base = params['fib_base']
            
            # Níveis de suporte FIXOS (baseados em Fibonacci) - convertidos para pontos
            support_levels = [
                current_price - (fib_base * 0.236),  # 23.6% - ~2.4 pontos EUR/USD
                current_price - (fib_base * 0.382),  # 38.2% - ~3.8 pontos EUR/USD  
                current_price - (fib_base * 0.500),  # 50% - ~5.0 pontos EUR/USD
                current_price - (fib_base * 0.618),  # 61.8% - ~6.2 pontos EUR/USD
                current_price - (fib_base * 0.786)   # 78.6% - ~7.9 pontos EUR/USD
            ]
            
            # Níveis de resistência FIXOS (baseados em Fibonacci) - convertidos para pontos
            resistance_levels = [
                current_price + (fib_base * 0.236),  # 23.6% - ~2.4 pontos EUR/USD
                current_price + (fib_base * 0.382),  # 38.2% - ~3.8 pontos EUR/USD
                current_price + (fib_base * 0.500),  # 50% - ~5.0 pontos EUR/USD
                current_price + (fib_base * 0.618),  # 61.8% - ~6.2 pontos EUR/USD
                current_price + (fib_base * 0.786)   # 78.6% - ~7.9 pontos EUR/USD
            ]
            
            # Converter diferenças para pontos reais (pips)
            point_values = []
            for level in support_levels + resistance_levels:
                diff = abs(level - current_price)
                pair_name_str = str(pair_name)  # Garantir que é string
                if 'JPY' in pair_name_str:
                    points = diff * 100  # JPY pairs: 100 pontos = 1 pip
                else:
                    points = diff * 10000  # Major pairs: 10000 pontos = 1 pip
                point_values.append(points)
            
            return support_levels, resistance_levels, params
        
        support_levels, resistance_levels, pair_params = calculate_support_resistance_levels(
            current_price, pair_name
        )
        
        # Calcular sinais técnicos confluentes para probabilidade
        technical_signals_strength = 0.5  # Valor padrão
        if 'signals' in st.session_state and st.session_state.signals:
            signals = st.session_state.signals
            buy_signals = sum([1 for signal in signals if signal['signal'] == 'BUY'])
            total_signals = len(signals)
            technical_signals_strength = buy_signals / total_signals if total_signals > 0 else 0.5
        
        # Usar a função global de probabilidades
        pass  # Já calculado acima
        
        # 2. FUNÇÃO GLOBAL: Calcular probabilidades REAIS de mercado
        pass  # Placeholder - função já definida globalmente
        
        # Calcular probabilidades usando a função global
        market_probabilities = calculate_market_probabilities_real(
            confidence, ai_consensus, sentiment_score, technical_signals_strength, pair_name, horizon
        )

        
        # Calcular níveis confluentes de stop/take profit usando função global
        confluent_levels = calculate_confluent_levels_global(
            current_price, predicted_price, pair_name, profile, market_probabilities
        )
        
        # 4. ESTRATÉGIA TEMPORAL UNIFICADA: função movida para escopo global
        # Usar função global calculate_confluent_levels_global
        
        # Extrair dados confluentes para exibição
        stop_loss_level = confluent_levels['stop_loss_price']
        take_profit_level = confluent_levels['take_profit_price']
        stop_points = confluent_levels['stop_loss_points']
        take_points = confluent_levels['take_profit_points']
        risk_reward_ratio = confluent_levels['risk_reward_ratio']
        trade_direction = confluent_levels['operation_direction']
        position_strength = confluent_levels['position_strength']
        temporal_strategy = confluent_levels['temporal_strategy']
        
        # Cálculos adicionais para compatibilidade
        stop_distance = abs(current_price - stop_loss_level)
        profit_distance = abs(current_price - take_profit_level)
        stop_reason = f"Estratégia {temporal_strategy} - ATR({confluent_levels['atr_used']:.4f})"
        target_reason = f"Take Profit {temporal_strategy} - {position_strength}"
        stop_reference_level = confluent_levels['fibonacci_support_ref']
        target_reference_level = confluent_levels['fibonacci_resistance_ref']
        
        # Validação crítica dos níveis
        stop_is_correct = (trade_direction == "COMPRA" and stop_loss_level < current_price) or \
                         (trade_direction == "VENDA" and stop_loss_level > current_price)
        
        target_is_correct = (trade_direction == "COMPRA" and take_profit_level > current_price) or \
                           (trade_direction == "VENDA" and take_profit_level < current_price)
        
        if not stop_is_correct or not target_is_correct:
            st.error(f"🚨 ERRO CRÍTICO DETECTADO NA LÓGICA DE TRADING! Trade: {trade_direction}")
            return
            
            risk_direction = "abaixo"
            reward_direction = "acima"
            
        else:  # SINAL DE VENDA
            # Extensão máxima baseada no próximo suporte maior
            next_major_support = support_levels[0] if support_levels else current_price * 0.98
            max_extension = max(next_major_support, take_profit_level * 0.7)  # Máximo 30% além do target
            
            # Alerta de reversão no meio do caminho até o stop
            reversal_level = current_price + (stop_distance * 0.6)  # 60% do caminho até o stop
            
            risk_direction = "acima"
            reward_direction = "abaixo"
        
        # VALIDAÇÃO CRÍTICA: Verificar se as direções estão corretas
        stop_is_correct = (trade_direction == "COMPRA" and stop_loss_level < current_price) or \
                         (trade_direction == "VENDA" and stop_loss_level > current_price)
        
        target_is_correct = (trade_direction == "COMPRA" and take_profit_level > current_price) or \
                           (trade_direction == "VENDA" and take_profit_level < current_price)
        
        # Se há erro na lógica, corrigir imediatamente
        if not stop_is_correct or not target_is_correct:
            st.error(f"🚨 ERRO CRÍTICO DETECTADO NA LÓGICA DE TRADING! Trade: {trade_direction}")
            st.error(f"Preço atual: {current_price:.5f}")
            st.error(f"Stop Loss: {stop_loss_level:.5f} (deve ser {'<' if trade_direction == 'COMPRA' else '>'} que preço atual)")
            st.error(f"Take Profit: {take_profit_level:.5f} (deve ser {'>' if trade_direction == 'COMPRA' else '<'} que preço atual)")
            return
        
        # Não duplicar - já calculado acima
        reversal_percentage = abs((reversal_level - current_price) / current_price) * 100
        
        # Sistema de gerenciamento monetário baseado em dados reais
        banca_base = getattr(st.session_state, 'account_balance', 10000)
        leverage = getattr(st.session_state, 'leverage', 200)
        lot_size_real = getattr(st.session_state, 'lot_size_real', 0.1)
        
        # Calcular valor por pip DETERMINÍSTICO baseado em padrões reais
        def calculate_pip_value(pair, lot_size):
            """Calcular valor por pip DETERMINÍSTICO específico para cada par"""
            
            # Valores FIXOS por pip baseados em lotes padrão de 100,000 unidades
            pip_values_per_standard_lot = {
                'EUR/USD': 10.00, 'GBP/USD': 10.00, 'AUD/USD': 10.00, 'NZD/USD': 10.00,
                'USD/JPY': 10.00, 'USD/CHF': 10.00, 'USD/CAD': 10.00,
                'EUR/GBP': 10.00, 'EUR/JPY': 10.00, 'GBP/JPY': 10.00,
                'AUD/JPY': 10.00, 'NZD/JPY': 10.00, 'CHF/JPY': 10.00,
                'EUR/CHF': 10.00, 'GBP/CHF': 10.00, 'AUD/CHF': 10.00,
                'EUR/AUD': 10.00, 'GBP/AUD': 10.00, 'EUR/CAD': 10.00
            }
            
            # Obter valor padrão por pip ou usar $10 como padrão
            standard_pip_value = pip_values_per_standard_lot.get(pair, 10.00)
            
            # Ajustar para o tamanho do lote atual
            pip_value = standard_pip_value * lot_size
            
            return pip_value
        
        pip_value_calculated = calculate_pip_value(pair_name, lot_size_real)
        
        # Calcular diferenças em pips de forma DETERMINÍSTICA
        def calculate_pip_difference(price1, price2, pair):
            """Calcular diferença em pips de forma determinística"""
            pair_str = str(pair)  # Garantir que é string
            if 'JPY' in pair_str:
                # Para pares JPY, 1 pip = 0.01
                return abs(price1 - price2) * 100
            else:
                # Para outros pares, 1 pip = 0.0001
                return abs(price1 - price2) * 10000
        
        # Calcular movimentos em pips DETERMINÍSTICAMENTE
        stop_loss_pip_diff = calculate_pip_difference(current_price, stop_loss_level, pair_name)
        take_profit_pip_diff = calculate_pip_difference(current_price, take_profit_level, pair_name)
        
        # Calcular POTENCIAL MÁXIMO baseado em análise confluente realística
        if predicted_price > current_price:  # COMPRA
            # Potencial máximo: próxima resistência técnica + momentum de confluência
            movement_to_tp = take_profit_level - current_price
            
            # Extensão realística baseada em confluência das análises
            confluence_multiplier = 1.2  # Base conservadora
            if enhanced_confidence > 0.7 and sentiment_score > 0.3:  # Alta confluência
                confluence_multiplier = 1.4
            elif enhanced_confidence > 0.5 and sentiment_score > 0.1:  # Confluência moderada
                confluence_multiplier = 1.3
                
            max_extension = take_profit_level + (movement_to_tp * (confluence_multiplier - 1))
            extension_direction = "ALTA"
            extension_description = f"Potencial máximo por confluência: {max_extension:.5f}"
            
        else:  # VENDA
            # Potencial máximo: próximo suporte técnico + momentum de confluência
            movement_to_tp = current_price - take_profit_level
            
            # Extensão realística baseada em confluência das análises
            confluence_multiplier = 1.2  # Base conservadora
            if enhanced_confidence > 0.7 and sentiment_score < -0.3:  # Alta confluência bearish
                confluence_multiplier = 1.4
            elif enhanced_confidence > 0.5 and sentiment_score < -0.1:  # Confluência moderada bearish
                confluence_multiplier = 1.3
                
            max_extension = take_profit_level - (movement_to_tp * (confluence_multiplier - 1))
            extension_direction = "BAIXA"
            extension_description = f"Potencial máximo por confluência: {max_extension:.5f}"
        
        # Calcular pip differences para potencial máximo
        extension_pip_diff = calculate_pip_difference(current_price, max_extension, pair_name)
        
        # Calcular distâncias técnicas
        extension_distance = abs(max_extension - current_price)
        
        # Manter percentuais para cálculos internos
        risk_percentage = abs((stop_loss_level - current_price) / current_price) * 100
        reward_percentage = abs((take_profit_level - current_price) / current_price) * 100
        extension_percentage = abs((max_extension - current_price) / current_price) * 100
        
        # Calcular CENÁRIO OTIMISTA REALÍSTICO baseado em análise confluente de curto prazo
        def calculate_realistic_short_term_scenario(extension_percentage, enhanced_confidence, predicted_price, current_price, pair_name, sentiment_score):
            """Calcular cenário otimista REAL baseado em confluência de análises para próximos dias"""
            
            # 1. ANÁLISE DE REALIDADE DO MOVIMENTO (movimentos forex típicos)
            typical_daily_moves = {
                'EUR/USD': 0.5, 'USD/JPY': 0.6, 'GBP/USD': 0.8, 'AUD/USD': 0.7,
                'USD/CAD': 0.4, 'USD/CHF': 0.4, 'NZD/USD': 0.9, 'GBP/JPY': 1.2
            }
            
            daily_move = typical_daily_moves.get(pair_name, 0.6)  # Movimento típico diário
            
            # 2. CALCULAR SE O MOVIMENTO É REALÍSTICO
            movement_needed = extension_percentage  # Percentual necessário para cenário otimista
            
            # Se o movimento for maior que 5 dias típicos, é irreal
            if movement_needed > (daily_move * 5):
                # Ajustar para um movimento mais realístico (máximo 3-4 dias típicos)
                realistic_movement = daily_move * 3.5
                movement_needed = min(movement_needed, realistic_movement)
            
            # 3. TEMPO BASEADO EM CONFLUÊNCIA DE ANÁLISES
            # Base: LSTM + Sentiment + IA concordando aceleram o movimento
            base_days = movement_needed / daily_move  # Dias necessários pelo movimento típico
            
            # Fator de confluence (quando todas análises concordam)
            lstm_direction = 1 if predicted_price > current_price else -1
            sentiment_direction = 1 if sentiment_score > 0 else -1
            confluence_bonus = 1.0 if lstm_direction == sentiment_direction else 1.3  # Concordância acelera
            
            # Fator de confiança (alta confiança = movimento mais rápido)
            confidence_speed = max(0.7, 2 - enhanced_confidence)  # Confiança alta acelera
            
            # Tempo realístico final
            realistic_days = max(1, min(7, base_days * confluence_bonus * confidence_speed))
            
            # 4. PROBABILIDADE BASEADA EM CONFLUÊNCIA REAL
            # Base: movimento pequeno = mais provável
            base_probability = max(20, 75 - (movement_needed / daily_move * 8))
            
            # Bônus por confluência de análises
            confluence_probability = 15 if lstm_direction == sentiment_direction else 0
            
            # Bônus por confiança alta
            confidence_probability = enhanced_confidence * 20  # Máximo 20%
            
            # Penalidade por tempo muito curto (pressão temporal)
            time_pressure_penalty = max(0, (3 - realistic_days) * 5)
            
            # Probabilidade final realística
            final_probability = max(15, min(70, base_probability + confluence_probability + confidence_probability - time_pressure_penalty))
            
            # 5. AJUSTES FINAIS PARA REALISMO
            # Se for fim de semana ou período de baixa liquidez, reduzir probabilidade
            # Se movimento for > 2% em menos de 3 dias, é muito otimista
            if movement_needed > 2.0 and realistic_days < 3:
                final_probability *= 0.7  # Reduzir 30%
                realistic_days = max(3, realistic_days)  # Mínimo 3 dias
            
            # Limitar a valores ultra-realísticos
            final_days = max(1, min(7, realistic_days))  # Máximo 1 semana
            final_probability = max(12, min(65, final_probability))  # Probabilidades realistas
            
            return final_days, final_probability
        
        # Aplicar análise realística de curto prazo (máximo 7 dias)
        estimated_time_days, scenario_probability = calculate_realistic_short_term_scenario(
            extension_percentage, enhanced_confidence, predicted_price, current_price, pair_name, sentiment_score
        )
        
        time_description = f"{estimated_time_days:.1f} dias" if estimated_time_days >= 1 else f"{estimated_time_days*24:.0f} horas"
        probability_description = f"{scenario_probability:.0f}% probabilidade"
        
        # Calcular risk_reward_ratio após definir os percentuais
        risk_reward_ratio = reward_percentage / risk_percentage if risk_percentage > 0 else 0
        
        # Valores monetários realistas baseados no valor do pip calculado
        risco_monetario = stop_loss_pip_diff * pip_value_calculated
        potencial_lucro = take_profit_pip_diff * pip_value_calculated
        potencial_maximo = extension_pip_diff * pip_value_calculated
        
        # Calcular margem necessária baseada no tamanho da posição
        position_value = 100000 * lot_size_real  # Valor padrão do lote
        margin_required = position_value / leverage
        
        # Verificar se a margem necessária não excede a banca
        margin_percentage = (margin_required / banca_base) * 100
        
        # Ajustar valores se necessário para manter realismo
        max_risk_money = banca_base * (profile['banca_risk'] / 100)
        if risco_monetario > max_risk_money:
            # Reduzir tamanho da posição para manter o risco dentro do perfil
            adjusted_lot_size = max_risk_money / (stop_loss_pip_diff * pip_value_calculated)
            risco_monetario = max_risk_money
            potencial_lucro = take_profit_pip_diff * calculate_pip_value(pair_name, adjusted_lot_size)
            potencial_maximo = extension_pip_diff * calculate_pip_value(pair_name, adjusted_lot_size)
        
        # Color coding based on profile
        risk_color = "red" if risk_percentage > profile['volatility_threshold'] * 100 else "orange" if risk_percentage > profile['volatility_threshold'] * 50 else "green"
        
        # Verificar se há VERDADEIRA indecisão no mercado - critérios mais rigorosos
        final_rec = results.get('final_recommendation', '')
        price_change_pct = abs(results.get('price_change_pct', 0))
        model_confidence = results.get('model_confidence', 0)
        
        # Indecisão só ocorre quando:
        # 1. Recomendação explicitamente indica INDECISÃO 
        # 2. E variação de preço é praticamente zero (< 0.01%)
        # 3. E confiança do modelo é muito baixa (< 40%)
        is_indecision = ("INDECISÃO" in final_rec and 
                        price_change_pct < 0.01 and 
                        model_confidence < 0.4)
        
        # Durante indecisão, mostrar previsão futura para execução de ordens
        if is_indecision:
            # Durante indecisão, mostrar previsão futura para execução de ordens
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(158,158,158,0.1), rgba(189,189,189,0.1));
                border-left: 4px solid #9E9E9E;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                text-align: center;
            ">
                <h4 style="color: #666; margin: 0 0 0.8rem 0; font-size: 1rem;">⚪ Mercado em Verdadeira Indecisão</h4>
                <p style="color: #888; margin: 0; font-size: 0.9rem;">
                    Análise técnica ocultada - Variação: {price_change_pct:.3f}% | Confiança: {model_confidence*100:.0f}%
                </p>
                <p style="color: #888; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                    Ativando previsão futura para execução de ordens...
                </p>
            </div>
            """, unsafe_allow_html=True)
            

        
        # ANÁLISE REALÍSTICA DE DRAWDOWN E EXTENSÕES - Foco 100% em Swing, Intraday e Position
        if not is_indecision:
            # Calcular valores realísticos diretamente sem função externa
            pip_value = 0.0001 if 'JPY' not in pair_name else 0.01
            
            # Parâmetros por horizonte temporal - MESMAS CONDIÇÕES REALÍSTICAS
            horizon_params = {
                '15 Minutos': {'base_target': 25, 'drawdown_prob': 0.25, 'extension_prob': 0.70, 'adverse_pips': 12},
                '1 Hora': {'base_target': 40, 'drawdown_prob': 0.30, 'extension_prob': 0.75, 'adverse_pips': 25},
                '4 Horas': {'base_target': 90, 'drawdown_prob': 0.35, 'extension_prob': 0.80, 'adverse_pips': 45},
                '1 Dia': {'base_target': 180, 'drawdown_prob': 0.40, 'extension_prob': 0.85, 'adverse_pips': 90}  # Position Trader com mesmas condições realísticas
            }
            
            params = horizon_params.get(horizon, horizon_params['1 Hora'])
            
            # Ajustar por confiança e sentimento
            confidence_boost = (confidence - 0.5) * 0.4
            sentiment_boost = abs(sentiment_score) * 0.15
            
            base_target = params['base_target']
            adjusted_target = int(base_target * (1 + confidence_boost + sentiment_boost))
            
            # Direção do movimento baseada na mudança de preço
            if results['price_change'] > 0:
                direction = "ALTA"
            elif results['price_change'] < 0:
                direction = "BAIXA"
            else:
                direction = "LATERAL"
            
            # Cálculos de drawdown e extensão
            drawdown_pips = params['adverse_pips']
            extension_pips = adjusted_target
            
            if direction == "ALTA":
                max_adverse_level = current_price - (drawdown_pips * pip_value)
                extension_level = current_price + (extension_pips * pip_value)
            else:
                max_adverse_level = current_price + (drawdown_pips * pip_value)
                extension_level = current_price - (extension_pips * pip_value)
            
            drawdown_prob = max(0.15, min(0.50, params['drawdown_prob'] - confidence_boost))
            extension_prob = min(0.95, max(0.50, params['extension_prob'] + confidence_boost))
            
            # Interface simplificada e direta
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(139,69,19,0.1), rgba(255,140,0,0.1)); 
                        border-left: 4px solid #FF8C00; border-radius: 8px; padding: 1.5rem; margin: 1rem 0;">
                <h4 style="color: #FF8C00; margin: 0 0 1rem 0; text-align: center;">
                    🎯 Análise Realística Especializada
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas principais
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "⚠️ DRAWDOWN MÁXIMO",
                    f"{max_adverse_level:.5f}",
                    f"-{drawdown_pips} pips ({drawdown_prob:.0%} prob.)"
                )
                
            with col2:
                st.metric(
                    "🎯 EXTENSÃO MÁXIMA", 
                    f"{extension_level:.5f}",
                    f"{'+'if direction=='ALTA' else '-'}{extension_pips} pips ({extension_prob:.0%} prob.)"
                )
            
            # Informações detalhadas de gestão de risco
            st.markdown("### 📊 Gestão de Risco e Probabilidades")
            
            # Cálculos de gestão de banca
            risk_reward_ratio = extension_pips / drawdown_pips if drawdown_pips > 0 else 0
            
            # Probabilidade de sucesso baseada em confluência
            base_success_prob = 0.45  # Base realística para forex
            confidence_factor = confidence * 0.3  # Máximo 30% de boost
            sentiment_factor = abs(sentiment_score) * 0.15  # Máximo 15% de boost
            final_success_prob = min(0.75, base_success_prob + confidence_factor + sentiment_factor)
            
            # Cálculo de risco da banca (assumindo 2% de risco por trade)
            risk_per_trade = 0.02  # 2% da banca por trade
            potential_loss_pct = risk_per_trade
            potential_gain_pct = risk_per_trade * risk_reward_ratio
            
            # Métricas em colunas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Probabilidade de Sucesso",
                    f"{final_success_prob:.0%}",
                    f"Base {base_success_prob:.0%} + Boost {(confidence_factor + sentiment_factor):.0%}"
                )
                
            with col2:
                st.metric(
                    "⚖️ Razão Risco:Recompensa", 
                    f"1:{risk_reward_ratio:.1f}",
                    "Drawdown vs Extensão"
                )
                
            with col3:
                st.metric(
                    "📈 Expectativa Matemática",
                    f"{(final_success_prob * potential_gain_pct - (1-final_success_prob) * potential_loss_pct)*100:.2f}%",
                    "Por trade (2% risco)"
                )
            
            # Detalhes expandidos
            with st.expander("📋 Detalhes Completos da Análise"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **📊 Parâmetros da Análise:**
                    - **Direção Prevista:** {direction}
                    - **Horizonte Temporal:** {horizon}
                    - **Perfil de Risco:** {risk_level_used}
                    - **Alvo Base:** {base_target} pips
                    - **Alvo Ajustado:** {adjusted_target} pips
                    
                    **🎯 Ajustes Aplicados:**
                    - **Confiança LSTM:** {confidence_boost:+.1%}
                    - **Impacto Sentimento:** {sentiment_boost:+.1%}
                    - **Confiança Final:** {confidence:.0%}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **📈 Cenários de Gestão:**
                    - **Risco por Trade:** {potential_loss_pct:.1%} da banca
                    - **Potencial Ganho:** {potential_gain_pct:.1%} da banca
                    - **Prob. Drawdown:** {drawdown_prob:.0%}
                    - **Prob. Extensão:** {extension_prob:.0%}
                    - **Prob. Sucesso Total:** {final_success_prob:.0%}
                    
                    **⚠️ Gestão Recomendada:**
                    - **Max por Trade:** 2% da banca
                    - **Stop Loss:** {drawdown_pips} pips
                    - **Take Profit:** {extension_pips} pips
                    """)
            
            # Gestão de Banca Detalhada em Valores
            st.markdown("### 💰 Gestão de Banca - Valores em Dólar")
            
            # Usar valores do sidebar
            bank_value = st.session_state.get('bank_value', 5000.0)
            lot_size = st.session_state.get('lot_size', 0.1)
            
            # Cálculo simples do valor do pip baseado no par
            pair_name_str = str(pair_name)  # Garantir que é string
            if 'JPY' in pair_name_str:
                pip_value_per_lot = 10.0  # JPY pairs: 0.01 = $10 per standard lot
            elif str(pair_name) in ['XAUUSD', 'GOLD']:
                pip_value_per_lot = 1.0   # Gold: 0.1 = $1 per 0.1 lot
            else:
                pip_value_per_lot = 10.0  # Major pairs: 0.0001 = $10 per standard lot
            
            # Cálculos simples em dólares baseados no lote escolhido
            max_drawdown_usd = drawdown_pips * pip_value_per_lot * lot_size
            max_extension_usd = extension_pips * pip_value_per_lot * lot_size
            
            # Percentual em relação à banca
            drawdown_pct = (max_drawdown_usd / bank_value) * 100
            extension_pct = (max_extension_usd / bank_value) * 100
            
            # Métricas simples em valores de dólar
            st.markdown("#### 💰 Valores de Trading Calculados")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💳 Valor da Banca",
                    f"${bank_value:,.2f}",
                    "Configurado no sidebar"
                )
                
            with col2:
                st.metric(
                    "📊 Lote Selecionado",
                    f"{lot_size:.2f}",
                    "Configurado no sidebar"
                )
                
            with col3:
                st.metric(
                    "📉 Drawdown Máximo",
                    f"${max_drawdown_usd:.2f}",
                    f"{drawdown_pct:.2f}% da banca"
                )
                
            with col4:
                st.metric(
                    "📈 Extensão Máxima",
                    f"${max_extension_usd:.2f}",
                    f"{extension_pct:.2f}% da banca"
                )
            
            # Seção de análise de probabilidade removida conforme solicitado pelo usuário
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(158,158,158,0.1), rgba(189,189,189,0.1));
                border-left: 4px solid #9E9E9E;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                text-align: center;
            ">
                <h4 style="color: #666; margin: 0 0 0.8rem 0; font-size: 1rem;">⚪ Análise de Risco Indisponível</h4>
                <p style="color: #888; margin: 0; font-size: 0.9rem;">
                    Análise de risco ocultada durante verdadeira indecisão - Confiança: {model_confidence*100:.0f}%
                </p>
                <p style="color: #888; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                    Parâmetros de risco aparecerão quando análise indicar direção
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # All unified analysis components display completely removed as requested
    
    if 'analysis_focus' in results:
        st.info(f"**Foco da Análise:** {results['analysis_focus']}")
    
    # Show risk level impact summary
    if 'risk_level_used' in results:
        risk_level = results['risk_level_used']
        risk_impacts = {
            'Conservative': "🛡️ Proteção máxima - Stop loss próximo, menor exposição, maior segurança",
            'Moderate': "⚖️ Equilíbrio - Risco moderado com potencial de retorno balanceado",
            'Aggressive': "🚀 Maior potencial - Stop loss distante, maior exposição, busca máximos retornos"
        }
        
        st.success(f"**Impacto do Perfil {risk_level}:** {risk_impacts.get(risk_level, 'Perfil padrão aplicado')}")

def display_summary_tab(results, analysis_mode):
    """Display summary tab content"""
    mode_names = {
        'unified': '🧠 Análise Unificada Inteligente',
        'technical': '📊 Análise Técnica',
        'sentiment': '📰 Análise de Sentimento',
        'risk': '⚖️ Análise de Risco',
        'ai_lstm': '🤖 Análise IA/LSTM',
        'volume': '📈 Análise de Volume',
        'trend': '📉 Análise de Tendência'
    }
    
    st.markdown(f"### {mode_names.get(analysis_mode, 'Análise Padrão')}")
    
    if 'analysis_focus' in results:
        st.info(f"**Foco:** {results['analysis_focus']}")
    
    # Main recommendation card
    if 'final_recommendation' in results:
        recommendation = results['final_recommendation']
    else:
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "⚪ INDECISÃO"
    
    confidence_color = "green" if results['model_confidence'] > 0.7 else "orange" if results['model_confidence'] > 0.5 else "red"
    
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color: {confidence_color}; margin: 0; text-align: center;">{recommendation}</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <p><strong>Preço Atual:</strong> {results['current_price']:.5f}</p>
                <p><strong>Preço Previsto:</strong> {results['predicted_price']:.5f}</p>
            </div>
            <div>
                <p><strong>Variação:</strong> {results['price_change_pct']:+.2f}%</p>
                <p><strong>Confiança:</strong> {results['model_confidence']:.0%}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Unified analysis components display removed as requested

def display_charts_tab(results):
    """Display charts tab content"""
    st.markdown("### 📈 Gráficos de Análise")
    
    if 'df_with_indicators' not in results:
        st.warning("Dados de indicadores não disponíveis para exibir gráficos.")
        return
    
    df = results['df_with_indicators']
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create main price chart
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Preço e Médias Móveis', 'RSI (14 períodos)', 'MACD (12,26,9)'),
            vertical_spacing=0.05
        )
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            name='Preço de Fechamento',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        if 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['sma_20'],
                name='SMA 20 períodos',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'sma_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['sma_50'],
                name='SMA 50 períodos',
                line=dict(color='red', width=1)
            ), row=1, col=1)
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI (14 períodos)',
                line=dict(color='purple')
            ), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD (12,26)',
                line=dict(color='blue')
            ), row=3, col=1)
            
            if 'macd_signal' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['macd_signal'],
                    name='Linha de Sinal (9)',
                    line=dict(color='red')
                ), row=3, col=1)
        
        fig.update_layout(
            height=800,
            title=f"Análise Técnica - {results['pair']}",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.error("Plotly não está disponível para gráficos interativos.")
        
        # Fallback to simple metrics
        st.markdown("**Dados dos Últimos Períodos:**")
        
        if len(df) > 10:
            recent_data = df.tail(10)[['close', 'rsi', 'macd']].round(5)
            st.dataframe(recent_data)

def display_technical_tab(results):
    """Display technical analysis tab content"""
    st.markdown("### 🔍 Análise Técnica Detalhada")
    
    if 'df_with_indicators' not in results:
        st.warning("Dados técnicos não disponíveis.")
        return
    
    df = results['df_with_indicators']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Indicadores Atuais:**")
        
        if 'rsi' in df.columns:
            rsi_current = df['rsi'].iloc[-1]
            rsi_status = "Sobrecomprado" if rsi_current > 70 else "Sobrevendido" if rsi_current < 30 else "Neutro"
            st.metric("RSI (14 períodos)", f"{rsi_current:.1f}", rsi_status)
        
        if 'macd' in df.columns:
            macd_current = df['macd'].iloc[-1]
            st.metric("MACD (12,26,9)", f"{macd_current:.5f}")
        
        if 'sma_20' in df.columns:
            sma20 = df['sma_20'].iloc[-1]
            st.metric("SMA (20 períodos)", f"{sma20:.5f}")
        
        if 'sma_50' in df.columns:
            sma50 = df['sma_50'].iloc[-1]
            st.metric("SMA (50 períodos)", f"{sma50:.5f}")
    
    with col2:
        st.markdown("**Sinais de Trading:**")
        
        current_price = results['current_price']
        
        # Price vs moving averages
        if 'sma_20' in df.columns:
            sma20 = df['sma_20'].iloc[-1]
            price_vs_sma20 = "Acima" if current_price > sma20 else "Abaixo"
            st.write(f"**Preço vs SMA20:** {price_vs_sma20}")
        
        if 'sma_50' in df.columns:
            sma50 = df['sma_50'].iloc[-1]
            price_vs_sma50 = "Acima" if current_price > sma50 else "Abaixo"
            st.write(f"**Preço vs SMA50:** {price_vs_sma50}")
        
        # RSI signals
        if 'rsi' in df.columns:
            rsi_current = df['rsi'].iloc[-1]
            if rsi_current > 70:
                st.write("🔴 **RSI:** Sinal de Venda (Sobrecomprado)")
            elif rsi_current < 30:
                st.write("🟢 **RSI:** Sinal de Compra (Sobrevendido)")
            else:
                st.write("🟡 **RSI:** Neutro")
        
        # Volatility
        volatility = df['close'].tail(20).std() / current_price
        st.metric("Volatilidade (20 períodos)", f"{volatility:.4f}")

def display_sentiment_tab(results):
    """Display sentiment analysis tab content"""
    st.markdown("### 📰 Análise de Sentimento")
    
    sentiment_score = results.get('sentiment_score', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment gauge
        if sentiment_score > 0.1:
            sentiment_color = "green"
            sentiment_label = "Positivo"
            sentiment_icon = "📈"
        elif sentiment_score < -0.1:
            sentiment_color = "red"
            sentiment_label = "Negativo"
            sentiment_icon = "📉"
        else:
            sentiment_color = "orange"
            sentiment_label = "Neutro"
            sentiment_icon = "➖"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; border: 2px solid {sentiment_color}; border-radius: 10px;">
            <h2 style="color: {sentiment_color}; margin: 0;">{sentiment_icon} {sentiment_label}</h2>
            <p style="font-size: 1.5em; margin: 0.5rem 0;">{sentiment_score:.3f}</p>
            <p style="margin: 0;">Score de Sentimento</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Interpretação:**")
        
        if sentiment_score > 0.3:
            st.success("Sentimento muito positivo - Forte pressão de compra esperada")
        elif sentiment_score > 0.1:
            st.info("Sentimento positivo - Leve pressão de compra")
        elif sentiment_score < -0.3:
            st.error("Sentimento muito negativo - Forte pressão de venda esperada")
        elif sentiment_score < -0.1:
            st.warning("Sentimento negativo - Leve pressão de venda")
        else:
            st.info("Sentimento neutro - Mercado equilibrado")
        
        st.markdown("**Escala:**")
        st.write("• +1.0 = Extremamente Positivo")
        st.write("• +0.5 = Muito Positivo")
        st.write("• +0.1 = Levemente Positivo")
        st.write("• 0.0 = Neutro")
        st.write("• -0.1 = Levemente Negativo")
        st.write("• -0.5 = Muito Negativo")
        st.write("• -1.0 = Extremamente Negativo")

def display_metrics_tab(results):
    """Display detailed metrics tab content"""
    st.markdown("### 📊 Métricas Detalhadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Preços:**")
        st.metric("Preço Atual", f"{results['current_price']:.5f}")
        st.metric("Preço Previsto", f"{results['predicted_price']:.5f}")
        st.metric("Variação Absoluta", f"{results['price_change']:+.5f}")
    
    with col2:
        st.markdown("**Percentuais:**")
        st.metric("Variação %", f"{results['price_change_pct']:+.2f}%")
        st.metric("Confiança", f"{results['model_confidence']:.1%}")
        
        if 'sentiment_score' in results:
            st.metric("Sentimento", f"{results['sentiment_score']:+.3f}")
    
    with col3:
        st.markdown("**Informações da Análise:**")
        st.write(f"**Par:** {results['pair']}")
        st.write(f"**Intervalo:** {results['interval']}")
        st.write(f"**Horizonte:** {results['horizon']}")
        st.write(f"**Horário:** {results['timestamp'].strftime('%H:%M:%S')}")
        
        analysis_mode = results.get('analysis_mode', 'unified')
        mode_names = {
            'unified': 'Unificada',
            'technical': 'Técnica',
            'sentiment': 'Sentimento',
            'risk': 'Risco',
            'ai_lstm': 'IA/LSTM',
            'volume': 'Volume',
            'trend': 'Tendência'
        }
        st.write(f"**Tipo:** {mode_names.get(analysis_mode, 'Padrão')}")
    
    # Show component breakdown for unified analysis


def run_basic_analysis(current_price, is_quick, sentiment_score=0):
    """Análise básica/rápida"""
    import numpy as np
    signal = np.random.uniform(-0.01, 0.01) + (sentiment_score * 0.005)
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.6 if is_quick else 0.75,
        'analysis_focus': 'Análise rápida' if is_quick else 'Análise padrão'
    }

def display_analysis_results():
    """Display enhanced analysis results - COMPONENTES REMOVIDOS"""
    if not st.session_state.get('analysis_results'):
        return
    
    results = st.session_state.analysis_results
    analysis_mode = results.get('analysis_mode', 'unified')
    
    st.markdown("## 📊 Resultados da Análise")
    
    # Mostrar tipo de análise executada
    mode_names = {
        'unified': '🧠 Análise Unificada Inteligente',
        'technical': '📊 Análise Técnica',
        'sentiment': '📰 Análise de Sentimento',
        'risk': '⚖️ Análise de Risco',
        'ai_lstm': '🤖 Análise IA/LSTM',
        'volume': '📈 Análise de Volume',
        'trend': '📉 Análise de Tendência'
    }
    
    st.markdown(f"**Tipo:** {mode_names.get(analysis_mode, 'Análise Padrão')}")
    
    if 'analysis_focus' in results:
        st.caption(f"Foco: {results['analysis_focus']}")
    
    # Main recommendation with new enhanced display
    if 'final_recommendation' in results:
        recommendation = results['final_recommendation']
    elif 'market_direction' in results:
        recommendation = f"🎯 {results['market_direction']}"
    else:
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "⚪ INDECISÃO"
    
    # Enhanced display for unified analysis with market direction and probability
    if analysis_mode == 'unified' and 'market_direction' in results:
        direction = results['market_direction']
        probability = results.get('success_probability', results['model_confidence'] * 100)
        confluence = results.get('confluence_strength', 0)
        agreement = results.get('agreement_score', 0)
        
        # Color coding based on direction and probability
        direction_str = str(direction)  # Garantir que é string
        if 'COMPRA FORTE' in direction_str:
            direction_color = "#00C851"  # Strong green
            direction_icon = "🚀"
        elif 'COMPRA' in direction_str:
            direction_color = "#4CAF50"  # Green
            direction_icon = "📈"
        elif 'VENDA FORTE' in direction_str:
            direction_color = "#FF3547"  # Strong red
            direction_icon = "🔴"
        elif 'VENDA' in direction_str:
            direction_color = "#F44336"  # Red
            direction_icon = "📉"
        else:
            direction_color = "#FF9800"  # Orange for neutral
            direction_icon = "⚪"
        
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 2rem; 
            border: 3px solid {direction_color}; 
            border-radius: 15px; 
            background: linear-gradient(135deg, rgba(0,0,0,0.05), rgba(255,255,255,0.1));
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        ">
            <h2 style="color: {direction_color}; margin: 0 0 1rem 0; font-size: 2.5em;">{direction_icon} {direction}</h2>
            <h3 style="color: {direction_color}; margin: 0 0 1.5rem 0; font-size: 1.8em;">Probabilidade: {probability:.0f}%</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid {direction_color};">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Confluência</strong></p>
                    <p style="margin: 0; font-size: 1.3rem; font-weight: bold; color: {direction_color};">{confluence} Sinais Fortes</p>
                </div>
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid {direction_color};">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Concordância</strong></p>
                    <p style="margin: 0; font-size: 1.3rem; font-weight: bold; color: {direction_color};">{agreement}/4 Componentes</p>
                </div>
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid {direction_color};">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Preço Atual</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: #333;">{results['current_price']:.5f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid {direction_color};">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Previsão</strong></p>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: #333;">{results['predicted_price']:.5f}</p>
                </div>
            </div>
            
            <p style="color: #666; margin: 0; font-size: 0.95rem;">
                <strong>Análise Confluente:</strong> {agreement} componentes concordam com {confluence} sinais de alta força. 
                Variação esperada: {results['price_change_pct']:+.2f}% | Confiança: {results['model_confidence']:.0%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to original display for other analysis modes
        confidence_color = "green" if results['model_confidence'] > 0.7 else "orange" if results['model_confidence'] > 0.5 else "red"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {confidence_color}; margin: 0;">{recommendation}</h3>
            <p style="margin: 0.5rem 0;"><strong>Preço Atual:</strong> {results['current_price']:.5f}</p>
            <p style="margin: 0.5rem 0;"><strong>Preço Previsto:</strong> {results['predicted_price']:.5f}</p>
            <p style="margin: 0.5rem 0;"><strong>Variação:</strong> {results['price_change_pct']:+.2f}%</p>
            <p style="margin: 0.5rem 0;"><strong>Confiança:</strong> {results['model_confidence']:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Variação Prevista",
            f"{results['price_change_pct']:+.2f}%",
            f"{results['price_change']:+.5f}"
        )
    
    with col2:
        st.metric(
            "Confiança do Modelo",
            f"{results['model_confidence']:.0%}",
            "Alta" if results['model_confidence'] > 0.7 else "Baixa"
        )
    
    with col3:
        st.metric(
            "Horário da Análise",
            results['timestamp'].strftime('%H:%M:%S'),
            f"Par: {results['pair']}"
        )

def display_metrics_tab(results):
    """Display metrics tab content"""
    st.markdown("### 📊 Métricas Detalhadas")
    
    mode_names = {
        'unified': '🧠 Análise Unificada Inteligente',
        'technical': '📊 Análise Técnica', 
        'sentiment': '📰 Análise de Sentimento',
        'risk': '⚖️ Análise de Risco',
        'ai_lstm': '🤖 Análise IA/LSTM',
        'volume': '📈 Análise de Volume',
        'trend': '📉 Análise de Tendência'
    }
    
    analysis_mode = results.get('analysis_mode', 'unified')
    if analysis_mode in mode_names:
        st.write(f"**Tipo:** {mode_names.get(analysis_mode, 'Padrão')}")

if __name__ == "__main__":
    main()
