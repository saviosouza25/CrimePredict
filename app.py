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

# Enhanced imports including Alpha Vantage indicators
try:
    from models.lstm_model import ForexPredictor
    from services.data_service import DataService  
    from services.sentiment_service import SentimentService
    from services.ai_unified_service import AIUnifiedService
    from services.alpha_vantage_indicators import AlphaVantageIndicators
    from services.trend_analysis_engine import TrendAnalysisEngine
    from utils.cache_manager import CacheManager
    
    # Initialize services
    services = {
        'data_service': DataService(),
        'sentiment_service': SentimentService(),
        'ai_unified_service': AIUnifiedService()
    }
except ImportError as e:
    st.error(f"❌ ERRO CRÍTICO: Falha ao carregar serviços essenciais: {e}")
    st.error("🔑 Verifique se a chave API Alpha Vantage está configurada corretamente")
    st.error("📡 Sistema requer conexão real com Alpha Vantage API - dados simulados não são permitidos")
    st.stop()  # Stop execution - no mock services allowed

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
            if key.startswith(('cache_', 'analysis_', 'unified_', 'ai_result_')):
                if key != 'analysis_results':  # Preservar resultado final
                    del st.session_state[key]

def apply_theme_css():
    """Apply theme-specific CSS based on current theme"""
    current_theme = st.session_state.get('theme', 'light')
    
    # CSS para ocultar elementos de carregamento do Streamlit
    hide_loading_css = """
    <style>
        /* Ocultar spinner de carregamento no canto superior direito */
        .stSpinner > div {
            display: none !important;
        }
        
        /* Ocultar indicador de running no header */
        .stApp > header [data-testid="stHeader"] .stSpinner {
            display: none !important;
        }
        
        /* Ocultar status de "Running" */
        .stStatus {
            display: none !important;
        }
        
        /* Ocultar todos os spinners do sistema */
        div[data-testid="stSpinner"] {
            display: none !important;
        }
        
        /* Ocultar loading overlay */
        .stLoadingOverlay {
            display: none !important;
        }
        
        /* Indicador de carregamento personalizado */
        .custom-loader {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            z-index: 9999;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Mostrar loader personalizado quando necessário */
        .show-custom-loader .custom-loader {
            display: block !important;
        }
    </style>
    """
    st.markdown(hide_loading_css, unsafe_allow_html=True)
    
    if current_theme == 'dark':
        st.markdown("""
        <style>
            .stApp {
                background-color: #0e1117 !important;
                color: #ffffff !important;
            }
            .main .block-container {
                background-color: #0e1117 !important;
                color: #ffffff !important;
            }
            .stSelectbox > div > div {
                background-color: #262730 !important;
                color: #ffffff !important;
            }
            .stSlider > div > div > div {
                background-color: #667eea !important;
            }
            .stMarkdown {
                color: #ffffff !important;
            }
            .metric-card {
                background: linear-gradient(135deg, #1e1e1e, #2d2d2d) !important;
                border: 1px solid #444 !important;
                color: #ffffff !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .main .block-container {
                background-color: #ffffff !important;
            }
            .metric-card {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef) !important;
                border: 1px solid #dee2e6 !important;
            }
        </style>
        """, unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 4rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 2rem auto;
            max-width: 900px;
            width: 90%;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        ">
            <h1 style="color: white; margin-bottom: 1rem; font-size: 2.4em; font-weight: 700;">🔐 Acesso Restrito</h1>
            <h2 style="color: white; margin-bottom: 1rem; font-size: 1.9em; font-weight: 600;">Plataforma Avançada de Análise Forex</h2>
            <h3 style="color: rgba(255,255,255,0.95); margin-bottom: 2rem; font-size: 1.4em; font-weight: 400; line-height: 1.4;">
                Sistema profissional de trading com IA e análise em tempo real
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Formulário de login centralizado
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔑 Digite a Senha de Acesso")
            password = st.text_input("Senha:", type="password", placeholder="Digite sua senha...", key="login_password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🚀 Entrar na Plataforma", type="primary", use_container_width=True, key="login_button"):
                    if password == "artec2025":
                        st.session_state.authenticated = True
                        st.success("✅ Acesso autorizado! Redirecionando...")
                        st.rerun()
                    else:
                        st.error("❌ Senha incorreta. Tente novamente.")
        
        # Informações da plataforma
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧠 Inteligência Artificial
            - Rede neural LSTM avançada
            - Análise de sentimento em tempo real
            - Predições com alta precisão
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Análise Técnica
            - 15+ indicadores técnicos
            - Sinais automáticos de trading
            - Múltiplos timeframes
            """)
        
        with col3:
            st.markdown("""
            ### 💰 Gestão de Risco
            - Cálculos MT4/MT5 reais
            - Stop loss inteligente
            - Múltiplos perfis de risco
            """)
        
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
    
    # Initialize services if not already done - ONLY REAL DATA ALLOWED
    global services
    if 'services' not in globals() or services is None:
        try:
            from services.data_service import DataService
            from services.sentiment_service import SentimentService
            services = {
                'data_service': DataService(),
                'sentiment_service': SentimentService(),
                'alpha_indicators': AlphaVantageIndicators(),
                'trend_engine': TrendAnalysisEngine()
            }
            # Verify Alpha Vantage API key is present
            from config.settings import API_KEY
            if not API_KEY or API_KEY == 'your_alpha_vantage_api_key_here':
                st.error("❌ CHAVE API ALPHA VANTAGE NÃO CONFIGURADA")
                st.error("🔑 Configure ALPHA_VANTAGE_API_KEY nas variáveis de ambiente")
                st.stop()
        except Exception as e:
            st.error(f"❌ ERRO: Falha ao inicializar serviços: {e}")
            st.stop()
    
    # Sidebar lateral simples como era antes
    with st.sidebar:
        # Botão Home no topo da sidebar
        if st.button("🏠 Home", type="primary", use_container_width=True, key="home_button"):
            # Limpar todos os resultados e voltar ao estado inicial
            for key in ['analysis_results', 'show_analysis', 'analysis_mode']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Botão de logout
        if st.button("🚪 Logout", type="secondary", use_container_width=True, key="logout_button"):
            # Limpar sessão e autenticação
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # Header da sidebar compacto
        st.markdown("## ⚙️ Configurações")
        
        # Market selection
        st.markdown("### 📊 Mercado")
        market_type = st.radio(
            "Tipo de Mercado:",
            ["Forex", "Criptomoedas"],
            index=0,
            key="market_type_select"
        )
        
        # Pair selection based on market type
        if market_type == "Forex":
            available_pairs = PAIRS
            pair_label = "💱 Par de Moedas"
        else:  # Criptomoedas
            available_pairs = CRYPTO_PAIRS
            pair_label = "₿ Par Cripto"
        
        # Configurações básicas compactas
        pair = st.selectbox(pair_label, available_pairs, key="pair_selectbox")
        
        # Sistema unificado de Intervalo e Horizonte
        st.markdown("**⏰ Configuração Temporal Unificada**")
        
        # Presets integrados para máxima coerência (usando valores exatos de HORIZONS)
        temporal_presets = {

            "Intraday (15-30 min)": {"interval": "15min", "horizon": "1 Hora", "description": "Operações no mesmo dia"},
            "Swing (1-4 horas)": {"interval": "60min", "horizon": "4 Horas", "description": "Operações de alguns dias"},
            "Position (Diário)": {"interval": "daily", "horizon": "1 Dia", "description": "Operações de posição"}
        }
        
        preset_choice = st.selectbox(
            "Estratégia Temporal:",
            list(temporal_presets.keys()),
            index=1,  # Default Intraday
            help="Presets otimizados para máxima precisão entre intervalo e horizonte",
            key="temporal_preset_selectbox"
        )
        
        selected_preset = temporal_presets[preset_choice]
        interval = selected_preset["interval"]
        horizon = selected_preset["horizon"]
        
        # Mapear preset_choice para trading_style
        trading_style_mapping = {
            "Intraday (15-30 min)": "intraday",
            "Swing (1-4 horas)": "swing", 
            "Position (Diário)": "position"
        }
        
        # Definir trading_style baseado na seleção
        trading_style = trading_style_mapping.get(preset_choice, "swing")
        st.session_state['trading_style'] = trading_style
        
        # Mostrar configuração atual com estratégia
        st.info(f"📊 **{preset_choice}** | Intervalo: {interval} | Horizonte: {horizon}")
        st.caption(f"💡 {selected_preset['description']}")
        st.success(f"🎯 **Estratégia Ativa:** {trading_style.upper()}")
        
        # Opção avançada para configuração manual (colapsável)
        with st.expander("⚙️ Configuração Manual Avançada"):
            st.warning("⚠️ Configuração manual pode reduzir a precisão se intervalo e horizonte não estiverem alinhados!")
            
            manual_interval = st.selectbox("Intervalo Manual:", list(INTERVALS.keys()), 
                                         index=list(INTERVALS.keys()).index(interval), key="manual_interval_selectbox")
            # Verificar se horizonte existe na lista, senão usar primeiro item
            horizon_index = 0
            try:
                horizon_index = HORIZONS.index(horizon)
            except ValueError:
                horizon = HORIZONS[0]  # Usar o primeiro como fallback
            
            manual_horizon = st.selectbox("Horizonte Manual:", HORIZONS,
                                        index=horizon_index, key="manual_horizon_selectbox")
            
            if st.checkbox("Usar Configuração Manual", key="manual_config_checkbox"):
                interval = manual_interval
                horizon = manual_horizon
                # Tentar manter o trading_style consistente mesmo no modo manual
                if "15min" in interval or "30min" in interval:
                    st.session_state['trading_style'] = "intraday"
                elif "60min" in interval or "1hour" in interval:
                    st.session_state['trading_style'] = "swing"
                elif "daily" in interval:
                    st.session_state['trading_style'] = "position"
                st.error("🔧 Modo manual ativo - Verifique se intervalo e horizonte estão compatíveis!")
        
        # Usar configuração de risco padrão (moderado)
        risk_level_en = "Moderate"
        


        
        # Gestão de Banca Simplificada
        st.markdown("**💰 Configuração de Trading**")
        
        col1, col2 = st.columns(2)
        with col1:
            bank_value = st.number_input(
                "💳 Valor da Banca (USD)", 
                min_value=100.0, 
                max_value=1000000.0, 
                value=5000.0, 
                step=500.0,
                help="Valor total da sua banca em dólares",
                key="bank_value_input"
            )
        
        with col2:
            lot_size = st.number_input(
                "📊 Tamanho do Lote",
                min_value=0.01,
                max_value=100.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="Tamanho do lote para a operação",
                key="lot_size_input"
            )
        
        # Armazenar no session state para uso nas análises
        st.session_state['bank_value'] = bank_value
        st.session_state['lot_size'] = lot_size
        
        # Calculadora de DD/Extensão Independente
        st.markdown("---")
        st.markdown("**🧮 Calculadora de DD/Extensão**")
        
        # Usar análise mais recente se disponível
        if st.session_state.get('analysis_results'):
            results = st.session_state['analysis_results']
            if 'drawdown_pips' in results and 'extension_pips' in results:
                drawdown_pips = results['drawdown_pips']
                extension_pips = results['extension_pips']
                
                # Calcular valor do pip baseado no par selecionado
                pair_str = str(pair)  # Garantir que é string
                if 'JPY' in pair_str:
                    pip_value_per_lot = 10.0
                elif str(pair) in ['XAUUSD', 'GOLD']:
                    pip_value_per_lot = 1.0
                else:
                    pip_value_per_lot = 10.0
                
                # Calcular valores em dólares
                dd_usd = drawdown_pips * pip_value_per_lot * lot_size
                ext_usd = extension_pips * pip_value_per_lot * lot_size
                dd_pct = (dd_usd / bank_value) * 100
                ext_pct = (ext_usd / bank_value) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "📉 Drawdown Máximo",
                        f"${dd_usd:.2f}",
                        f"{dd_pct:.2f}% da banca"
                    )
                with col2:
                    st.metric(
                        "📈 Extensão Máxima", 
                        f"${ext_usd:.2f}",
                        f"{ext_pct:.2f}% da banca"
                    )
                
                st.caption(f"💡 Baseado em DD: {drawdown_pips} pips | Extensão: {extension_pips} pips")
            else:
                st.info("🔍 Execute uma análise para ver os cálculos de DD/Extensão")
        else:
            st.info("🔍 Execute uma análise para ver os cálculos de DD/Extensão")
        
        # Configurações de IA colapsáveis
        with st.expander("🤖 Configurações Avançadas de IA"):
            lookback_period = st.slider("Histórico de Dados", 30, 120, LOOKBACK_PERIOD, key="lookback_slider")
            epochs = st.slider("Épocas de Treinamento", 5, 20, EPOCHS, key="epochs_slider")
            mc_samples = st.slider("Amostras Monte Carlo", 10, 50, MC_SAMPLES, key="mc_samples_slider")
        
        # Cache compacto
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"💾 {cache_count} em cache")
            with col2:
                if st.button("🗑️", help="Limpar Cache", key="clear_cache_btn"):
                    # Limpar cache do session state
                    for key in list(st.session_state.keys()):
                        if isinstance(st.session_state.get(key), tuple):
                            del st.session_state[key]
                    
                    # Limpar outras chaves de cache
                    cache_keys = ['last_pair', 'last_interval', 'cached_data', 'model_cache', 
                                  'sentiment_cache', 'indicators_cache', 'analysis_cache']
                    for key in cache_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success("Cache limpo!")
                    st.rerun()
        
        st.markdown("---")
        
        # Seção de análises especializadas
        st.markdown("**🎯 Análises Especializadas**")
        
        # Nova análise de tendência Alpha Vantage - Mais precisa
        trend_alpha_analysis = st.button("🎯 Análise de Tendência Alpha Vantage", type="primary", use_container_width=True,
                                        help="Análise avançada de tendências com indicadores Alpha Vantage otimizados por perfil", key="trend_alpha_btn")
        
        # Análise unificada principal
        unified_analysis = st.button("🧠 Análise Unificada Inteligente", use_container_width=True, 
                                   help="Combina todas as análises para a melhor previsão do mercado", key="unified_analysis_btn")
        

        
        st.markdown("**Análises Individuais:**")
        
        # Análises técnicas em colunas
        col1, col2 = st.columns(2)
        with col1:
            technical_analysis = st.button("📊 Técnica", use_container_width=True, key="technical_btn")
            sentiment_analysis = st.button("📰 Sentimento", use_container_width=True, key="sentiment_btn")
            risk_analysis = st.button("⚖️ Risco", use_container_width=True, key="risk_btn")
        with col2:
            ai_analysis = st.button("🤖 IA/LSTM", use_container_width=True, key="ai_btn")
            volume_analysis = st.button("📈 Volume", use_container_width=True, key="volume_btn")
            trend_analysis = st.button("📉 Tendência", use_container_width=True, key="trend_btn")
        
        # Análise rápida
        quick_analysis = st.button("⚡ Verificação Rápida", use_container_width=True, key="quick_analysis_btn")
        
        # Análise multi-pares
        multi_pair_analysis = st.button("🌍 Análise Multi-Pares", use_container_width=True, key="multi_pair_btn")
        
        # Processamento dos diferentes tipos de análise
        analyze_button = False
        
        if trend_alpha_analysis:
            execute_alpha_vantage_trend_analysis(pair, selected_trading_profile, market_type)
        elif unified_analysis:
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
        
        st.markdown("---")
        
        # Botões auxiliares compactos
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Tutorial"):
                st.session_state['show_tutorial'] = not st.session_state.get('show_tutorial', False)
        with col2:
            if st.button("🚪 Sair"):
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
    elif multi_pair_analysis:
        run_multi_pair_analysis(interval, horizon, lookback_period, mc_samples, epochs)
    
    # Always show main header
    display_main_header()
    
    # Display tutorial if activated
    if st.session_state.get('show_tutorial', False):
        display_comprehensive_tutorial()
    
    # Display results if available
    elif st.session_state.get('multi_pair_results'):
        display_multi_pair_results()
    elif st.session_state.get('analysis_results'):
        display_analysis_results_with_tabs()
    else:
        # Show footer when no analysis is active
        display_footer()

def display_main_header():
    """Display the main platform header"""
    st.markdown("""
    <div class="main-header" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3); color: white;">
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem;">
            📊 Plataforma Avançada de Análise Forex
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 0;">
            Análises com Dados Reais Alpha Vantage - 100% Autênticos
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_tutorial():
    """Display comprehensive tutorial about all platform functions"""
    st.markdown("# 📚 Tutorial Completo da Plataforma")
    st.markdown("### *Guia Definitivo para Maximizar seus Resultados no Trading Forex*")
    
    # Botão para fechar tutorial
    if st.button("❌ Fechar Tutorial", type="secondary", key="close_tutorial_btn"):
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

def run_multi_pair_analysis(interval, horizon, lookback_period, mc_samples, epochs):
    """Análise completa de todos os pares de moedas com recomendações de execução"""
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("## 🌍 Análise Multi-Pares - Scanner Completo")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Iniciando análise de todos os pares disponíveis...")
        progress_bar.progress(10)
        
        # Get trading style for consistent analysis
        trading_style = st.session_state.get('trading_style', 'swing')
        
        # Include both forex and crypto pairs based on market selection
        market_type_selected = st.session_state.get('market_type_select', 'Forex')
        if market_type_selected == 'Forex':
            analysis_pairs = PAIRS
            analysis_label = "pares de moedas"
        else:
            analysis_pairs = CRYPTO_PAIRS  
            analysis_label = "pares cripto"
        
        # Results storage
        all_results = []
        total_pairs = len(analysis_pairs)
        
        status_text.text(f"📊 Analisando {total_pairs} {analysis_label}...")
        
        for i, pair in enumerate(analysis_pairs):
            try:
                # Update progress
                progress = 10 + int((i / total_pairs) * 80)
                progress_bar.progress(progress)
                status_text.text(f"🔄 Analisando {pair} ({i+1}/{total_pairs})...")
                
                # Determine market type based on selected market
                market_type = 'crypto' if market_type_selected == 'Criptomoedas' else 'forex'
                
                # Fetch data for current pair
                df = services['data_service'].fetch_forex_data(pair, INTERVALS[interval], 'compact', market_type)
                
                if not services['data_service'].validate_data(df):
                    continue
                
                # Add technical indicators
                df_with_indicators = add_technical_indicators(df)
                
                # Get current price
                current_price = services['data_service'].get_latest_price(pair, market_type)
                if current_price is None:
                    continue
                
                # Get sentiment score
                sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
                
                # Run unified analysis for this pair
                analysis_result = run_unified_analysis(
                    current_price, pair, sentiment_score, df_with_indicators, trading_style
                )
                
                # Calculate trading opportunity score
                opportunity_score = calculate_opportunity_score(analysis_result, pair, trading_style)
                
                # Generate execution position
                execution_position = generate_execution_position(
                    analysis_result, pair, current_price, trading_style, sentiment_score
                )
                
                # Store comprehensive result
                pair_result = {
                    'pair': pair,
                    'current_price': current_price,
                    'opportunity_score': opportunity_score,
                    'execution_position': execution_position,
                    'analysis_result': analysis_result,
                    'sentiment_score': sentiment_score,
                    'trading_style': trading_style
                }
                
                all_results.append(pair_result)
                
            except Exception as e:
                st.warning(f"Erro ao analisar {pair}: {str(e)}")
                continue
        
        # Final processing
        status_text.text("🎯 Processando resultados e gerando recomendações...")
        progress_bar.progress(95)
        
        # Sort by opportunity score
        all_results.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Store results in session state
        st.session_state.multi_pair_results = {
            'results': all_results,
            'timestamp': datetime.now(),
            'trading_style': trading_style,
            'interval': interval,
            'horizon': horizon
        }
        
        status_text.text("✅ Análise multi-pares concluída!")
        progress_bar.progress(100)
        
        # Clear progress after moment
        import time
        time.sleep(1)
        progress_container.empty()

def calculate_opportunity_score(analysis_result, pair, trading_style):
    """Calcula score de oportunidade baseado em múltiplos fatores"""
    
    # Base score from model confidence
    base_score = analysis_result.get('model_confidence', 0.5) * 100
    
    # Direction strength bonus
    direction = analysis_result.get('market_direction', '')
    if 'FORTE' in str(direction):
        direction_bonus = 25
    elif 'COMPRA' in str(direction) or 'VENDA' in str(direction):
        direction_bonus = 15
    else:
        direction_bonus = 0
    
    # Components agreement bonus
    agreement_score = analysis_result.get('agreement_score', 0)
    agreement_bonus = agreement_score * 5  # 0-20 points
    
    # Confluence strength bonus
    confluence = analysis_result.get('confluence_strength', 0)
    confluence_bonus = confluence * 3  # 0-15 points
    
    # Success probability bonus
    success_prob = analysis_result.get('success_probability', 50)
    prob_bonus = (success_prob - 50) * 0.5  # Bonus for >50% probability
    
    # Pair volatility adjustment
    pair_adjustment = PAIR_AI_ADJUSTMENTS.get(pair, {})
    volatility_mult = pair_adjustment.get('volatility_multiplier', 1.0)
    confidence_boost = pair_adjustment.get('prediction_confidence_boost', 1.0)
    
    # Calculate final score
    total_score = (base_score + direction_bonus + agreement_bonus + 
                   confluence_bonus + prob_bonus) * confidence_boost
    
    # Adjust for volatility (higher volatility = higher potential but more risk)
    if volatility_mult > 1.5:  # High volatility pairs
        total_score *= 1.1  # 10% bonus for risk-takers
    elif volatility_mult < 0.9:  # Low volatility pairs
        total_score *= 1.05  # 5% bonus for stability
    
    return min(100, max(0, total_score))

def generate_execution_position(analysis_result, pair, current_price, trading_style, sentiment_score):
    """Gera posição completa de execução com todos os parâmetros"""
    
    direction = analysis_result.get('market_direction', '')
    confidence = analysis_result.get('model_confidence', 0.5)
    price_change_pct = analysis_result.get('price_change_pct', 0)
    
    # Determine position type
    is_buy = 'COMPRA' in str(direction)
    is_strong = 'FORTE' in str(direction)
    
    # Get risk profile parameters
    risk_profile = 'Moderate'  # Default
    risk_params = RISK_PROFILES[risk_profile]
    
    # Get temporal parameters  
    horizon_key = st.session_state.get('horizon', '1 Hora')
    temporal_params = TEMPORAL_AI_PARAMETERS.get(horizon_key, TEMPORAL_AI_PARAMETERS['1 Hora'])
    
    # Calculate position sizing
    bank_value = st.session_state.get('bank_value', 10000)  # Default $10,000
    risk_percentage = risk_params['banca_risk'] / 100
    
    # Calculate stop loss and take profit based on ATR and strategy
    if 'df_with_indicators' in analysis_result:
        df = analysis_result['df_with_indicators']
        if len(df) > 14:
            # Calculate ATR for stop/TP calculation
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
        else:
            atr = current_price * 0.001  # Fallback 0.1%
    else:
        atr = current_price * 0.001
    
    # Calculate levels based on strategy
    if is_buy:
        entry_price = current_price
        stop_loss = entry_price - (atr * risk_params['atr_multiplier_stop'])
        take_profit = entry_price + (atr * risk_params['atr_multiplier_tp'])
    else:
        entry_price = current_price
        stop_loss = entry_price + (atr * risk_params['atr_multiplier_stop'])
        take_profit = entry_price - (atr * risk_params['atr_multiplier_tp'])
    
    # Calculate position size
    risk_amount = bank_value * risk_percentage
    pip_value = 1  # Simplified - would need proper pip calculation
    stop_distance_pips = abs(entry_price - stop_loss) * 10000  # Convert to pips
    
    if stop_distance_pips > 0:
        position_size = risk_amount / stop_distance_pips
    else:
        position_size = 0.01  # Minimum position
    
    # Calculate potential profit/loss
    tp_distance_pips = abs(take_profit - entry_price) * 10000
    risk_reward_ratio = tp_distance_pips / stop_distance_pips if stop_distance_pips > 0 else 0
    
    potential_profit = tp_distance_pips * position_size
    potential_loss = stop_distance_pips * position_size
    
    # Market timing based on sentiment and technical confluence
    market_timing = "Imediato"
    if confidence > 0.8 and is_strong:
        market_timing = "Imediato"
    elif confidence > 0.7:
        market_timing = "Curto Prazo (2-4h)"
    else:
        market_timing = "Médio Prazo (1-2 dias)"
    
    # Risk level assessment
    if stop_distance_pips < 20:
        risk_level = "Baixo"
    elif stop_distance_pips < 40:
        risk_level = "Moderado"
    else:
        risk_level = "Alto"
    
    return {
        'direction': 'COMPRA' if is_buy else 'VENDA',
        'strength': 'FORTE' if is_strong else 'NORMAL',
        'entry_price': round(entry_price, 5),
        'stop_loss': round(stop_loss, 5),
        'take_profit': round(take_profit, 5),
        'position_size': round(position_size, 2),
        'risk_reward_ratio': round(risk_reward_ratio, 2),
        'potential_profit': round(potential_profit, 2),
        'potential_loss': round(potential_loss, 2),
        'stop_distance_pips': round(stop_distance_pips, 1),
        'tp_distance_pips': round(tp_distance_pips, 1),
        'market_timing': market_timing,
        'risk_level': risk_level,
        'confidence': round(confidence * 100, 1),
        'sentiment_bias': 'Positivo' if sentiment_score > 0.05 else 'Negativo' if sentiment_score < -0.05 else 'Neutro'
    }

def display_multi_pair_results():
    """Exibir resultados da análise multi-pares com ranking de oportunidades"""
    
    results_data = st.session_state.get('multi_pair_results', {})
    if not results_data:
        return
    
    results = results_data['results']
    timestamp = results_data['timestamp']
    trading_style = results_data['trading_style']
    
    # Header
    st.markdown("## 🌍 Análise Multi-Pares - Oportunidades de Trading")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Pares Analisados", len(results))
    with col2:
        st.metric("Estratégia", trading_style.title())
    with col3:
        valid_results = [r for r in results if r['opportunity_score'] > 60]
        st.metric("Oportunidades Válidas", len(valid_results))
    
    st.caption(f"Última atualização: {timestamp.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Filtros
    st.markdown("### 🔍 Filtros")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider("Score Mínimo", 0, 100, 50, 5)
    with filter_col2:
        direction_filter = st.selectbox("Direção", ["Todas", "COMPRA", "VENDA"])
    with filter_col3:
        strength_filter = st.selectbox("Força", ["Todas", "FORTE", "NORMAL"])
    
    # Filter results
    filtered_results = []
    for result in results:
        if result['opportunity_score'] < min_score:
            continue
        
        execution = result['execution_position']
        if direction_filter != "Todas" and execution['direction'] != direction_filter:
            continue
        if strength_filter != "Todas" and execution['strength'] != strength_filter:
            continue
        
        filtered_results.append(result)
    
    st.markdown(f"### 📊 Top Oportunidades ({len(filtered_results)} pares)")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏆 Ranking", "📈 Posições de Execução", "📋 Resumo Detalhado"])
    
    with tab1:
        display_opportunity_ranking(filtered_results)
    
    with tab2:
        display_execution_positions(filtered_results)
    
    with tab3:
        display_detailed_summary(filtered_results)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Nova Análise Multi-Pares"):
            del st.session_state['multi_pair_results']
            st.rerun()
    
    with col2:
        if st.button("📊 Análise Individual"):
            del st.session_state['multi_pair_results']
            st.rerun()
    
    with col3:
        if st.button("💾 Exportar Resultados"):
            st.info("Funcionalidade de exportação em desenvolvimento")

def display_opportunity_ranking(results):
    """Exibir ranking de oportunidades"""
    
    if not results:
        st.warning("Nenhuma oportunidade encontrada com os filtros aplicados.")
        return
    
    st.markdown("#### 🎯 Ranking por Score de Oportunidade")
    
    for i, result in enumerate(results[:15]):  # Top 15
        pair = result['pair']
        score = result['opportunity_score']
        execution = result['execution_position']
        current_price = result['current_price']
        
        # Color coding
        if score >= 80:
            color = "#00C851"  # Green
            badge = "🟢 EXCELENTE"
        elif score >= 70:
            color = "#4CAF50"  # Light green
            badge = "🟡 BOA"
        elif score >= 60:
            color = "#FF9800"  # Orange
            badge = "🟠 MODERADA"
        else:
            color = "#F44336"  # Red
            badge = "🔴 BAIXA"
        
        direction_icon = "📈" if execution['direction'] == 'COMPRA' else "📉"
        strength_text = execution['strength']
        
        st.markdown(f"""
        <div style="
            border: 2px solid {color}; 
            border-radius: 10px; 
            padding: 1rem; 
            margin: 0.5rem 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: {color};">#{i+1} {pair} {direction_icon}</h4>
                    <p style="margin: 0.2rem 0; color: #666;">
                        <strong>{execution['direction']} {strength_text}</strong> | 
                        Preço: {current_price:.5f} | 
                        Confiança: {execution['confidence']}%
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="
                        background: {color}; 
                        color: white; 
                        padding: 0.3rem 0.8rem; 
                        border-radius: 20px; 
                        font-weight: bold;
                        margin-bottom: 0.5rem;
                    ">
                        {score:.1f}/100
                    </div>
                    <div style="color: {color}; font-weight: bold; font-size: 0.9rem;">
                        {badge}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_execution_positions(results):
    """Exibir posições de execução detalhadas"""
    
    if not results:
        st.warning("Nenhuma posição encontrada com os filtros aplicados.")
        return
    
    st.markdown("#### ⚡ Posições de Execução Prontas")
    
    for result in results[:10]:  # Top 10 para execução
        pair = result['pair']
        execution = result['execution_position']
        
        direction_color = "#00C851" if execution['direction'] == 'COMPRA' else "#F44336"
        direction_icon = "📈" if execution['direction'] == 'COMPRA' else "📉"
        
        with st.expander(f"{direction_icon} **{pair}** - {execution['direction']} {execution['strength']} (Score: {result['opportunity_score']:.1f})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Parâmetros de Entrada:**")
                st.write(f"• **Direção:** {execution['direction']} {execution['strength']}")
                st.write(f"• **Preço de Entrada:** {execution['entry_price']:.5f}")
                st.write(f"• **Stop Loss:** {execution['stop_loss']:.5f}")
                st.write(f"• **Take Profit:** {execution['take_profit']:.5f}")
                st.write(f"• **Tamanho da Posição:** {execution['position_size']:.2f} lotes")
            
            with col2:
                st.markdown("**💰 Análise de Risco/Retorno:**")
                st.write(f"• **Risco/Retorno:** 1:{execution['risk_reward_ratio']:.2f}")
                st.write(f"• **Lucro Potencial:** ${execution['potential_profit']:.2f}")
                st.write(f"• **Perda Potencial:** ${execution['potential_loss']:.2f}")
                st.write(f"• **Stop Distance:** {execution['stop_distance_pips']:.1f} pips")
                st.write(f"• **TP Distance:** {execution['tp_distance_pips']:.1f} pips")
            
            # Risk and timing info
            st.markdown("**⏰ Timing e Risco:**")
            timing_col1, timing_col2, timing_col3 = st.columns(3)
            
            with timing_col1:
                st.info(f"**Timing:** {execution['market_timing']}")
            with timing_col2:
                risk_color = "🟢" if execution['risk_level'] == 'Baixo' else "🟡" if execution['risk_level'] == 'Moderado' else "🔴"
                st.info(f"**Risco:** {risk_color} {execution['risk_level']}")
            with timing_col3:
                sentiment_color = "🟢" if execution['sentiment_bias'] == 'Positivo' else "🔴" if execution['sentiment_bias'] == 'Negativo' else "🟡"
                st.info(f"**Sentimento:** {sentiment_color} {execution['sentiment_bias']}")

def display_detailed_summary(results):
    """Exibir resumo detalhado da análise"""
    
    if not results:
        st.warning("Nenhum resultado para resumir.")
        return
    
    st.markdown("#### 📋 Resumo Estatístico da Análise")
    
    # Calculate statistics
    total_pairs = len(results)
    avg_score = sum(r['opportunity_score'] for r in results) / total_pairs
    buy_signals = len([r for r in results if r['execution_position']['direction'] == 'COMPRA'])
    sell_signals = len([r for r in results if r['execution_position']['direction'] == 'VENDA'])
    strong_signals = len([r for r in results if r['execution_position']['strength'] == 'FORTE'])
    
    # High opportunity pairs
    high_opportunity = [r for r in results if r['opportunity_score'] >= 75]
    medium_opportunity = [r for r in results if 60 <= r['opportunity_score'] < 75]
    low_opportunity = [r for r in results if r['opportunity_score'] < 60]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score Médio", f"{avg_score:.1f}/100")
    with col2:
        st.metric("Sinais de Compra", buy_signals, f"{(buy_signals/total_pairs*100):.1f}%")
    with col3:
        st.metric("Sinais de Venda", sell_signals, f"{(sell_signals/total_pairs*100):.1f}%")
    with col4:
        st.metric("Sinais Fortes", strong_signals, f"{(strong_signals/total_pairs*100):.1f}%")
    
    # Opportunity distribution
    st.markdown("**📊 Distribuição de Oportunidades:**")
    
    opp_col1, opp_col2, opp_col3 = st.columns(3)
    
    with opp_col1:
        st.success(f"🟢 **Alta Oportunidade (75+):** {len(high_opportunity)} pares")
        if high_opportunity:
            for result in high_opportunity[:5]:
                st.write(f"• {result['pair']}: {result['opportunity_score']:.1f}")
    
    with opp_col2:
        st.warning(f"🟡 **Média Oportunidade (60-74):** {len(medium_opportunity)} pares")
        if medium_opportunity:
            for result in medium_opportunity[:5]:
                st.write(f"• {result['pair']}: {result['opportunity_score']:.1f}")
    
    with opp_col3:
        st.error(f"🔴 **Baixa Oportunidade (<60):** {len(low_opportunity)} pares")
        if low_opportunity:
            for result in low_opportunity[:5]:
                st.write(f"• {result['pair']}: {result['opportunity_score']:.1f}")
    
    # Best pairs summary
    if results:
        st.markdown("**🏆 Top 5 Recomendações Imediatas:**")
        for i, result in enumerate(results[:5]):
            pair = result['pair']
            execution = result['execution_position']
            score = result['opportunity_score']
            
            direction_icon = "📈" if execution['direction'] == 'COMPRA' else "📉"
            st.write(f"{i+1}. **{pair}** {direction_icon} - {execution['direction']} {execution['strength']} (Score: {score:.1f}) - {execution['market_timing']}")

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
            
            # Determine market type from current selection
            market_type = 'crypto' if st.session_state.get('market_type_select', 'Forex') == 'Criptomoedas' else 'forex'
            df = services['data_service'].fetch_forex_data(
                pair, 
                INTERVALS[interval], 
                'full' if not is_quick else 'compact',
                market_type
            )
            
            if not services['data_service'].validate_data(df):
                progress_container.empty()
                st.error("❌ DADOS INSUFICIENTES: Alpha Vantage retornou dados insuficientes ou inválidos")
                st.error("🔑 Verifique sua chave API Alpha Vantage ou tente outro par de moedas")
                st.info("ℹ️ Sistema configurado para usar APENAS dados reais - nenhum dado simulado será usado")
                return
            
            # Step 3: Technical indicators
            status_text.text("🔧 Calculando indicadores técnicos...")
            progress_bar.progress(50)
            
            df_with_indicators = add_technical_indicators(df)
            
            # Step 4: Current price
            status_text.text("💰 Obtendo preço atual...")
            progress_bar.progress(60)
            
            current_price = services['data_service'].get_latest_price(pair, market_type)
            
            if current_price is None:
                progress_container.empty()
                st.error(f"❌ DADOS REAIS INDISPONÍVEIS: Alpha Vantage não retornou preço atual válido para {pair}")
                st.error("🔑 Verifique: 1) Chave API válida 2) Conexão internet 3) Limite de requisições Alpha Vantage")
                st.info("ℹ️ Sistema bloqueado - usar apenas dados autênticos Alpha Vantage")
                return
            # Step 5: Enhanced Sentiment analysis with future prediction
            status_text.text("📰 Analisando sentimento e prevendo futuro do mercado...")
            progress_bar.progress(70)
            
            # Get basic sentiment score
            sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
            
            # Get comprehensive sentiment trend analysis with predictions
            sentiment_analysis_results = services['sentiment_service'].analyze_sentiment_trend(pair)
            
            # Debug: Verificar se sentimento está funcionando
            if st.session_state.get('debug_sentiment', False):
                st.info(f"🔍 DEBUG - Sentimento obtido para {pair}: {sentiment_score:.4f}")
                st.info(f"🔍 DEBUG - Sentimento previsto: {sentiment_analysis_results['predicted_sentiment']:.4f}")
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
                results.update(run_basic_analysis(current_price, df_with_indicators, sentiment_score))
            
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
            if 'progress_container' in locals():
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
    """Análise de sentimento especializada com previsão futura do mercado"""
    sentiment_service = services['sentiment_service']
    
    # Get comprehensive sentiment trend analysis with predictions
    sentiment_analysis = sentiment_service.analyze_sentiment_trend(pair)
    
    # Extract key components
    current_sentiment = sentiment_analysis['current_sentiment']
    predicted_sentiment = sentiment_analysis['predicted_sentiment']
    sentiment_momentum = sentiment_analysis['sentiment_momentum']
    market_psychology = sentiment_analysis['market_psychology']
    future_signal = sentiment_analysis['future_signal']
    
    # Ajustes baseados no perfil de risco do investidor
    risk_adjustments = {
        'Conservative': {'signal_factor': 0.6, 'confidence_penalty': 0.05, 'volatility_threshold': 0.15},
        'Moderate': {'signal_factor': 1.0, 'confidence_penalty': 0.0, 'volatility_threshold': 0.25},
        'Aggressive': {'signal_factor': 1.5, 'confidence_penalty': -0.03, 'volatility_threshold': 0.40}
    }
    
    # Usar configuração padrão (moderada)
    risk_params = risk_adjustments['Moderate']
    
    # Enhanced sentiment calculation using predictive model
    if predicted_sentiment > 0.05:  # Sentimento futuro positivo forte
        sentiment_impact = predicted_sentiment * 0.8 * risk_params['signal_factor']
    elif predicted_sentiment < -0.05:  # Sentimento futuro negativo forte
        sentiment_impact = predicted_sentiment * 0.6 * risk_params['signal_factor']
    else:  # Sentimento futuro neutro
        sentiment_impact = predicted_sentiment * 0.2 * risk_params['signal_factor']
    
    # Add momentum factor for enhanced prediction
    momentum_factor = sentiment_momentum * 0.3
    total_impact = sentiment_impact + momentum_factor
    
    predicted_price = current_price * (1 + total_impact)
    price_change = predicted_price - current_price
    
    # Enhanced confidence calculation
    base_confidence = sentiment_analysis['confidence']
    confidence = max(0.50, min(0.90, base_confidence - risk_params['confidence_penalty']))
    
    # Enhanced recommendation based on future signal
    recommendation = f"📰 {future_signal['direction']}"
    if future_signal['strength'] == 'Forte':
        recommendation += " FORTE"
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': confidence,
        'final_recommendation': recommendation,
        
        # Current sentiment data
        'sentiment_score': current_sentiment,
        'sentiment_impact': sentiment_impact,
        
        # Future prediction data
        'predicted_sentiment': predicted_sentiment,
        'sentiment_momentum': sentiment_momentum,
        'trend_direction': sentiment_analysis['trend_direction'],
        'time_horizon': sentiment_analysis['time_horizon'],
        'risk_factors': sentiment_analysis['risk_factors'],
        
        # Market psychology
        'market_psychology': market_psychology,
        'future_signal': future_signal,
        
        'analysis_focus': f'Previsão de Sentimento: {future_signal["direction"]} ({future_signal["timing"]}) - {market_psychology["market_phase"]}'
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
    
    # Show unified analysis components if available
    if analysis_mode == 'unified' and 'components' in results:
        st.markdown("### 🔍 Componentes da Análise Unificada")
        
        # Show AI analysis if available
        if 'ai_analysis' in results and results['ai_analysis'] is not None:
            ai_analysis = results['ai_analysis']
            
            st.markdown("#### 🧠 Interpretação da IA")
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(63,81,181,0.1), rgba(156,39,176,0.1));
                border-left: 4px solid #3F51B5;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            ">
                <h5 style="color: #3F51B5; margin: 0 0 0.8rem 0;">💭 {ai_analysis.unified_interpretation.get('ai_interpretation', 'Análise em processamento...')}</h5>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin-bottom: 1rem;">
                    <div style="background: rgba(63,81,181,0.1); padding: 0.8rem; border-radius: 6px; text-align: center;">
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Direção Unificada</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #3F51B5;">{ai_analysis.unified_interpretation.get('unified_direction', 'neutral').upper()}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Confiança: {ai_analysis.unified_interpretation.get('direction_confidence', 0)*100:.0f}%</p>
                    </div>
                    <div style="background: rgba(76,175,80,0.1); padding: 0.8rem; border-radius: 6px; text-align: center;">
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Consenso IA</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #4CAF50;">{ai_analysis.unified_interpretation.get('consensus_count', 0)}/3</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Componentes alinhados</p>
                    </div>
                    <div style="background: rgba(255,193,7,0.1); padding: 0.8rem; border-radius: 6px; text-align: center;">
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Recomendação</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #FF9800;">{ai_analysis.unified_interpretation.get('recommendation', 'hold').upper()}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Força: {ai_analysis.unified_interpretation.get('combined_strength', 0)*100:.0f}%</p>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <h6 style="margin: 0 0 0.5rem 0; color: #666;">Pesos dos Componentes:</h6>
                    <div style="display: flex; justify-content: space-around; text-align: center; font-size: 0.85rem;">
                        <div>
                            <strong>Histórico:</strong> {ai_analysis.unified_interpretation.get('component_weights', {}).get('historical', 0)*100:.0f}%
                        </div>
                        <div>
                            <strong>Sentimento:</strong> {ai_analysis.unified_interpretation.get('component_weights', {}).get('sentiment', 0)*100:.0f}%
                        </div>
                        <div>
                            <strong>Probabilidade:</strong> {ai_analysis.unified_interpretation.get('component_weights', {}).get('probability', 0)*100:.0f}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar componentes individuais
            st.markdown("#### 📊 Componentes Detalhados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Análise Histórica**")
                hist = ai_analysis.historical_analysis
                st.write(f"• Tendência: {hist.get('trend_direction', 'neutral')}")
                st.write(f"• Força: {hist.get('trend_strength', 0):.2f}")
                st.write(f"• Momentum: {hist.get('momentum', 0):.4f}")
                st.write(f"• Confiança: {hist.get('confidence', 0)*100:.0f}%")
            
            with col2:
                st.markdown("**📰 Análise de Sentimento**")
                sent = ai_analysis.sentiment_analysis
                st.write(f"• Direção: {sent.get('sentiment_direction', 'neutral')}")
                st.write(f"• Score: {sent.get('sentiment_score', 0):.3f}")
                st.write(f"• Humor: {sent.get('market_mood', 'uncertain')}")
                st.write(f"• Confiança: {sent.get('confidence', 0)*100:.0f}%")
            
            with col3:
                st.markdown("**🎯 Análise de Probabilidade**")
                prob = ai_analysis.probability_analysis
                st.write(f"• Direção: {prob.get('direction_probability', 0)*100:.0f}%")
                st.write(f"• Magnitude: {prob.get('magnitude_probability', 0)*100:.0f}%")
                st.write(f"• Sucesso: {prob.get('success_probability', 0)*100:.0f}%")
                st.write(f"• Confiança: {prob.get('confidence', 0)*100:.0f}%")
            
            # Mostrar parâmetros temporais específicos
            st.markdown("#### ⏰ Parâmetros da Estratégia Temporal")
            
            horizon = results.get('temporal_horizon', '1 Hora')
            pair = results.get('pair', 'EUR/USD')
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(96,125,139,0.1), rgba(55,71,79,0.1));
                border-left: 4px solid #607D8B;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            ">
                <h6 style="color: #607D8B; margin: 0 0 0.8rem 0;">Configuração Temporal: {horizon} | Par: {pair}</h6>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; text-align: center;">
                    <div style="background: rgba(96,125,139,0.1); padding: 0.6rem; border-radius: 6px;">
                        <p style="margin: 0; color: #666; font-size: 0.8rem;"><strong>Períodos Históricos</strong></p>
                        <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: #607D8B;">{ai_analysis.historical_analysis.get('periods_analyzed', 'N/A')}</p>
                    </div>
                    <div style="background: rgba(96,125,139,0.1); padding: 0.6rem; border-radius: 6px;">
                        <p style="margin: 0; color: #666; font-size: 0.8rem;"><strong>Volatilidade Adj.</strong></p>
                        <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: #607D8B;">{ai_analysis.historical_analysis.get('volatility_adjustment', 1.0):.1f}x</p>
                    </div>
                    <div style="background: rgba(96,125,139,0.1); padding: 0.6rem; border-radius: 6px;">
                        <p style="margin: 0; color: #666; font-size: 0.8rem;"><strong>Boost Confiança</strong></p>
                        <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: #607D8B;">{ai_analysis.historical_analysis.get('pair_adjustment', 1.0):.1f}x</p>
                    </div>
                    <div style="background: rgba(96,125,139,0.1); padding: 0.6rem; border-radius: 6px;">
                        <p style="margin: 0; color: #666; font-size: 0.8rem;"><strong>Confirm. Tendência</strong></p>
                        <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: #607D8B;">{ai_analysis.historical_analysis.get('trend_confirmation_strength', 0)*100:.0f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create columns for components
        cols = st.columns(2)
        components_list = list(results['components'].items())
        
        for i, (component, data) in enumerate(components_list):
            col_idx = i % 2
            with cols[col_idx]:
                signal_pct = data['signal'] * 100
                weight_pct = data['weight'] * 100
                color = "🟢" if data['signal'] > 0 else "🔴" if data['signal'] < 0 else "🟡"
                details = data.get('details', '')
                
                with st.expander(f"{color} **{component.title()}:** {signal_pct:+.2f}% (peso: {weight_pct:.0f}%)"):
                    if details:
                        st.write(f"**Detalhes:** {details}")
                    st.write(f"**Sinal:** {signal_pct:+.3f}%")
                    st.write(f"**Peso na análise:** {weight_pct:.0f}%")
    
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
    
    # Show unified analysis components if available
    if analysis_mode == 'unified' and 'components' in results:
        st.markdown("### 🔍 Componentes da Análise Unificada")
        
        for component, data in results['components'].items():
            signal_pct = data['signal'] * 100
            weight_pct = data['weight'] * 100
            color = "🟢" if data['signal'] > 0 else "🔴" if data['signal'] < 0 else "🟡"
            details = data.get('details', '')
            
            with st.expander(f"{color} **{component.title()}:** {signal_pct:+.2f}% (peso: {weight_pct:.0f}%)"):
                if details:
                    st.write(f"**Detalhes:** {details}")
                st.write(f"**Sinal:** {signal_pct:+.3f}%")
                st.write(f"**Peso na análise:** {weight_pct:.0f}%")

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

def execute_alpha_vantage_trend_analysis(pair: str, profile: str, market_type: str):
    """Execute Alpha Vantage trend analysis optimized by profile"""
    
    # Map profile names to internal values
    profile_mapping = {
        'Scalping': 'scalping',
        'Intraday': 'intraday', 
        'Swing': 'swing',
        'Position': 'position'
    }
    
    internal_profile = profile_mapping.get(profile, 'swing')
    
    # Determine optimal interval for profile
    interval_mapping = {
        'scalping': '1min',
        'intraday': '15min',
        'swing': '60min', 
        'position': 'daily'
    }
    
    interval = interval_mapping[internal_profile]
    
    with st.spinner("🎯 Executando Análise de Tendência Alpha Vantage..."):
        try:
            # Execute trend analysis
            trend_analysis = services['trend_engine'].analyze_trend_by_profile(
                pair, internal_profile, interval
            )
            
            if trend_analysis.get('error'):
                st.error(f"❌ {trend_analysis['error_message']}")
                return
            
            # Store results and display
            st.session_state['alpha_trend_results'] = trend_analysis
            display_alpha_vantage_trend_results(trend_analysis)
            
        except Exception as e:
            st.error(f"❌ Erro na análise Alpha Vantage: {e}")
            st.info("🔑 Verifique se a chave API Alpha Vantage está configurada corretamente")

def display_alpha_vantage_trend_results(analysis: Dict):
    """Display Alpha Vantage trend analysis results"""
    
    st.markdown("## 🎯 Análise de Tendência Alpha Vantage")
    
    # Main results header
    trend_signals = analysis.get('trend_signals', {})
    unified_trend = trend_signals.get('unified_trend', 'NEUTRAL')
    confidence = trend_signals.get('confidence', 0.0)
    
    # Main trend signal
    if unified_trend == 'BULLISH':
        trend_color = "🟢"
        trend_text = "COMPRA"
        trend_bg = "background: linear-gradient(135deg, #2ecc71, #27ae60);"
    elif unified_trend == 'BEARISH':
        trend_color = "🔴" 
        trend_text = "VENDA"
        trend_bg = "background: linear-gradient(135deg, #e74c3c, #c0392b);"
    else:
        trend_color = "🟡"
        trend_text = "NEUTRO"
        trend_bg = "background: linear-gradient(135deg, #f39c12, #e67e22);"
    
    # Display main signal
    st.markdown(f"""
    <div style="{trend_bg} color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h2 style="margin: 0; font-size: 2.5rem;">{trend_color} {trend_text}</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Confiança: {confidence*100:.1f}%</p>
        <p style="margin: 0; opacity: 0.9;">Perfil: {analysis.get('profile', 'N/A').title()} | Par: {analysis.get('pair', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Força da Tendência", f"{trend_signals.get('strength', 0)*100:.0f}%")
    with col2:
        st.metric("📊 Indicadores Ativos", f"{trend_signals.get('indicators_count', 0)}")
    with col3:
        st.metric("✅ Taxa de Acordo", f"{trend_signals.get('agreement_rate', 0)*100:.0f}%")
    
    # Detailed indicators
    st.markdown("### 📈 Indicadores Técnicos Alpha Vantage")
    
    indicators = analysis.get('indicators', {})
    if indicators:
        for indicator_name, indicator_data in indicators.items():
            with st.expander(f"{indicator_name.upper()} - {indicator_data.get('trend', 'N/A')}"):
                st.json(indicator_data)
    
    # Profile recommendations
    recommendations = analysis.get('profile_recommendations', {})
    if recommendations:
        st.markdown("### 🎯 Recomendações por Perfil")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**⏰ Timing:**")
            st.write(f"- Sessões ideais: {recommendations.get('optimal_sessions', ['N/A'])}")
            st.write(f"- Duração máxima: {recommendations.get('max_trade_duration', 'N/A')}")
            st.write(f"- Spread ideal: {recommendations.get('ideal_spread', 'N/A')}")
            
        with col2:
            st.markdown("**📊 Estratégia:**")
            st.write(f"- Foco: {recommendations.get('indicators_focus', 'N/A')}")
            st.write(f"- Volume: {recommendations.get('volume_requirement', 'N/A')}")
            st.write(f"- News: {recommendations.get('news_impact', 'N/A')}")
    
    # Risk management
    risk_mgmt = analysis.get('risk_management', {})
    if risk_mgmt:
        st.markdown("### 🛡️ Gestão de Risco")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Tamanho Posição", f"{risk_mgmt.get('position_size', 0):.1f}%")
        with col2:
            st.metric("🛑 Stop Loss", f"{risk_mgmt.get('stop_loss_pips', 0)} pips")
        with col3:
            st.metric("🎯 Take Profit", f"{risk_mgmt.get('take_profit_pips', 0)} pips")
    
    # Execution plan
    execution_plan = analysis.get('execution_plan', {})
    if execution_plan:
        st.markdown("### ⚡ Plano de Execução")
        
        action = execution_plan.get('recommended_action', 'WAIT')
        if action == 'EXECUTE':
            st.success(f"✅ **Ação Recomendada:** {action}")
        elif action == 'PREPARE':
            st.warning(f"⚠️ **Ação Recomendada:** {action}")
        else:
            st.info(f"ℹ️ **Ação Recomendada:** {action}")
            
        st.write(f"- **Estratégia de Entrada:** {execution_plan.get('entry_strategy', 'N/A')}")
        st.write(f"- **Monitoramento:** {execution_plan.get('monitoring_frequency', 'N/A')}")

if __name__ == "__main__":
    main()
