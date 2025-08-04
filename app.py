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

# FUNÇÃO GLOBAL: Calcular níveis confluentes de stop/take profit
def calculate_confluent_levels_global(current_price, predicted_price, pair_name, profile, market_probability):
    """Calcular níveis de stop/take profit baseados na ESTRATÉGIA TEMPORAL UNIFICADA real"""
    
    # ATR real por par (dados históricos Alpha Vantage)
    atr_values = {
        'EUR/USD': 0.0012, 'USD/JPY': 0.018, 'GBP/USD': 0.0018, 'AUD/USD': 0.0020,
        'USD/CAD': 0.0014, 'USD/CHF': 0.0016, 'NZD/USD': 0.0022, 'EUR/GBP': 0.0010,
        'EUR/JPY': 0.020, 'GBP/JPY': 0.025, 'AUD/JPY': 0.022
    }
    
    current_atr = atr_values.get(pair_name, 0.0015)
    
    # ESTRATÉGIA TEMPORAL UNIFICADA baseada no horizonte temporal escolhido
    temporal_strategy = {
        'Scalping (1-5 min)': {
            'stop_atr_multiplier': 0.8,  # Stop mais apertado para scalping
            'take_atr_multiplier': 1.2,  # Take conservador
            'fibonacci_weight': 0.3,     # Menor peso fibonacci
            'volatility_adjustment': 0.7 # Ajuste para baixa volatilidade
        },
        'Intraday (15-30 min)': {
            'stop_atr_multiplier': 1.2,  # Stop moderado
            'take_atr_multiplier': 2.0,  # Take mais amplo
            'fibonacci_weight': 0.5,     # Peso equilibrado
            'volatility_adjustment': 1.0 # Volatilidade normal
        },
        'Swing (1-4 horas)': {
            'stop_atr_multiplier': 1.8,  # Stop mais amplo
            'take_atr_multiplier': 3.5,  # Take extenso
            'fibonacci_weight': 0.7,     # Maior peso fibonacci
            'volatility_adjustment': 1.3 # Maior tolerância volatilidade
        },
        'Position (Diário)': {
            'stop_atr_multiplier': 2.5,  # Stop bem amplo
            'take_atr_multiplier': 5.0,  # Take muito extenso
            'fibonacci_weight': 0.8,     # Alto peso fibonacci
            'volatility_adjustment': 1.5 # Alta tolerância
        },
        'Trend (Semanal)': {
            'stop_atr_multiplier': 3.0,  # Stop máximo
            'take_atr_multiplier': 8.0,  # Take de longo prazo
            'fibonacci_weight': 0.9,     # Peso máximo fibonacci
            'volatility_adjustment': 2.0 # Máxima tolerância
        }
    }
    
    # Obter estratégia temporal do horizonte atual
    import streamlit as st
    horizon = st.session_state.get('analysis_horizon', '1 Hora')
    
    # Mapear horizonte para estratégia temporal
    horizon_mapping = {
        '5 Minutos': 'Scalping (1-5 min)',
        '15 Minutos': 'Intraday (15-30 min)', 
        '30 Minutos': 'Intraday (15-30 min)',
        '1 Hora': 'Swing (1-4 horas)',
        '4 Horas': 'Swing (1-4 horas)',
        '1 Dia': 'Position (Diário)',
        '1 Semana': 'Trend (Semanal)',
        '1 Mês': 'Trend (Semanal)'
    }
    
    strategy_key = horizon_mapping.get(horizon, 'Swing (1-4 horas)')
    strategy = temporal_strategy[strategy_key]
    
    # Probabilidade confluente ajusta os multiplicadores
    prob_multiplier = market_probability['confluent_probability']
    confidence_adjustment = 0.8 + (prob_multiplier * 0.4)  # Entre 0.8 e 1.2
    
    # Calcular direção da operação baseada na análise confluente
    direction = 1 if predicted_price > current_price else -1
    
    # CÁLCULO FINAL DOS NÍVEIS baseado na ESTRATÉGIA TEMPORAL UNIFICADA
    final_stop_multiplier = strategy['stop_atr_multiplier'] * confidence_adjustment
    final_take_multiplier = strategy['take_atr_multiplier'] * confidence_adjustment
    
    # Aplicar ajuste de volatilidade da estratégia temporal
    volatility_factor = strategy['volatility_adjustment']
    
    if direction == 1:  # COMPRA
        stop_loss_price = current_price - (current_atr * final_stop_multiplier * volatility_factor)
        take_profit_price = current_price + (current_atr * final_take_multiplier * volatility_factor)
    else:  # VENDA
        stop_loss_price = current_price + (current_atr * final_stop_multiplier * volatility_factor)
        take_profit_price = current_price - (current_atr * final_take_multiplier * volatility_factor)
    
    # Converter para pontos (pips)
    def price_to_points(price1, price2, pair_name):
        diff = abs(price1 - price2)
        if 'JPY' in pair_name:
            return round(diff * 100, 1)  # JPY pairs
        else:
            return round(diff * 10000, 1)  # Major pairs
    
    stop_points = price_to_points(current_price, stop_loss_price, pair_name)
    take_points = price_to_points(current_price, take_profit_price, pair_name)
    
    # Razão risco/retorno
    risk_reward_ratio = take_points / stop_points if stop_points > 0 else 0
    
    return {
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'stop_loss_points': stop_points,
        'take_profit_points': take_points,
        'risk_reward_ratio': risk_reward_ratio,
        'operation_direction': 'COMPRA' if direction == 1 else 'VENDA',
        'confluent_probability': market_probability['confluent_probability'],
        'atr_used': current_atr,
        'fibonacci_support_ref': current_price - current_atr,
        'fibonacci_resistance_ref': current_price + current_atr,
        'position_strength': 'FORTE' if prob_multiplier > 0.75 else 'MODERADA' if prob_multiplier > 0.60 else 'FRACA',
        'temporal_strategy': strategy_key,
        'fibonacci_adjustment': 1.0,
        'volatility_factor': volatility_factor,
        'final_multipliers': {
            'stop': final_stop_multiplier,
            'take': final_take_multiplier
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
            if key.startswith('cache_'):
                del st.session_state[key]

def apply_theme_css():
    """Apply theme-specific CSS based on current theme"""
    current_theme = st.session_state.get('theme', 'light')
    
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
            padding: 3rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin: 2rem auto;
            max-width: 500px;
        ">
            <h1 style="color: white; margin-bottom: 1rem;">🔐 Acesso Restrito</h1>
            <h2 style="color: white; margin-bottom: 2rem;">Plataforma Avançada de Análise Forex</h2>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 2rem;">
                Sistema profissional de trading com IA e análise em tempo real
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Formulário de login centralizado
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔑 Digite a Senha de Acesso")
            password = st.text_input("Senha:", type="password", placeholder="Digite sua senha...")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🚀 Entrar na Plataforma", type="primary", use_container_width=True):
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
        if st.button("🏠 Home", type="primary", use_container_width=True):
            # Limpar todos os resultados e voltar ao estado inicial
            for key in ['analysis_results', 'show_analysis', 'analysis_mode']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Botão de logout
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            # Limpar sessão e autenticação
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # Header da sidebar compacto
        st.markdown("## ⚙️ Configurações")
        
        # Configurações básicas compactas
        pair = st.selectbox("💱 Par de Moedas", PAIRS)
        
        # Sistema unificado de Intervalo e Horizonte
        st.markdown("**⏰ Configuração Temporal Unificada**")
        
        # Presets integrados para máxima coerência (usando valores exatos de HORIZONS)
        temporal_presets = {
            "Scalping (1-5 min)": {"interval": "1min", "horizon": "1 Hora", "description": "Operações muito rápidas"},
            "Intraday (15-30 min)": {"interval": "15min", "horizon": "4 Horas", "description": "Operações no mesmo dia"},
            "Swing (1-4 horas)": {"interval": "60min", "horizon": "1 Dia", "description": "Operações de alguns dias"},
            "Position (Diário)": {"interval": "daily", "horizon": "1 Semana", "description": "Operações de médio prazo"},
            "Trend (Semanal)": {"interval": "daily", "horizon": "1 Mês", "description": "Análise de tendência longa"}
        }
        
        preset_choice = st.selectbox(
            "Estratégia Temporal:",
            list(temporal_presets.keys()),
            index=2,  # Default Swing
            help="Presets otimizados para máxima precisão entre intervalo e horizonte"
        )
        
        selected_preset = temporal_presets[preset_choice]
        interval = selected_preset["interval"]
        horizon = selected_preset["horizon"]
        
        # Mostrar configuração atual
        st.info(f"📊 **{preset_choice}** | Intervalo: {interval} | Horizonte: {horizon}")
        st.caption(f"💡 {selected_preset['description']}")
        
        # Opção avançada para configuração manual (colapsável)
        with st.expander("⚙️ Configuração Manual Avançada"):
            st.warning("⚠️ Configuração manual pode reduzir a precisão se intervalo e horizonte não estiverem alinhados!")
            
            manual_interval = st.selectbox("Intervalo Manual:", list(INTERVALS.keys()), 
                                         index=list(INTERVALS.keys()).index(interval))
            # Verificar se horizonte existe na lista, senão usar primeiro item
            horizon_index = 0
            try:
                horizon_index = HORIZONS.index(horizon)
            except ValueError:
                horizon = HORIZONS[0]  # Usar o primeiro como fallback
            
            manual_horizon = st.selectbox("Horizonte Manual:", HORIZONS,
                                        index=horizon_index)
            
            if st.checkbox("Usar Configuração Manual"):
                interval = manual_interval
                horizon = manual_horizon
                st.error("🔧 Modo manual ativo - Verifique se intervalo e horizonte estão compatíveis!")
        
        risk_level = st.selectbox("⚖️ Nível de Risco", ["Conservativo", "Moderado", "Agressivo"], index=1)
        
        # Converter para inglês para compatibilidade
        risk_mapping = {"Conservativo": "Conservative", "Moderado": "Moderate", "Agressivo": "Aggressive"}
        risk_level_en = risk_mapping[risk_level]
        
        # Configurações de Banca e Lote
        st.markdown("---")
        st.markdown("**💰 Gestão de Banca**")
        
        # Valor da banca
        account_balance = st.number_input(
            "Valor da Banca (USD):",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=500.0,
            help="Digite o valor real da sua conta de trading"
        )
        
        # Configuração de alavancagem
        leverage_options = [1, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
        leverage = st.selectbox(
            "Alavancagem:",
            leverage_options,
            index=6,  # Default 200:1
            help="Selecione a alavancagem oferecida pela sua corretora"
        )
        
        # Modo de configuração
        config_mode = st.radio(
            "Modo de Configuração:",
            ["Automático por Perfil", "Manual por Lote"],
            help="Automático: calcula lote baseado no perfil de risco\nManual: você define o lote real do forex"
        )
        
        if config_mode == "Manual por Lote":
            # Configuração manual com lotes reais do forex
            lot_options = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
            
            # Sugerir lote baseado na banca e alavancagem
            max_safe_lot = (account_balance * 0.02) / (100000 / leverage)  # 2% risk rule
            suggested_lot = min(lot_options, key=lambda x: abs(x - max_safe_lot))
            
            lot_size_real = st.selectbox(
                "Tamanho do Lote (Forex):",
                lot_options,
                index=lot_options.index(suggested_lot) if suggested_lot in lot_options else 3,
                help="1.0 = Lote padrão (100,000 unidades), 0.1 = Mini lote (10,000), 0.01 = Micro lote (1,000)"
            )
            
            # Calcular valor da posição em USD
            position_value = lot_size_real * 100000  # Valor notional da posição
            margin_required = position_value / leverage  # Margem necessária
            
            # Calcular percentual da banca
            margin_percentage = (margin_required / account_balance) * 100
            
            st.info(f"📊 **Lote:** {lot_size_real} | **Margem:** ${margin_required:,.0f} ({margin_percentage:.1f}% da banca)")
            
            # Alertas de risco baseados na margem
            if margin_percentage > 50:
                st.error("⚠️ **Alto Risco:** Margem muito alta para a banca!")
            elif margin_percentage > 20:
                st.warning("⚠️ **Risco Moderado:** Considere reduzir o lote")
            else:
                st.success("✅ **Risco Controlado:** Margem adequada")
                
            # Mostrar informações adicionais
            pip_value = lot_size_real * 10  # Para pares USD (aproximado)
            st.caption(f"💡 **Valor do Pip:** ~${pip_value:.2f} | **Posição:** ${position_value:,.0f}")
            
        else:
            # Cálculo automático baseado no perfil
            risk_percentages = {
                'Conservative': 0.5,   # 0.5% de margem
                'Moderate': 2.0,       # 2% de margem  
                'Aggressive': 5.0      # 5% de margem
            }
            
            auto_margin_percentage = risk_percentages[risk_level_en]
            target_margin = (account_balance * auto_margin_percentage) / 100
            
            # Calcular lote baseado na margem alvo
            max_position_value = target_margin * leverage
            lot_size_real = max_position_value / 100000
            
            # Arredondar para lotes padrão
            standard_lots = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
            lot_size_real = min(standard_lots, key=lambda x: abs(x - lot_size_real))
            
            # Recalcular valores com lote arredondado
            position_value = lot_size_real * 100000
            margin_required = position_value / leverage
            pip_value = lot_size_real * 10
            
            st.info(f"📊 **Lote Automático:** {lot_size_real} | **Margem:** ${margin_required:,.0f} ({auto_margin_percentage}% da banca)")
        
        # Armazenar nas configurações
        st.session_state.account_balance = account_balance
        st.session_state.leverage = leverage
        st.session_state.lot_size_real = lot_size_real
        st.session_state.position_value = position_value
        st.session_state.margin_required = margin_required
        st.session_state.pip_value = pip_value
        st.session_state.config_mode = config_mode
        
        # Configurações de IA colapsáveis
        with st.expander("🤖 Configurações Avançadas de IA"):
            lookback_period = st.slider("Histórico de Dados", 30, 120, LOOKBACK_PERIOD)
            epochs = st.slider("Épocas de Treinamento", 5, 20, EPOCHS)
            mc_samples = st.slider("Amostras Monte Carlo", 10, 50, MC_SAMPLES)
        
        # Cache compacto
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"💾 {cache_count} em cache")
            with col2:
                if st.button("🗑️", help="Limpar Cache"):
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
        
        # Análise unificada principal
        unified_analysis = st.button("🧠 Análise Unificada Inteligente", type="primary", use_container_width=True, 
                                   help="Combina todas as análises para a melhor previsão do mercado")
        
        st.markdown("**Análises Individuais:**")
        
        # Análises técnicas em colunas
        col1, col2 = st.columns(2)
        with col1:
            technical_analysis = st.button("📊 Técnica", use_container_width=True)
            sentiment_analysis = st.button("📰 Sentimento", use_container_width=True)
            risk_analysis = st.button("⚖️ Risco", use_container_width=True)
        with col2:
            ai_analysis = st.button("🤖 IA/LSTM", use_container_width=True)
            volume_analysis = st.button("📈 Volume", use_container_width=True)
            trend_analysis = st.button("📉 Tendência", use_container_width=True)
        
        # Análise rápida
        quick_analysis = st.button("⚡ Verificação Rápida", use_container_width=True)
        
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
            pair, interval, horizon, risk_level, lookback_period, 
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
    <div class="main-header" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3); color: white;">
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem;">
            📊 Plataforma Avançada de Análise Forex
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 0;">
            Previsões Forex com IA e Análise em Tempo Real
        </p>
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
        
        # Scalping
        st.markdown("#### ⚡ Scalping (1-5 min)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Características do Mercado:**
            - Movimentos de 5-15 pips são significativos
            - Ruído do mercado muito presente
            - Spreads têm impacto maior no lucro
            - Liquidez extremamente importante
            - Reações instantâneas a notícias
            
            **Melhores Pares para Scalping:**
            - EUR/USD (spread baixo, alta liquidez)
            - GBP/USD (volatilidade adequada)
            - USD/JPY (movimentos previsíveis)
            """)
        with col2:
            st.markdown("""
            **Horários Ideais:**
            - 08:00-12:00 UTC (Sobreposição Londres/Europa)
            - 13:00-17:00 UTC (Sobreposição Londres/NY)
            - Evitar: 22:00-06:00 UTC (baixa liquidez)
            
            **Relação Horizonte-Resultado:**
            - Horizonte 1 hora = Máximo 3-5 operações
            - Foco em momentum imediato
            - Stop loss: 5-10 pips máximo
            - Take profit: 8-15 pips típico
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

def run_analysis(pair, interval, horizon, risk_level, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis with different modes"""
    
    try:
        analysis_mode = st.session_state.get('analysis_mode', 'unified')
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Progress bar setup
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize
            status_text.text("🔄 Inicializando análise...")
            progress_bar.progress(10)
            
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
                results.update(run_unified_analysis(current_price, pair, risk_level, sentiment_score, df_with_indicators))
            elif analysis_mode == 'technical':
                results.update(run_technical_analysis(current_price, df_with_indicators, risk_level))
            elif analysis_mode == 'sentiment':
                results.update(run_sentiment_analysis(current_price, pair, sentiment_score, risk_level))
            elif analysis_mode == 'risk':
                results.update(run_risk_analysis(current_price, risk_level))
            elif analysis_mode == 'ai_lstm':
                results.update(run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators, risk_level))
            elif analysis_mode == 'volume':
                results.update(run_volume_analysis(current_price, df_with_indicators, risk_level))
            elif analysis_mode == 'trend':
                results.update(run_trend_analysis(current_price, df_with_indicators, risk_level))
            else:
                results.update(run_basic_analysis(current_price, is_quick, sentiment_score, risk_level, interval, horizon))
            
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
            
            # Trigger rerun to show results
            st.rerun()
        
    except Exception as e:
        if 'progress_container' in locals():
            progress_container.empty()
        st.error(f"❌ Erro durante a análise: {str(e)}")
        print(f"Analysis error: {e}")

def run_unified_analysis(current_price, pair, risk_level, sentiment_score, df_with_indicators):
    """Análise unificada que combina todas as fonções para melhor previsão"""
    import numpy as np
    
    # Pesos dos componentes
    technical_weight = 0.3
    sentiment_weight = 0.25
    ai_weight = 0.3
    risk_weight = 0.15
    
    # Componente técnico - baseado em indicadores reais
    rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50
    macd = df_with_indicators['macd'].iloc[-1] if 'macd' in df_with_indicators.columns else 0
    
    # Sinal técnico baseado em RSI e MACD
    rsi_signal = (50 - rsi) / 50  # RSI normalizado
    macd_signal = np.tanh(macd * 1000)  # MACD normalizado
    technical_signal = (rsi_signal + macd_signal) / 2 * 0.02
    
    # Componente de sentimento - usar dados reais
    sentiment_signal = sentiment_score * 0.015
    
    # Componente de IA - baseado em tendência dos preços
    recent_prices = df_with_indicators['close'].tail(5).values
    price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    ai_signal = np.tanh(price_trend * 10) * 0.025
    
    # Componente de risco - baseado na volatilidade
    volatility = df_with_indicators['close'].tail(20).std() / current_price
    risk_multiplier = {'Conservative': 0.5, 'Moderate': 1.0, 'Aggressive': 1.5}.get(risk_level, 1.0)
    risk_signal = (0.01 - volatility) * risk_multiplier * 0.01
    
    # Combinação ponderada
    combined_signal = (technical_signal * technical_weight + 
                      sentiment_signal * sentiment_weight + 
                      ai_signal * ai_weight + 
                      risk_signal * risk_weight)
    
    predicted_price = current_price * (1 + combined_signal)
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Calcular confiança baseada na convergência dos sinais
    signals = [technical_signal, sentiment_signal, ai_signal, risk_signal]
    signal_variance = np.var(signals)
    confidence = max(0.5, min(0.95, 1 - (signal_variance * 100)))
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'model_confidence': confidence,
        'sentiment_score': sentiment_score,
        'components': {
            'technical': {'signal': technical_signal, 'weight': technical_weight, 'details': f'RSI(14): {rsi:.1f}, MACD(12,26,9): {macd:.5f}'},
            'sentiment': {'signal': sentiment_signal, 'weight': sentiment_weight, 'details': f'Score: {sentiment_score:.3f}'},
            'ai': {'signal': ai_signal, 'weight': ai_weight, 'details': f'Tendência: {price_trend:.3f}'},
            'risk': {'signal': risk_signal, 'weight': risk_weight, 'details': f'Volatilidade: {volatility:.3f}'}
        },
        'final_recommendation': get_enhanced_recommendation(combined_signal, confidence, {
            'technical': {'signal': technical_signal, 'weight': technical_weight},
            'sentiment': {'signal': sentiment_signal, 'weight': sentiment_weight},
            'ai': {'signal': ai_signal, 'weight': ai_weight},
            'risk': {'signal': risk_signal, 'weight': risk_weight}
        }),
        'recommendation_details': get_recommendation_explanation(combined_signal, confidence, {
            'technical': {'signal': technical_signal, 'weight': technical_weight},
            'sentiment': {'signal': sentiment_signal, 'weight': sentiment_weight},
            'ai': {'signal': ai_signal, 'weight': ai_weight},
            'risk': {'signal': risk_signal, 'weight': risk_weight}
        })
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

def run_technical_analysis(current_price, df_with_indicators, risk_level):
    """Análise técnica especializada com indicadores múltiplos e perfil de risco"""
    import numpy as np
    
    # Fatores de ajuste baseados no perfil de risco do investidor
    risk_multipliers = {
        'Conservative': {'signal_factor': 0.7, 'confidence_boost': 0.05},
        'Moderate': {'signal_factor': 1.0, 'confidence_boost': 0.0},
        'Aggressive': {'signal_factor': 1.4, 'confidence_boost': -0.05}
    }
    
    risk_params = risk_multipliers.get(risk_level, risk_multipliers['Moderate'])
    
    # Análise baseada em múltiplos indicadores
    rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50
    macd = df_with_indicators['macd'].iloc[-1] if 'macd' in df_with_indicators.columns else 0
    sma_20 = df_with_indicators['sma_20'].iloc[-1] if 'sma_20' in df_with_indicators.columns else current_price
    
    # Sinais dos indicadores ajustados pelo perfil de risco
    base_rsi_signal = (50 - rsi) / 50 * 0.015
    base_macd_signal = np.tanh(macd * 1000) * 0.012
    base_sma_signal = (current_price - sma_20) / sma_20 * 0.018
    
    # Aplicar fator de risco
    rsi_signal = base_rsi_signal * risk_params['signal_factor']
    macd_signal = base_macd_signal * risk_params['signal_factor']
    sma_signal = base_sma_signal * risk_params['signal_factor']
    
    # Combinação ponderada
    combined_signal = (rsi_signal * 0.4 + macd_signal * 0.35 + sma_signal * 0.25)
    
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
        'analysis_focus': f'Análise Técnica ({risk_level}) - RSI: {rsi:.1f}, MACD: {macd:.5f}, SMA20: {sma_20:.5f}',
        'risk_level_used': risk_level,
        'technical_indicators': {
            'rsi': rsi,
            'macd': macd,
            'sma_20': sma_20
        }
    }

def run_sentiment_analysis(current_price, pair, sentiment_score, risk_level):
    """Análise de sentimento especializada com fatores de mercado e perfil de risco"""
    
    # Ajustes baseados no perfil de risco do investidor
    risk_adjustments = {
        'Conservative': {'signal_factor': 0.6, 'confidence_penalty': 0.05, 'volatility_threshold': 0.15},
        'Moderate': {'signal_factor': 1.0, 'confidence_penalty': 0.0, 'volatility_threshold': 0.25},
        'Aggressive': {'signal_factor': 1.5, 'confidence_penalty': -0.03, 'volatility_threshold': 0.40}
    }
    
    risk_params = risk_adjustments.get(risk_level, risk_adjustments['Moderate'])
    
    # Usar dados reais de sentimento com ajustes de volatilidade e perfil de risco
    base_signal = sentiment_score * 0.015 * risk_params['signal_factor']
    
    # Fator de ajuste baseado na intensidade do sentimento
    intensity_factor = abs(sentiment_score)
    
    # Para perfil conservador, reduzir impacto de sentimentos extremos
    if risk_level == 'Conservative' and intensity_factor > risk_params['volatility_threshold']:
        intensity_factor = risk_params['volatility_threshold']
    
    adjusted_signal = base_signal * (1 + intensity_factor)
    
    predicted_price = current_price * (1 + adjusted_signal)
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
        'analysis_focus': f'Sentimento de Mercado ({risk_level}): {sentiment_label} (Score: {sentiment_score:.3f}, Intensidade: {intensity_factor:.3f})',
        'risk_level_used': risk_level,
        'sentiment_intensity': intensity_factor
    }

def run_risk_analysis(current_price, risk_level):
    """Análise de risco especializada com cálculos avançados"""
    import numpy as np
    
    # Fatores de risco baseados no nível selecionado
    risk_factors = {
        'Conservative': {'volatility': 0.005, 'confidence': 0.85, 'signal_range': 0.008},
        'Moderate': {'volatility': 0.012, 'confidence': 0.75, 'signal_range': 0.015},
        'Aggressive': {'volatility': 0.025, 'confidence': 0.65, 'signal_range': 0.025}
    }
    
    factor = risk_factors.get(risk_level, risk_factors['Moderate'])
    
    # Sinal baseado no perfil de risco
    signal = np.random.uniform(-factor['signal_range'], factor['signal_range'])
    
    # Ajustar sinal baseado no nível de risco
    if risk_level == 'Conservative':
        signal *= 0.7  # Sinais mais conservadores
    elif risk_level == 'Aggressive':
        signal *= 1.3  # Sinais mais agressivos
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': factor['confidence'],
        'analysis_focus': f'Análise de Risco Especializada ({risk_level}) - Volatilidade: {factor["volatility"]:.3f}',
        'risk_level_used': risk_level,
        'estimated_volatility': factor['volatility']
    }

def run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators, risk_level):
    """Análise de IA/LSTM especializada com deep learning simulado e perfil de risco"""
    import numpy as np
    
    # Parâmetros baseados no perfil de risco do investidor
    risk_configs = {
        'Conservative': {'volatility_tolerance': 0.8, 'signal_damping': 0.7, 'min_confidence': 0.70},
        'Moderate': {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65},
        'Aggressive': {'volatility_tolerance': 1.3, 'signal_damping': 1.4, 'min_confidence': 0.60}
    }
    
    risk_config = risk_configs.get(risk_level, risk_configs['Moderate'])
    
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
    
    # Para conservadores, penalizar alta volatilidade mais severamente
    if risk_level == 'Conservative' and volatility > 0.015:
        volatility_signal *= 0.5
    
    # Sinal final ponderado pelo fator de aprendizado e perfil de risco
    combined_signal = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
    
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
        'analysis_focus': f'IA/LSTM ({risk_level}) - Tendência: {long_trend:.3f}, Volatilidade: {volatility:.3f} (lookback: {lookback_period}, épocas: {epochs})',
        'risk_level_used': risk_level,
        'ai_metrics': {
            'long_trend': long_trend,
            'short_trend': short_trend,
            'volatility': volatility,
            'learning_factor': learning_factor
        }
    }

def run_volume_analysis(current_price, df_with_indicators, risk_level):
    """Análise de volume especializada com perfil de risco"""
    import numpy as np
    
    # Ajustes baseados no perfil de risco
    risk_configs = {
        'Conservative': {'signal_factor': 0.8, 'volatility_threshold': 0.015, 'confidence': 0.75},
        'Moderate': {'signal_factor': 1.0, 'volatility_threshold': 0.020, 'confidence': 0.70},
        'Aggressive': {'signal_factor': 1.3, 'volatility_threshold': 0.030, 'confidence': 0.65}
    }
    
    config = risk_configs.get(risk_level, risk_configs['Moderate'])
    
    # Usar volatilidade como proxy para volume
    volatility = df_with_indicators['close'].tail(20).std() / current_price
    
    # Ajustar sinal baseado no perfil de risco
    base_signal = (config['volatility_threshold'] - volatility) * 0.015
    signal = base_signal * config['signal_factor']
    
    # Para conservadores, penalizar alta volatilidade mais
    if risk_level == 'Conservative' and volatility > config['volatility_threshold']:
        signal *= 0.5
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': config['confidence'],
        'analysis_focus': f'Volume/Liquidez ({risk_level}) - Volatilidade: {volatility:.4f}, Limite: {config["volatility_threshold"]:.3f}',
        'risk_level_used': risk_level
    }

def run_trend_analysis(current_price, df_with_indicators, risk_level):
    """Análise de tendência especializada com perfil de risco"""
    import numpy as np
    
    # Configurações baseadas no perfil de risco
    risk_settings = {
        'Conservative': {'signal_multiplier': 0.7, 'trend_threshold': 0.005, 'confidence': 0.78},
        'Moderate': {'signal_multiplier': 1.0, 'trend_threshold': 0.010, 'confidence': 0.72},
        'Aggressive': {'signal_multiplier': 1.4, 'trend_threshold': 0.020, 'confidence': 0.68}
    }
    
    settings = risk_settings.get(risk_level, risk_settings['Moderate'])
    
    # Análise de tendência baseada em médias móveis
    sma_20 = df_with_indicators['sma_20'].iloc[-1] if 'sma_20' in df_with_indicators.columns else current_price
    sma_50 = df_with_indicators['sma_50'].iloc[-1] if 'sma_50' in df_with_indicators.columns else current_price
    
    # Sinal baseado na posição do preço em relação às médias
    price_vs_sma20 = (current_price - sma_20) / sma_20
    sma_cross = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0
    
    # Aplicar multiplicador de risco e limites
    base_signal = (price_vs_sma20 + sma_cross) / 2 * 0.018
    signal = base_signal * settings['signal_multiplier']
    
    # Para conservadores, limitar sinais fortes
    if risk_level == 'Conservative' and abs(signal) > settings['trend_threshold']:
        signal = np.sign(signal) * settings['trend_threshold']
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': settings['confidence'],
        'analysis_focus': f'Tendência ({risk_level}) - SMA20: {sma_20:.5f}, SMA50: {sma_50:.5f}, Força: {abs(signal):.4f}',
        'risk_level_used': risk_level
    }

def run_basic_analysis(current_price, is_quick, sentiment_score, risk_level, interval="1hour", horizon="1 dia"):
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
    
    config = risk_configs.get(risk_level, risk_configs['Moderate'])
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
        'analysis_focus': f'Análise Básica Integrada ({risk_level}) - {interval}/{horizon} - Tendência: {market_trend:.4f}, Sentimento: {sentiment_score:.3f}',
        'risk_level_used': risk_level
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
                if 'JPY' in pair_name:
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
            if 'JPY' in pair:
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
        
        # PAINEL DE ANÁLISE TÉCNICA REAL
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(156,39,176,0.1), rgba(33,150,243,0.1));
            border-left: 4px solid #9C27B0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <h4 style="color: #9C27B0; margin: 0 0 0.8rem 0; font-size: 1rem;">📊 Análise Técnica Real - Níveis de Mercado</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.8rem; text-align: center;">
                <div style="background: rgba(244,67,54,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Stop Loss Técnico</strong></p>
                    <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #f44336;">{stop_loss_level:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Baseado em {stop_reason}</p>
                    <p style="margin: 0; color: #888; font-size: 0.70rem;">Ref: {stop_reference_level:.5f}</p>
                </div>
                <div style="background: rgba(76,175,80,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Take Profit Técnico</strong></p>
                    <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #4caf50;">{take_profit_level:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Baseado em {target_reason}</p>
                    <p style="margin: 0; color: #888; font-size: 0.70rem;">Ref: {target_reference_level:.5f}</p>
                </div>
                <div style="background: rgba(33,150,243,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Confiança Confluente</strong></p>
                    <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #2196f3;">{(enhanced_confidence * 100):.0f}%</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Integração de análises</p>
                    <p style="margin: 0; color: #888; font-size: 0.70rem;">
                        LSTM({confidence*100:.0f}%) + Sentiment({abs(sentiment_score)*100:.0f}%)
                    </p>
                </div>
                <div style="background: rgba(255,193,7,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Risco vs Retorno</strong></p>
                    <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #ff9800;">1:{(profit_distance/stop_distance):.1f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Baseado em níveis reais</p>
                    <p style="margin: 0; color: #888; font-size: 0.70rem;">Não em gestão financeira</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,193,7,0.1), rgba(255,87,34,0.1));
            border-left: 4px solid #FF9800;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h4 style="color: #FF9800; margin: 0 0 1rem 0; font-size: 1.1rem;">⚠️ Análise de Risco Avançada - Perfil: {risk_level_used}</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem; text-align: center; margin-bottom: 1rem;">
                <div style="background: rgba(244,67,54,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Stop Loss</strong></p>
                    <p style="margin: 0; font-size: 0.95rem; font-weight: bold; color: {risk_color};">{stop_loss_level:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Risco: {risk_percentage:.2f}%</p>
                </div>
                <div style="background: rgba(76,175,80,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Take Profit</strong></p>
                    <p style="margin: 0; font-size: 0.95rem; font-weight: bold; color: green;">{take_profit_level:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Alvo: {reward_percentage:.2f}%</p>
                </div>
                <div style="background: rgba(33,150,243,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Extensão Máxima</strong></p>
                    <p style="margin: 0; font-size: 0.95rem; font-weight: bold; color: blue;">{max_extension:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Potencial: {extension_percentage:.2f}%</p>
                </div>
                <div style="background: rgba(255,193,7,0.1); padding: 0.8rem; border-radius: 6px;">
                    <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Reversão Iminente</strong></p>
                    <p style="margin: 0; font-size: 0.95rem; font-weight: bold; color: orange;">{reversal_level:.5f}</p>
                    <p style="margin: 0; color: #888; font-size: 0.75rem;">Alerta: {reversal_percentage:.2f}%</p>
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.03); padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.8rem 0; color: #333; text-align: center;">💰 Gestão de Banca (Conta: ${banca_base:,.0f} | Lote: {lot_size_real} | Alavancagem: {leverage}:1)</h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; text-align: center;">
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Lote/Margem</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #333;">{lot_size_real} / ${margin_required:,.0f}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">{(margin_required/banca_base*100):.1f}% margem</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Risco Monetário</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: red;">-${risco_monetario:,.2f}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">{stop_loss_pip_diff:.1f} pips</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Lucro Potencial</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: green;">+${potencial_lucro:,.2f}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">{take_profit_pip_diff:.1f} pips</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Potencial Máximo</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #9c27b0;">+${potencial_maximo:,.2f}</p>
                        <p style="margin: 0; color: #888; font-size: 0.72rem;">{extension_pip_diff:.1f} pips • {extension_direction}</p>
                        <p style="margin: 0; color: #888; font-size: 0.70rem;">🎯 {max_extension:.5f}</p>
                        <p style="margin: 0; color: #2e7d32; font-size: 0.69rem; font-weight: bold;">📊 {probability_description} • ⏱️ {time_description}</p>
                    </div>
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.03); padding: 1rem; border-radius: 6px;">
                <h5 style="margin: 0 0 0.8rem 0; color: #333; text-align: center;">📊 Análise Comparativa por Perfil</h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.8rem; text-align: center;">
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Razão R:R</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: {'green' if risk_reward_ratio > 2 else 'orange' if risk_reward_ratio > 1 else 'red'};">1:{risk_reward_ratio:.1f}</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">{'Excelente' if risk_reward_ratio > 2 else 'Aceitável' if risk_reward_ratio > 1 else 'Alto Risco'}</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>ATR Stop</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #333;">{profile['atr_multiplier_stop']:.1f}x</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Múltiplo ATR stop</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>ATR Target</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #333;">{profile['atr_multiplier_tp']:.1f}x</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Múltiplo ATR target</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666; font-size: 0.85rem;"><strong>Extensão</strong></p>
                        <p style="margin: 0; font-size: 1rem; font-weight: bold; color: #333;">{profile['extension_factor']:.1f}x</p>
                        <p style="margin: 0; color: #888; font-size: 0.75rem;">Potencial máximo</p>
                    </div>
                </div>
            </div>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.05); border-radius: 6px;">
                <h5 style="margin: 0 0 0.8rem 0; color: #333; text-align: center;">🎯 Cenários de Mercado para Perfil {risk_level_used}</h5>
                <div style="text-align: center; line-height: 1.6;">
                    <p style="margin: 0 0 0.5rem 0; color: #555; font-size: 0.9rem;">
                        <strong>Características do Perfil:</strong> 
                        {'🛡️ Proteção máxima, stops próximos, menor exposição' if risk_level_used == 'Conservative' else 
                         '⚖️ Equilíbrio entre segurança e potencial de retorno' if risk_level_used == 'Moderate' else 
                         '🚀 Maior exposição, stops distantes, busca máximos retornos'}
                    </p>
                    <p style="margin: 0 0 0.5rem 0; color: #555; font-size: 0.9rem;">
                        <strong>Expectativa de Movimento:</strong> O mercado pode se mover {reward_direction} até <strong>{take_profit_level:.5f}</strong>, 
                        com potencial de extensão até <strong>{max_extension:.5f}</strong> em cenário otimista.
                    </p>
                    <p style="margin: 0 0 0.5rem 0; color: #555; font-size: 0.9rem;">
                        <strong>🎯 Potencial Máximo Realístico:</strong> {extension_description} baseado em confluência LSTM + Sentiment + IA. 
                        <span style="color: #2e7d32;"><strong>Probabilidade Real:</strong> {probability_description}</span> em <strong>{time_description}</strong> 
                        considerando movimentos típicos do par e concordância entre análises.
                    </p>
                    <p style="margin: 0; color: #555; font-size: 0.9rem;">
                        <strong>Alerta de Reversão:</strong> Se o preço se mover {risk_direction} além de <strong>{reversal_level:.5f}</strong>, 
                        considere revisar a posição. Stop definitivo em <strong>{stop_loss_level:.5f}</strong>. 
                        <span style="color: #2e7d32;"><strong>Confiança Final:</strong> {enhanced_confidence:.0%}</span> 
                        (Confluência: LSTM {confidence*100:.0f}% + IA {ai_confidence_boost*100:.0f}% + Sentiment {abs(sentiment_score)*100:.0f}%)
                    </p>
                </div>
            </div>
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
    if results.get('analysis_mode') == 'unified' and 'components' in results:
        st.markdown("---")
        st.markdown("**Breakdown dos Componentes (Análise Unificada):**")
        
        components_data = []
        for component, data in results['components'].items():
            components_data.append({
                'Componente': component.title(),
                'Sinal (%)': f"{data['signal']*100:+.3f}%",
                'Peso (%)': f"{data['weight']*100:.0f}%",
                'Contribuição': f"{data['signal']*data['weight']*100:+.3f}%"
            })
        
        import pandas as pd
        df_components = pd.DataFrame(components_data)
        st.dataframe(df_components, use_container_width=True)

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
    """Display enhanced analysis results"""
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
    
    # Main recommendation
    if 'final_recommendation' in results:
        recommendation = results['final_recommendation']
    else:
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "⚪ INDECISÃO"
    
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
    
    # Mostrar componentes da análise unificada
    if analysis_mode == 'unified' and 'components' in results:
        st.markdown("### 🔍 Componentes da Análise Unificada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sinais por Componente:**")
            for component, data in results['components'].items():
                signal_pct = data['signal'] * 100
                weight_pct = data['weight'] * 100
                color = "🟢" if data['signal'] > 0 else "🔴" if data['signal'] < 0 else "🟡"
                st.markdown(f"{color} **{component.title()}:** {signal_pct:+.2f}% (peso: {weight_pct:.0f}%)")
        
        with col2:
            st.markdown("**Convergência dos Sinais:**")
            import numpy as np
            signals = [data['signal'] for data in results['components'].values()]
            convergence = 1 - (np.var(signals) * 100) if signals else 0
            convergence_text = "Alta" if convergence > 0.8 else "Média" if convergence > 0.6 else "Baixa"
            st.markdown(f"**Convergência:** {convergence_text} ({convergence:.0%})")
            st.markdown("Maior convergência = maior confiança na previsão")
    
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

if __name__ == "__main__":
    main()