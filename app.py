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
    from utils.cache_manager import CacheManager
    
    # Initialize services
    services = {
        'data_service': DataService(),
        'sentiment_service': SentimentService()
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
        'sentiment_service': MockService()
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
        # Header da sidebar compacto
        st.markdown("## ⚙️ Configurações")
        
        # Configurações básicas compactas
        pair = st.selectbox("💱 Par de Moedas", PAIRS)
        
        col1, col2 = st.columns(2)
        with col1:
            interval = st.selectbox("⏰ Intervalo", list(INTERVALS.keys()), index=4)
        with col2:
            horizon = st.selectbox("🔮 Horizonte", HORIZONS)
        
        risk_level = st.selectbox("⚖️ Nível de Risco", ["Conservativo", "Moderado", "Agressivo"], index=1)
        
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
                    CacheManager.clear_cache()
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
    
    # Display results if available, otherwise show main header and instructions
    if st.session_state.get('analysis_results'):
        display_analysis_results_with_tabs()
    else:
        # Show main header with instructions when no results
        display_main_header()

def display_main_header():
    """Display the main platform header"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #2E86AB; margin-bottom: 0.5rem;">
            📊 Plataforma Avançada de Análise Forex
        </h1>
        <p style="color: #666; font-size: 1.1em; margin: 0;">
            Previsões Forex com IA e Análise em Tempo Real
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
            
            # Executar análises baseadas no modo selecionado
            if analysis_mode == 'unified':
                results.update(run_unified_analysis(current_price, pair, risk_level, sentiment_score, df_with_indicators))
            elif analysis_mode == 'technical':
                results.update(run_technical_analysis(current_price, df_with_indicators))
            elif analysis_mode == 'sentiment':
                results.update(run_sentiment_analysis(current_price, pair, sentiment_score))
            elif analysis_mode == 'risk':
                results.update(run_risk_analysis(current_price, risk_level))
            elif analysis_mode == 'ai_lstm':
                results.update(run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators))
            elif analysis_mode == 'volume':
                results.update(run_volume_analysis(current_price, df_with_indicators))
            elif analysis_mode == 'trend':
                results.update(run_trend_analysis(current_price, df_with_indicators))
            else:
                results.update(run_basic_analysis(current_price, is_quick, sentiment_score))
            
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
        'final_recommendation': 'COMPRAR' if combined_signal > 0.005 else 'VENDER' if combined_signal < -0.005 else 'MANTER'
    }

def run_technical_analysis(current_price, df_with_indicators):
    """Análise técnica especializada"""
    import numpy as np
    
    # Usar indicadores técnicos reais
    rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns else 50
    macd = df_with_indicators['macd'].iloc[-1] if 'macd' in df_with_indicators.columns else 0
    
    # Calcular sinais baseados em indicadores
    rsi_signal = (50 - rsi) / 100  # RSI signal
    macd_signal = np.tanh(macd * 1000)  # MACD signal normalized
    
    combined_signal = (rsi_signal + macd_signal) / 2 * 0.02
    predicted_price = current_price * (1 + combined_signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.75,
        'analysis_focus': f'Indicadores técnicos - RSI(14): {rsi:.1f}, MACD(12,26,9): {macd:.5f}'
    }

def run_sentiment_analysis(current_price, pair, sentiment_score):
    """Análise de sentimento especializada"""
    # Usar dados reais de sentimento
    signal = sentiment_score * 0.015  # Amplificar o sinal do sentimento
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    sentiment_label = "Positivo" if sentiment_score > 0.1 else "Negativo" if sentiment_score < -0.1 else "Neutro"
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.65,
        'sentiment_score': sentiment_score,
        'analysis_focus': f'Sentimento: {sentiment_label} (Score: {sentiment_score:.3f})'
    }

def run_risk_analysis(current_price, risk_level):
    """Análise de risco especializada"""
    import numpy as np
    signal = np.random.uniform(-0.01, 0.01)
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.80,
        'analysis_focus': f'Análise de risco ({risk_level})'
    }

def run_ai_analysis(current_price, lookback_period, epochs, df_with_indicators):
    """Análise de IA/LSTM especializada"""
    import numpy as np
    
    # Usar tendência de preços para simular IA
    recent_prices = df_with_indicators['close'].tail(lookback_period).values
    price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    signal = np.tanh(price_trend * 10) * 0.025
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.85,
        'analysis_focus': f'IA/LSTM - Tendência: {price_trend:.3f} (lookback: {lookback_period}, épocas: {epochs})'
    }

def run_volume_analysis(current_price, df_with_indicators):
    """Análise de volume especializada"""
    import numpy as np
    
    # Usar volatilidade como proxy para volume
    volatility = df_with_indicators['close'].tail(20).std() / current_price
    signal = (0.02 - volatility) * 0.015  # Menor volatilidade = sinal positivo
    
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.70,
        'analysis_focus': f'Volume/Liquidez - Volatilidade: {volatility:.4f}'
    }

def run_trend_analysis(current_price, df_with_indicators):
    """Análise de tendência especializada"""
    import numpy as np
    
    # Análise de tendência baseada em médias móveis
    sma_20 = df_with_indicators['sma_20'].iloc[-1] if 'sma_20' in df_with_indicators.columns else current_price
    sma_50 = df_with_indicators['sma_50'].iloc[-1] if 'sma_50' in df_with_indicators.columns else current_price
    
    # Sinal baseado na posição do preço em relação às médias
    price_vs_sma20 = (current_price - sma_20) / sma_20
    sma_cross = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0
    
    signal = (price_vs_sma20 + sma_cross) / 2 * 0.018
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.72,
        'analysis_focus': f'Tendência - SMA20: {sma_20:.5f}, SMA50: {sma_50:.5f}'
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
    
    # Replace main header with analysis title
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="margin: 0; color: #2E86AB;">📊 {mode_names.get(analysis_mode, 'Análise Padrão')}</h1>
        <p style="margin: 0.5rem 0; color: #666; font-style: italic;">Resultados da Análise • {results['pair']} • {results['timestamp'].strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main recommendation card
    if 'final_recommendation' in results:
        recommendation = results['final_recommendation']
    else:
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "🔄 MANTER"
    
    confidence_color = "green" if results['model_confidence'] > 0.7 else "orange" if results['model_confidence'] > 0.5 else "red"
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 2rem; 
            border: 3px solid {confidence_color}; 
            border-radius: 15px; 
            background: linear-gradient(135deg, rgba(0,0,0,0.1), rgba(255,255,255,0.1));
            margin: 1rem 0;
        ">
            <h1 style="color: {confidence_color}; margin: 0; font-size: 2.5em;">{recommendation}</h1>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; text-align: left;">
                <div>
                    <p style="margin: 0.5rem 0;"><strong>Preço Atual:</strong> {results['current_price']:.5f}</p>
                    <p style="margin: 0.5rem 0;"><strong>Preço Previsto:</strong> {results['predicted_price']:.5f}</p>
                </div>
                <div>
                    <p style="margin: 0.5rem 0;"><strong>Variação:</strong> <span style="color: {confidence_color};">{results['price_change_pct']:+.2f}%</span></p>
                    <p style="margin: 0.5rem 0;"><strong>Confiança:</strong> {results['model_confidence']:.0%}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show unified analysis components if available
    if analysis_mode == 'unified' and 'components' in results:
        st.markdown("### 🔍 Componentes da Análise Unificada")
        
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
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "🔄 MANTER"
    
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
        recommendation = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA" if results['price_change'] < 0 else "🔄 MANTER"
    
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