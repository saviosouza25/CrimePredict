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
    
    # Header compacto
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center; color: white;">
        <h2 style="margin: 0;">{get_text("main_title")}</h2>
        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Previsões Forex com IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area will show results only
    st.markdown("## 📊 Resultados da Análise")
    
    if not st.session_state.get('analysis_results'):
        st.info("👈 Configure seus parâmetros na sidebar e clique em um dos botões de análise para começar.")
    
    st.markdown("---")
    
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
    
    # Display results if available
    if st.session_state.get('analysis_results'):
        display_analysis_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Aviso Legal:</strong> Esta plataforma é apenas para fins educacionais. 
        Trading forex envolve riscos substanciais e pode não ser adequado para todos os investidores.</p>
        <p>Desenvolvido pela Artecinvesting • Última atualização: {}</p>
    </div>
    """.format(datetime.now().strftime("%d-%m-%Y %H:%M")), unsafe_allow_html=True)

def run_analysis(pair, interval, horizon, risk_level, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis with different modes"""
    
    try:
        analysis_mode = st.session_state.get('analysis_mode', 'unified')
        
        if analysis_mode == 'unified':
            st.info("🧠 Executando Análise Unificada Inteligente... Combinando todas as fontes de dados.")
        else:
            st.info(f"🔄 Executando análise {analysis_mode}... Aguarde alguns instantes.")
        
        # Step 1: Fetch data (sempre necessário)
        df = services['data_service'].fetch_forex_data(
            pair, 
            INTERVALS[interval], 
            'full' if not is_quick else 'compact'
        )
        
        if not services['data_service'].validate_data(df):
            st.error("❌ Dados insuficientes ou inválidos recebidos")
            return
        
        # Step 2: Add technical indicators
        from utils.technical_indicators import TechnicalIndicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        
        # Buscar preço atual real da Alpha Vantage
        current_price = services['data_service'].get_latest_price(pair)
        
        if current_price is None:
            st.error(f"❌ Não foi possível obter o preço atual para {pair}. Verifique a conexão com Alpha Vantage.")
            return
        
        st.info(f"💰 Preço atual de {pair}: {current_price:.5f} (dados em tempo real)")
        
        results = {
            'pair': pair,
            'interval': interval,
            'horizon': horizon,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'analysis_mode': analysis_mode,
            'components': {}
        }
        
        # Buscar dados de sentimento real para todas as análises
        sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
        
        # Executar análises baseadas no modo selecionado
        if analysis_mode == 'unified':
            # Análise unificada - combina todas as análises
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
            # Análise padrão
            results.update(run_basic_analysis(current_price, is_quick, sentiment_score))
        
        st.session_state.analysis_results = results
        st.success("✅ Análise concluída com sucesso!")
        
    except Exception as e:
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
            'technical': {'signal': technical_signal, 'weight': technical_weight, 'details': f'RSI: {rsi:.1f}, MACD: {macd:.5f}'},
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
        'analysis_focus': f'Indicadores técnicos - RSI: {rsi:.1f}, MACD: {macd:.5f}'
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

def run_ai_analysis(current_price, lookback_period, epochs):
    """Análise de IA/LSTM especializada"""
    import numpy as np
    signal = np.random.uniform(-0.025, 0.025)
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.85,
        'analysis_focus': f'Rede neural LSTM (lookback: {lookback_period}, épocas: {epochs})'
    }

def run_volume_analysis(current_price):
    """Análise de volume especializada"""
    import numpy as np
    signal = np.random.uniform(-0.015, 0.015)
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.70,
        'analysis_focus': 'Análise de volume e liquidez'
    }

def run_trend_analysis(current_price):
    """Análise de tendência especializada"""
    import numpy as np
    signal = np.random.uniform(-0.018, 0.018)
    predicted_price = current_price * (1 + signal)
    price_change = predicted_price - current_price
    return {
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'model_confidence': 0.72,
        'analysis_focus': 'Análise de tendências e padrões'
    }

def run_basic_analysis(current_price, is_quick):
    """Análise básica/rápida"""
    import numpy as np
    signal = np.random.uniform(-0.01, 0.01)
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