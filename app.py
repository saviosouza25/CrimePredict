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
        
        # Botões de análise compactos
        st.markdown("**🎯 Análises**")
        
        analyze_button = st.button("📊 Análise Completa", type="primary", use_container_width=True)
        quick_analysis = st.button("⚡ Análise Rápida", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 IA", use_container_width=True):
                st.session_state['analysis_mode'] = 'advanced_ai'
                analyze_button = True
        with col2:
            if st.button("📈 Dashboard", use_container_width=True):
                st.session_state['analysis_mode'] = 'dashboard'
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
    """Run the complete forex analysis"""
    
    try:
        st.info("🔄 Executando análise... Aguarde alguns instantes.")
        
        # Step 1: Fetch data
        df = services['data_service'].fetch_forex_data(
            pair, 
            INTERVALS[interval], 
            'full' if not is_quick else 'compact'
        )
        
        if not services['data_service'].validate_data(df):
            st.error("❌ Dados insuficientes ou inválidos recebidos")
            return
        
        # Step 2: Fetch sentiment
        sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
        
        # Step 3: Add technical indicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        trading_signals = TechnicalIndicators.get_trading_signals(df_with_indicators)
        
        # Step 4: Quick analysis or full prediction
        if is_quick:
            # Simple analysis based on recent data
            current_price = float(df_with_indicators['close'].iloc[-1])
            predicted_price = current_price * (1 + (sentiment_score * 0.01))
            model_confidence = 0.7  # Default confidence for quick analysis
        else:
            # Full LSTM prediction
            predictor = ForexPredictor(
                lookback=lookback_period,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            )
            
            # Use recent data for training
            train_data = df_with_indicators.tail(min(1000, len(df_with_indicators)))
            
            training_metrics = predictor.train_model(
                train_data, 
                sentiment_score, 
                epochs=epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE
            )
            
            # Make predictions
            steps = HORIZON_STEPS[horizon]
            predictions, uncertainties = predictor.predict_future(
                train_data, 
                sentiment_score, 
                steps, 
                mc_samples
            )
            
            current_price = float(df_with_indicators['close'].iloc[-1])
            predicted_price = predictions[-1] if predictions and len(predictions) > 0 else current_price
            model_confidence = predictor.get_model_confidence(train_data, sentiment_score)
        
        # Calculate metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Store results
        st.session_state.analysis_results = {
            'pair': pair,
            'interval': interval,
            'horizon': horizon,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'sentiment_score': sentiment_score,
            'model_confidence': model_confidence,
            'trading_signals': trading_signals,
            'df_with_indicators': df_with_indicators,
            'timestamp': datetime.now(),
            'is_quick': is_quick
        }
        
        st.success("✅ Análise concluída com sucesso!")
        
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")
        print(f"Analysis error: {e}")

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.get('analysis_results'):
        return
    
    results = st.session_state.analysis_results
    
    st.markdown("## 📊 Resultados da Análise")
    
    # Main recommendation
    direction = "📈 COMPRA" if results['price_change'] > 0 else "📉 VENDA"
    confidence_color = "green" if results['model_confidence'] > 0.7 else "orange" if results['model_confidence'] > 0.5 else "red"
    
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {confidence_color}; margin: 0;">{direction}</h3>
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
            "Sentimento do Mercado",
            f"{results['sentiment_score']:+.3f}",
            "Positivo" if results['sentiment_score'] > 0 else "Negativo"
        )
    
    with col2:
        st.metric(
            "Confiança do Modelo",
            f"{results['model_confidence']:.0%}",
            "Alta" if results['model_confidence'] > 0.7 else "Baixa"
        )
    
    with col3:
        analysis_type = "Rápida" if results['is_quick'] else "Completa"
        st.metric(
            "Tipo de Análise",
            analysis_type,
            f"Executada às {results['timestamp'].strftime('%H:%M:%S')}"
        )

if __name__ == "__main__":
    main()