import sys
import os

# Ensure we can find all installed packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import custom modules
from config.settings import *
from config.languages import get_text
from services.data_service import DataService
from services.sentiment_service import SentimentService
from services.indicators import TechnicalIndicators
from models.lstm_model import ForexPredictor
from utils.visualization import ForexVisualizer
from utils.cache_manager import CacheManager
import hashlib
import base64

# Authentication configuration
VALID_CREDENTIALS = {
    "artec": "e10adc3949ba59abbe56e057f20f883e"  # MD5 hash of "123456"
}

def get_logo_base64():
    """Get the company logo as base64 encoded string."""
    try:
        with open("assets/company_logo.png", "rb") as f:
            logo_data = f.read()
            return base64.b64encode(logo_data).decode()
    except FileNotFoundError:
        return ""

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == hashlib.md5(password.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
            del st.session_state["username"]  # Don't store the username
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password with company logo
    logo_base64 = get_logo_base64()
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 60vh;">
        <div style="text-align: center; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); background: white; min-width: 400px;">
            <div style="margin-bottom: 1.5rem;">
                <img src="data:image/png;base64,{logo_base64}" style="max-width: 120px; height: auto;" />
            </div>
            <h2 style="color: #333; margin-bottom: 2rem;">{get_text("login_title")}</h2>
            <p style="color: #666; margin-bottom: 2rem;">{get_text("login_subtitle")}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input(get_text("username_placeholder"), key="username", placeholder=get_text("username_placeholder"))
        st.text_input(get_text("password_placeholder"), type="password", key="password", placeholder=get_text("password_placeholder"), on_change=password_entered)
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error(get_text("invalid_credentials"))
            
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #888; font-size: 0.9em;">
            <p>{get_text("login_secure_text")}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return False

# Page configuration
st.set_page_config(
    page_title=get_text("login_title"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-alert {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .sidebar .stSelectbox label {
        font-weight: bold;
        color: #333;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Custom loading spinner - hide default Streamlit spinners */
    .stSpinner > div {
        display: none !important;
    }
    
    /* Replace with custom circular spinner */
    .stSpinner::before {
        content: '';
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        display: block;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom spinner for manual use */
    .custom-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    return {
        'data_service': DataService(),
        'sentiment_service': SentimentService(),
        'visualizer': ForexVisualizer()
    }

services = initialize_services()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    # Check authentication first
    if not check_password():
        return
    
    # Header with company logo
    logo_base64 = get_logo_base64()
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width: 60px; height: auto;" />
            <div>
                <h1 style="margin: 0;">{get_text("main_title")}</h1>
                <p style="color: white; text-align: center; margin: 0; font-size: 1.1rem;">
                    Previsões Forex com IA e Análise em Tempo Real
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified sidebar configuration
    with st.sidebar:
        st.markdown("## 📊 Análise de Trading")
        
        # Add logout button
        if st.button("🚪 Sair", help="Sair da plataforma"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Currency pair selection
        pair = st.selectbox(
            get_text("sidebar_currency_pair"),
            PAIRS,
            help="Selecione o par de moedas para analisar"
        )
        
        # Time interval
        interval = st.selectbox(
            get_text("sidebar_time_interval"),
            list(INTERVALS.keys()),
            index=4,  # Default to 60min
            help="Período do gráfico"
        )
        
        # Prediction horizon  
        horizon = st.selectbox(
            "Período de Previsão",
            HORIZONS,
            help="Tempo de previsão"
        )
        
        st.markdown("---")
        
        # Analysis buttons - simplified
        analyze_button = st.button("🎯 Obter Sinal de Trading", type="primary", help="Executar análise completa com previsão IA")
        quick_analysis = st.button("⚡ Verificação Rápida", help="Análise rápida com sinais básicos")
        
        st.markdown("---")
        
        # Risk settings - simplified
        risk_level = st.selectbox(
            get_text("sidebar_risk_level"),
            list(RISK_LEVELS.keys()),
            index=1,  # Default to Moderate
            help="Sua tolerância ao risco para dimensionamento de posição"
        )
        
        # Advanced settings - collapsed by default
        with st.expander("Opções Avançadas"):
            lookback_period = st.slider(
                "Histórico de Dados",
                min_value=30,
                max_value=120,
                value=LOOKBACK_PERIOD,
                help="Períodos históricos para treinamento da IA"
            )
            
            epochs = st.slider(
                "Intensidade do Treinamento IA",
                min_value=5,
                max_value=20,
                value=EPOCHS,
                help="Mais épocas = melhor precisão mas mais lento"
            )
            
            mc_samples = st.slider(
                "Amostras de Previsão",
                min_value=10,
                max_value=50,
                value=MC_SAMPLES,
                help="Amostras para estimativa de incerteza"
            )
            
            if st.button("Limpar Cache"):
                CacheManager.clear_cache()
                st.success("Cache limpo!")
                st.rerun()
        
        # Simple status
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        if cache_count > 0:
            st.info(f"💾 {cache_count} análises em cache disponíveis")
    
    # Main content area
    if analyze_button or quick_analysis:
        run_analysis(
            pair, interval, horizon, risk_level, lookback_period, 
            mc_samples, epochs, quick_analysis
        )
    
    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Aviso Legal:</strong> Esta plataforma é apenas para fins educacionais. 
        Trading forex envolve riscos substanciais e pode não ser adequado para todos os investidores.</p>
        <p>Desenvolvido com ❤️ usando Streamlit • Última atualização: {}</p>
    </div>
    """.format(datetime.now().strftime("%d-%m-%Y %H:%M")), unsafe_allow_html=True)

def run_analysis(pair, interval, horizon, risk_level, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis"""
    
    # Custom loading display
    loading_container = st.container()
    with loading_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown("""
            <div class="custom-spinner">
                <div class="spinner"></div>
            </div>
            """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch data
        status_text.text("📥 Buscando dados forex...")
        progress_bar.progress(10)
        
        df = services['data_service'].fetch_forex_data(
            pair, 
            INTERVALS[interval], 
            'full' if not is_quick else 'compact'
        )
        
        if not services['data_service'].validate_data(df):
            st.error("❌ Dados insuficientes ou inválidos recebidos")
            return
        
        progress_bar.progress(25)
        
        # Step 2: Fetch sentiment
        status_text.text("📰 Analisando sentimento do mercado...")
        sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
        sentiment_signal = services['sentiment_service'].get_sentiment_signal(sentiment_score)
        sentiment_strength = services['sentiment_service'].get_sentiment_strength(sentiment_score)
        
        progress_bar.progress(40)
        
        # Step 3: Add technical indicators
        status_text.text("📊 Calculando indicadores técnicos...")
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        trading_signals = TechnicalIndicators.get_trading_signals(df_with_indicators)
        
        progress_bar.progress(60)
        
        # Step 4: Train model and predict
        status_text.text("🧠 Training AI model...")
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
            epochs=epochs if not is_quick else 5,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        progress_bar.progress(80)
        
        # Step 5: Make predictions
        status_text.text("🔮 Generating predictions...")
        steps = HORIZON_STEPS[horizon]
        predictions, uncertainties = predictor.predict_future(
            train_data, 
            sentiment_score, 
            steps, 
            mc_samples
        )
        
        # Calculate confidence
        model_confidence = predictor.get_model_confidence(train_data, sentiment_score)
        
        progress_bar.progress(95)
        
        # Step 6: Store results
        status_text.text("💾 Finalizing analysis...")
        
        current_price = float(df_with_indicators['close'].iloc[-1])
        predicted_price = predictions[-1] if predictions and len(predictions) > 0 else current_price
        
        # Validation: Ensure predicted price is realistic (within 10% of current price for safety)
        max_change = current_price * 0.1  # 10% maximum change as sanity check
        if abs(predicted_price - current_price) > max_change:
            # If prediction seems unrealistic, use a more conservative estimate
            if predicted_price > current_price:
                predicted_price = current_price + (max_change * 0.5)  # 5% increase max
            else:
                predicted_price = current_price - (max_change * 0.5)  # 5% decrease max
        
        # Calculate additional metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Calculate price variation range against the expected trend
        uncertainty = uncertainties[-1] if uncertainties and len(uncertainties) > 0 else 0.0
        
        # Calculate where price could move OPPOSITE to our prediction
        # This shows risk of prediction being completely wrong
        risk_multiplier = 1.5  # How far opposite the prediction could go
        
        if price_change > 0:  # We predict UP (BUY) - show downside risk if wrong
            counter_trend_target = current_price - (abs(price_change) * risk_multiplier)
            risk_direction = "Downside"
            risk_description = "if bullish prediction fails"
        else:  # We predict DOWN (SELL) - show upside risk if wrong
            counter_trend_target = current_price + (abs(price_change) * risk_multiplier)
            risk_direction = "Upside" 
            risk_description = "if bearish prediction fails"
        
        # Debug logging (temporary)
        print(f"DEBUG - Current Price: {current_price:.5f}")
        print(f"DEBUG - Raw Predicted Price: {predictions[-1] if predictions else 'None'}")
        print(f"DEBUG - Final Predicted Price: {predicted_price:.5f}")
        print(f"DEBUG - Price Change: {price_change_pct:.2f}%")
        print(f"DEBUG - Counter-trend Risk: {risk_direction} to {counter_trend_target:.5f} ({risk_description})")
        
        # Risk assessment
        risk_tolerance = RISK_LEVELS[risk_level]
        position_size = min(risk_tolerance, abs(price_change_pct) / 100)
        
        st.session_state.analysis_results = {
            'pair': pair,
            'interval': interval,
            'horizon': horizon,
            'timestamp': datetime.now(),
            'data': df_with_indicators,
            'sentiment': {
                'score': sentiment_score,
                'signal': sentiment_signal,
                'strength': sentiment_strength
            },
            'predictions': predictions,
            'uncertainties': uncertainties,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'model_confidence': model_confidence,
            'training_metrics': training_metrics,
            'trading_signals': trading_signals,
            'risk_assessment': {
                'risk_level': risk_level,
                'position_size': position_size,
                'stop_loss': current_price * (1 - risk_tolerance),
                'take_profit': current_price * (1 + risk_tolerance * 2)
            },
            'counter_trend_risk': {
                'direction': risk_direction,
                'target_price': counter_trend_target,
                'risk_percentage': abs(counter_trend_target - current_price) / current_price * 100,
                'description': risk_description,
                'prediction_direction': 'Bullish' if price_change > 0 else 'Bearish'
            }
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Análise completa!")
        
        # Remove the loading spinner
        spinner_placeholder.empty()
        
        st.markdown("""
        <div class="success-alert">
            <strong>Análise concluída com sucesso!</strong> 
            Verifique os resultados abaixo.
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        # Remove spinner on error too
        spinner_placeholder.empty()
        
        st.markdown(f"""
        <div class="error-alert">
            <strong>Falha na análise:</strong> {str(e)}
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar.empty()
        status_text.empty()

def get_trading_recommendation(results):
    """Calculate overall trading recommendation based on all signals"""
    
    # Validate the price change makes sense
    current_price = results['current_price']
    predicted_price = results['predicted_price']
    price_change_pct = results['price_change_pct']
    
    # Debug logging
    print(f"RECOMMENDATION DEBUG - Current: {current_price:.5f}, Predicted: {predicted_price:.5f}, Change: {price_change_pct:.2f}%")
    
    # Get individual signals
    price_signal = 1 if results['price_change'] > 0 else -1
    sentiment_signal = 1 if results['sentiment']['score'] > 0 else -1 if results['sentiment']['score'] < 0 else 0
    
    # Get technical signals from the most recent data
    signals = results['trading_signals'].tail(1).iloc[0]
    
    # Count bullish/bearish technical signals
    tech_signals = 0
    if signals.get('SMA_Bullish', False):
        tech_signals += 1
    elif signals.get('SMA_Bearish', False):
        tech_signals -= 1
        
    if signals.get('MACD_Bullish', False):
        tech_signals += 1
    elif signals.get('MACD_Bearish', False):
        tech_signals -= 1
        
    if signals.get('RSI_Oversold', False):
        tech_signals += 1
    elif signals.get('RSI_Overbought', False):
        tech_signals -= 1
    
    # Calculate overall score
    overall_score = price_signal + sentiment_signal + (tech_signals / 2)
    
    # Determine recommendation
    if overall_score >= 1.5:
        return "STRONG BUY", "success", "📈"
    elif overall_score >= 0.5:
        return "BUY", "success", "📈"
    elif overall_score <= -1.5:
        return "STRONG SELL", "error", "📉"
    elif overall_score <= -0.5:
        return "SELL", "error", "📉"
    else:
        return "HOLD", "warning", "➡️"

def display_analysis_results():
    """Display simplified analysis results with prominent trading recommendation"""
    results = st.session_state.analysis_results
    
    # Get trading recommendation
    recommendation, rec_type, rec_icon = get_trading_recommendation(results)
    
    # Prominent Trading Recommendation Section
    st.markdown("---")
    
    # Large recommendation display
    if rec_type == "success":
        bg_color = "#d4edda"
        border_color = "#28a745"
    elif rec_type == "error":
        bg_color = "#f8d7da"
        border_color = "#dc3545"
    else:
        bg_color = "#fff3cd"
        border_color = "#ffc107"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 3px solid {border_color};
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    ">
        <h1 style="color: {border_color}; margin: 0; font-size: 3em;">
            {rec_icon} {recommendation}
        </h1>
        <h2 style="color: #333; margin: 10px 0; font-size: 1.5em;">
            {results['pair']} - {results['horizon']} Outlook
        </h2>
        <p style="color: #666; font-size: 1.2em; margin: 0;">
            Expected Price Change: <strong>{results['price_change_pct']:+.2f}%</strong> | 
            Reversal Risk: <strong>{results['counter_trend_risk']['target_price']:.5f}</strong> | 
            Confidence: <strong>{results['model_confidence']:.0%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in a simplified format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"{results['current_price']:.5f}",
            help="Current market price"
        )
    
    with col2:
        st.metric(
            "Target Price",
            f"{results['predicted_price']:.5f}",
            delta=f"{results['price_change']:+.5f}",
            help=f"Predicted price for {results['horizon']}"
        )
    
    with col3:
        sentiment_emoji = "🟢" if results['sentiment']['score'] > 0 else "🔴" if results['sentiment']['score'] < 0 else "🟡"
        st.metric(
            "Market Sentiment",
            f"{sentiment_emoji} {results['sentiment']['signal']}",
            help="Overall market sentiment from news analysis"
        )
    
    # Simplified tabs - only essential information
    tab1, tab2, tab3 = st.tabs([
        "📊 Price & Signals", "📰 Analysis Details", "⚖️ Risk & Settings"
    ])
    
    with tab1:
        # Price chart with key technical signals
        price_chart = services['visualizer'].create_price_chart(
            results['data'].tail(200), 
            results['pair'],
            indicators=True
        )
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Key signals summary in a compact format
        signals = results['trading_signals'].tail(1).iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi_signal = "Oversold" if signals['RSI_Oversold'] else "Overbought" if signals['RSI_Overbought'] else "Neutral"
            rsi_color = "🟢" if signals['RSI_Oversold'] else "🔴" if signals['RSI_Overbought'] else "🟡"
            st.info(f"RSI: {rsi_color} {rsi_signal}")
        
        with col2:
            macd_signal = "Bullish" if signals['MACD_Bullish'] else "Bearish" if signals['MACD_Bearish'] else "Neutral"
            macd_color = "🟢" if signals['MACD_Bullish'] else "🔴" if signals['MACD_Bearish'] else "🟡"
            st.info(f"MACD: {macd_color} {macd_signal}")
        
        with col3:
            sma_signal = "Bullish" if signals['SMA_Bullish'] else "Bearish"
            sma_color = "🟢" if signals['SMA_Bullish'] else "🔴"
            st.info(f"Trend: {sma_color} {sma_signal}")
        
        with col4:
            bb_signal = "Squeeze" if signals['BB_Squeeze'] else "Normal"
            bb_color = "🟠" if signals['BB_Squeeze'] else "🟡"
            st.info(f"Volatility: {bb_color} {bb_signal}")
    
    with tab2:
        # Combined analysis view
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AI Prediction")
            direction = "Bullish" if results['price_change'] > 0 else "Bearish"
            direction_icon = "📈" if results['price_change'] > 0 else "📉"
            confidence_level = "High" if results['model_confidence'] > 0.7 else "Medium" if results['model_confidence'] > 0.4 else "Low"
            
            st.markdown(f"""
            - **Direction:** {direction_icon} {direction}
            - **Confidence:** {confidence_level} ({results['model_confidence']:.0%})
            - **Expected Move:** {results['price_change_pct']:+.2f}%
            - **Target Price:** {results['predicted_price']:.5f}
            """)
            
        with col2:
            st.markdown("#### Risk Analysis")
            trend_direction = "Bullish" if results['price_change'] > 0 else "Bearish"
            counter_risk = results['counter_trend_risk']
            
            # Show risk of prediction being wrong
            prediction_icon = "📈" if counter_risk['prediction_direction'] == 'Bullish' else "📉"
            risk_icon = "📉" if counter_risk['direction'] == 'Downside' else "📈"
            
            st.markdown(f"""
            - **Our Prediction:** {prediction_icon} {counter_risk['prediction_direction']}
            - **Risk if Wrong:** {risk_icon} {counter_risk['direction']} risk
            - **Counter-trend Target:** {counter_risk['target_price']:.5f}
            - **Risk Exposure:** {counter_risk['risk_percentage']:.1f}%
            """)
        
        # Market sentiment section
        st.markdown("#### Market Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_direction = "Positive" if results['sentiment']['score'] > 0 else "Negative" if results['sentiment']['score'] < 0 else "Neutral"
            sentiment_icon = "😊" if results['sentiment']['score'] > 0 else "😟" if results['sentiment']['score'] < 0 else "😐"
            
            st.markdown(f"""
            - **Sentiment:** {sentiment_icon} {sentiment_direction}
            - **Score:** {results['sentiment']['score']:.2f}
            - **Strength:** {results['sentiment']['strength']}
            """)
        
        with col2:
            # Show what happens if the analysis is completely wrong
            counter_risk = results['counter_trend_risk']
            
            st.markdown(f"""
            **If Prediction is Wrong:**
            - Our forecast: {counter_risk['prediction_direction']}
            - Risk direction: {counter_risk['direction']}
            - Target if opposite: {counter_risk['target_price']:.5f}
            - Potential move: {counter_risk['risk_percentage']:.1f}%
            - Risk level: {'HIGH' if counter_risk['risk_percentage'] > 4 else 'MODERATE' if counter_risk['risk_percentage'] > 2 else 'LOW'}
            """)
        
        # Prediction chart
        prediction_chart = services['visualizer'].create_prediction_chart(
            results['data'].tail(100),
            results['predictions'],
            results['uncertainties'],
            results['pair'],
            results['horizon']
        )
        st.plotly_chart(prediction_chart, use_container_width=True)
    
    with tab3:
        # Risk management and position sizing
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Suggested Position")
            risk_data = results['risk_assessment']
            st.markdown(f"""
            - **Risk Level:** {risk_data['risk_level']}
            - **Position Size:** {risk_data['position_size']:.1%} of portfolio
            - **Stop Loss:** {risk_data['stop_loss']:.5f}
            - **Take Profit:** {risk_data['take_profit']:.5f}
            """)
            
            # Market variation warning based on counter-trend risk
            counter_risk = results['counter_trend_risk']
            if counter_risk['risk_percentage'] > 2.0:
                st.warning(f"High risk exposure: {counter_risk['risk_percentage']:.1f}%")
            
            # Show the risk forecast
            counter_risk = results['counter_trend_risk']
            st.markdown(f"""
            **Risk Forecast (Prediction Failure):**
            - Our prediction: {counter_risk['prediction_direction']}
            - Risk direction: {counter_risk['direction']}
            - Target if we're wrong: {counter_risk['target_price']:.5f}
            - Maximum risk exposure: {counter_risk['risk_percentage']:.1f}%
            """)
            
            # Simple risk warnings
            if results['model_confidence'] < 0.5:
                st.warning("Low confidence - reduce position size")
            if abs(results['price_change_pct']) > 2:
                st.warning("Large move predicted - high risk/reward")
        
        with col2:
            st.markdown("#### Analysis Settings")
            st.markdown(f"""
            - **Pair:** {results['pair']}
            - **Timeframe:** {results['interval']}
            - **Prediction Horizon:** {results['horizon']}
            - **Analysis Time:** {results['timestamp'].strftime('%H:%M:%S')}
            """)
            
            # Model performance summary
            st.markdown("#### Model Performance")
            train_loss = results['training_metrics'].get('final_train_loss', 0)
            val_loss = results['training_metrics'].get('final_val_loss', 0)
            st.markdown(f"""
            - **Training Loss:** {train_loss:.4f}
            - **Validation Loss:** {val_loss:.4f}
            - **Model Confidence:** {results['model_confidence']:.0%}
            """)
        mae = results['training_metrics'].get('mae', 0)
        rmse = results['training_metrics'].get('rmse', 0)
        
        if mae < 0.01:
            st.success("✅ Excellent model performance (MAE < 0.01)")
        elif mae < 0.02:
            st.info("ℹ️ Good model performance (MAE < 0.02)")
        else:
            st.warning("⚠️ Model performance could be improved (MAE > 0.02)")

if __name__ == "__main__":
    main()
