import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import custom modules
from config.settings import *
from services.data_service import DataService
from services.sentiment_service import SentimentService
from services.indicators import TechnicalIndicators
from models.lstm_model import ForexPredictor
from utils.visualization import ForexVisualizer
from utils.cache_manager import CacheManager

# Page configuration
st.set_page_config(
    page_title="Advanced Forex Analysis Platform",
    page_icon="📊",
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
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Forex Analysis Platform</h1>
        <p style="color: white; text-align: center; margin: 0; font-size: 1.1rem;">
            AI-Powered Forex Predictions with Real-Time Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 📊 Analysis Configuration")
        
        # Currency pair selection
        pair = st.selectbox(
            "Select Currency Pair",
            PAIRS,
            help="Choose the currency pair to analyze"
        )
        
        # Time interval
        interval = st.selectbox(
            "Select Time Interval",
            list(INTERVALS.keys()),
            index=4,  # Default to 60min
            help="Choose the data interval for analysis"
        )
        
        # Prediction horizon
        horizon = st.selectbox(
            "Prediction Horizon",
            HORIZONS,
            help="Select how far into the future to predict"
        )
        
        # Risk tolerance
        risk_level = st.selectbox(
            "Risk Tolerance",
            list(RISK_LEVELS.keys()),
            index=1,  # Default to Moderate
            help="Your risk tolerance level"
        )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            lookback_period = st.slider(
                "Lookback Period",
                min_value=30,
                max_value=120,
                value=LOOKBACK_PERIOD,
                help="Number of historical periods to use for prediction"
            )
            
            mc_samples = st.slider(
                "Monte Carlo Samples",
                min_value=10,
                max_value=50,
                value=MC_SAMPLES,
                help="Number of samples for uncertainty estimation"
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=5,
                max_value=20,
                value=EPOCHS,
                help="Number of training epochs for the model"
            )
        
        st.markdown("---")
        
        # Action buttons
        analyze_button = st.button("🔍 Run Complete Analysis", type="primary")
        
        col1, col2 = st.columns(2)
        with col1:
            quick_analysis = st.button("⚡ Quick Analysis")
        with col2:
            clear_cache = st.button("🗑️ Clear Cache")
        
        if clear_cache:
            CacheManager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        # Display cache info
        st.markdown("---")
        st.markdown("### 💾 Cache Status")
        cache_count = len([k for k in st.session_state.keys() if isinstance(st.session_state.get(k), tuple)])
        st.info(f"Cached items: {cache_count}")
    
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
        <p>⚠️ <strong>Disclaimer:</strong> This platform is for educational purposes only. 
        Trading forex involves substantial risk and may not be suitable for all investors.</p>
        <p>Built with ❤️ using Streamlit • Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

def run_analysis(pair, interval, horizon, risk_level, lookback_period, mc_samples, epochs, is_quick=False):
    """Run the complete forex analysis"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch data
        status_text.text("📥 Fetching forex data...")
        progress_bar.progress(10)
        
        df = services['data_service'].fetch_forex_data(
            pair, 
            INTERVALS[interval], 
            'full' if not is_quick else 'compact'
        )
        
        if not services['data_service'].validate_data(df):
            st.error("❌ Insufficient or invalid data received")
            return
        
        progress_bar.progress(25)
        
        # Step 2: Fetch sentiment
        status_text.text("📰 Analyzing market sentiment...")
        sentiment_score = services['sentiment_service'].fetch_news_sentiment(pair)
        sentiment_signal = services['sentiment_service'].get_sentiment_signal(sentiment_score)
        sentiment_strength = services['sentiment_service'].get_sentiment_strength(sentiment_score)
        
        progress_bar.progress(40)
        
        # Step 3: Add technical indicators
        status_text.text("📊 Calculating technical indicators...")
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
        predicted_price = predictions[-1] if predictions else current_price
        
        # Calculate additional metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
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
            }
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.markdown("""
        <div class="success-alert">
            <strong>Analysis completed successfully!</strong> 
            Check the results below.
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-alert">
            <strong>Analysis failed:</strong> {str(e)}
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar.empty()
        status_text.empty()

def display_analysis_results():
    """Display comprehensive analysis results"""
    results = st.session_state.analysis_results
    
    # Key metrics header
    st.markdown("## 📈 Analysis Results")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"{results['current_price']:.5f}",
            help="Current market price"
        )
    
    with col2:
        st.metric(
            "Predicted Price",
            f"{results['predicted_price']:.5f}",
            delta=f"{results['price_change']:+.5f}",
            help=f"Prediction for {results['horizon']}"
        )
    
    with col3:
        st.metric(
            "Price Change",
            f"{results['price_change_pct']:+.2f}%",
            help="Expected percentage change"
        )
    
    with col4:
        st.metric(
            "Model Confidence",
            f"{results['model_confidence']:.1%}",
            help="AI model confidence level"
        )
    
    with col5:
        sentiment_color = "🟢" if results['sentiment']['score'] > 0 else "🔴" if results['sentiment']['score'] < 0 else "🟡"
        st.metric(
            "Market Sentiment",
            f"{sentiment_color} {results['sentiment']['signal']}",
            help=f"Sentiment strength: {results['sentiment']['strength']}"
        )
    
    # Main charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Chart", "🔮 Predictions", "📰 Sentiment", "⚖️ Risk Analysis", "🔧 Model Performance"
    ])
    
    with tab1:
        st.markdown("### Technical Analysis Chart")
        price_chart = services['visualizer'].create_price_chart(
            results['data'].tail(200), 
            results['pair'],
            indicators=True
        )
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Trading signals summary
        st.markdown("### 🚦 Trading Signals")
        signals = results['trading_signals']
        recent_signals = signals.tail(1).iloc[0]
        
        signal_cols = st.columns(4)
        with signal_cols[0]:
            rsi_signal = "🟢 Oversold" if recent_signals['RSI_Oversold'] else "🔴 Overbought" if recent_signals['RSI_Overbought'] else "🟡 Neutral"
            st.info(f"RSI: {rsi_signal}")
        
        with signal_cols[1]:
            macd_signal = "🟢 Bullish" if recent_signals['MACD_Bullish'] else "🔴 Bearish" if recent_signals['MACD_Bearish'] else "🟡 Neutral"
            st.info(f"MACD: {macd_signal}")
        
        with signal_cols[2]:
            sma_signal = "🟢 Bullish" if recent_signals['SMA_Bullish'] else "🔴 Bearish"
            st.info(f"SMA: {sma_signal}")
        
        with signal_cols[3]:
            bb_signal = "🟠 Squeeze" if recent_signals['BB_Squeeze'] else "🟡 Normal"
            st.info(f"Bollinger: {bb_signal}")
    
    with tab2:
        st.markdown("### Prediction Analysis")
        prediction_chart = services['visualizer'].create_prediction_chart(
            results['data'].tail(100),
            results['predictions'],
            results['uncertainties'],
            results['pair'],
            results['horizon']
        )
        st.plotly_chart(prediction_chart, use_container_width=True)
        
        # Prediction details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Prediction Summary")
            direction = "📈 **BULLISH**" if results['price_change'] > 0 else "📉 **BEARISH**"
            confidence_level = "High" if results['model_confidence'] > 0.7 else "Medium" if results['model_confidence'] > 0.4 else "Low"
            
            st.markdown(f"""
            - **Direction:** {direction}
            - **Confidence:** {confidence_level} ({results['model_confidence']:.1%})
            - **Expected Move:** {results['price_change_pct']:+.2f}%
            - **Time Horizon:** {results['horizon']}
            """)
        
        with col2:
            st.markdown("#### Key Levels")
            st.markdown(f"""
            - **Current:** {results['current_price']:.5f}
            - **Target:** {results['predicted_price']:.5f}
            - **Support:** {results['current_price'] * 0.995:.5f}
            - **Resistance:** {results['current_price'] * 1.005:.5f}
            """)
    
    with tab3:
        st.markdown("### Market Sentiment Analysis")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sentiment_gauge = services['visualizer'].create_sentiment_gauge(
                results['sentiment']['score'],
                results['sentiment']['signal']
            )
            st.plotly_chart(sentiment_gauge, use_container_width=True)
        
        with col2:
            st.markdown("#### Sentiment Breakdown")
            st.markdown(f"""
            - **Overall Sentiment:** {results['sentiment']['signal']}
            - **Sentiment Score:** {results['sentiment']['score']:.3f}
            - **Strength:** {results['sentiment']['strength']}
            - **Market Bias:** {'Positive' if results['sentiment']['score'] > 0 else 'Negative' if results['sentiment']['score'] < 0 else 'Neutral'}
            """)
            
            if abs(results['sentiment']['score']) > 0.3:
                st.warning("⚠️ Strong sentiment detected. Consider potential volatility.")
            elif abs(results['sentiment']['score']) < 0.1:
                st.info("ℹ️ Neutral sentiment. Technical analysis may be more reliable.")
    
    with tab4:
        st.markdown("### Risk Management")
        
        risk_chart = services['visualizer'].create_risk_metrics_chart(
            results['predictions'],
            results['uncertainties']
        )
        if risk_chart.data:
            st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk assessment
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Position Sizing")
            risk_data = results['risk_assessment']
            st.markdown(f"""
            - **Risk Level:** {risk_data['risk_level']}
            - **Recommended Position:** {risk_data['position_size']:.2%}
            - **Stop Loss:** {risk_data['stop_loss']:.5f}
            - **Take Profit:** {risk_data['take_profit']:.5f}
            """)
        
        with col2:
            st.markdown("#### Risk Warnings")
            if results['model_confidence'] < 0.5:
                st.error("⚠️ Low model confidence. Consider reducing position size.")
            if abs(results['sentiment']['score']) > 0.4:
                st.warning("⚠️ High sentiment volatility expected.")
            if results['price_change_pct'] > 2:
                st.warning("⚠️ Large price movement predicted. High risk/reward.")
            
            st.info("💡 Always use proper risk management and never risk more than you can afford to lose.")
    
    with tab5:
        st.markdown("### Model Performance Metrics")
        
        # Training metrics
        metrics_df = services['visualizer'].create_performance_metrics_table(
            results['training_metrics']
        )
        st.dataframe(metrics_df, use_container_width=True)
        
        # Model details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model Configuration")
            st.markdown(f"""
            - **Model Type:** Enhanced LSTM with Attention
            - **Lookback Period:** {LOOKBACK_PERIOD} periods
            - **Hidden Size:** {HIDDEN_SIZE}
            - **Layers:** {NUM_LAYERS}
            - **Dropout:** {DROPOUT}
            """)
        
        with col2:
            st.markdown("#### Training Details")
            st.markdown(f"""
            - **Epochs Trained:** {results['training_metrics'].get('epochs_trained', 'N/A')}
            - **Final Training Loss:** {results['training_metrics'].get('final_train_loss', 0):.6f}
            - **Validation Loss:** {results['training_metrics'].get('final_val_loss', 0):.6f}
            - **Model Confidence:** {results['model_confidence']:.1%}
            """)
        
        # Performance indicators
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
