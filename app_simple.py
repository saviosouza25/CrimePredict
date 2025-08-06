"""
App Principal Simplificado para Análise Forex
Versão funcional otimizada para Replit sem dependências complexas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import os

# Configuração da página
st.set_page_config(
    page_title="Forex Analysis Platform",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para design
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f24 100%);
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #5b8def 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e2328 0%, #2a2f36 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3a4046;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class SimpleForexAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.pairs = {
            'EUR/USD': 'EURUSD',
            'GBP/USD': 'GBPUSD', 
            'USD/JPY': 'USDJPY',
            'AUD/USD': 'AUDUSD',
            'USD/CAD': 'USDCAD',
            'NZD/USD': 'NZDUSD',
            'USD/CHF': 'USDCHF'
        }
    
    def fetch_forex_data(self, pair_symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Buscar dados forex da Alpha Vantage"""
        try:
            if interval == 'daily':
                function = 'FX_DAILY'
                time_key = 'Time Series FX (Daily)'
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={pair_symbol[:3]}&to_symbol={pair_symbol[3:]}&outputsize=compact&apikey={self.api_key}'
            else:
                function = 'FX_INTRADAY'
                time_key = f'Time Series FX ({interval})'
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={pair_symbol[:3]}&to_symbol={pair_symbol[3:]}&interval={interval}&outputsize=compact&apikey={self.api_key}'
            
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if time_key not in data:
                return None
            
            df = pd.DataFrame.from_dict(data[time_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            df.columns = ['open', 'high', 'low', 'close']
            
            return df
            
        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular EMA"""
        return prices.ewm(span=period).mean()
    
    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calcular MACD"""
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = self.calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def analyze_pair(self, pair: str, interval: str = 'daily') -> dict:
        """Análise completa de um par"""
        pair_symbol = self.pairs.get(pair, pair.replace('/', ''))
        data = self.fetch_forex_data(pair_symbol, interval)
        
        if data is None or len(data) < 30:
            return None
        
        # Calcular indicadores
        rsi = self.calculate_rsi(data['close'])
        ema_12 = self.calculate_ema(data['close'], 12)
        ema_26 = self.calculate_ema(data['close'], 26)
        macd_line, signal_line, histogram = self.calculate_macd(data['close'])
        
        # Valores atuais
        current_price = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_ema_12 = ema_12.iloc[-1]
        current_ema_26 = ema_26.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        # Análise de tendência
        ema_trend = 'Bullish' if current_ema_12 > current_ema_26 else 'Bearish'
        rsi_condition = 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'
        macd_trend = 'Bullish' if current_macd > current_signal else 'Bearish'
        
        # Score combinado
        score = 50
        if ema_trend == 'Bullish':
            score += 15
        if rsi_condition == 'Neutral' and current_rsi > 50:
            score += 10
        elif rsi_condition == 'Oversold':
            score += 20
        elif rsi_condition == 'Overbought':
            score -= 10
        if macd_trend == 'Bullish':
            score += 15
        
        # Volatilidade
        volatility = data['close'].pct_change().std() * 100
        
        # Recomendação
        if score > 70:
            recommendation = 'BUY'
        elif score < 30:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        return {
            'pair': pair,
            'current_price': current_price,
            'rsi': current_rsi,
            'ema_12': current_ema_12,
            'ema_26': current_ema_26,
            'macd': current_macd,
            'signal': current_signal,
            'ema_trend': ema_trend,
            'rsi_condition': rsi_condition,
            'macd_trend': macd_trend,
            'score': score,
            'volatility': volatility,
            'recommendation': recommendation,
            'data': data,
            'rsi_series': rsi,
            'ema_12_series': ema_12,
            'ema_26_series': ema_26,
            'macd_series': macd_line,
            'signal_series': signal_line
        }

def create_charts(analysis_data):
    """Criar gráficos para análise"""
    data = analysis_data['data']
    
    # Gráfico principal com candlestick
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Preço', 'RSI', 'MACD'),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Preço'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=analysis_data['ema_12_series'],
            name='EMA 12',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=analysis_data['ema_26_series'],
            name='EMA 26',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=analysis_data['rsi_series'],
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # Linhas RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=analysis_data['macd_series'],
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=analysis_data['signal_series'],
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"Análise Técnica - {analysis_data['pair']}",
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    """Função principal"""
    
    # Header
    st.markdown('<h1 class="main-header">💱 Plataforma de Análise Forex</h1>', unsafe_allow_html=True)
    st.markdown("### Sistema Simplificado com Dados Reais Alpha Vantage")
    
    # Verificar API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
    if api_key == 'demo':
        st.warning("⚠️ Usando API key demo. Configure ALPHA_VANTAGE_API_KEY para dados reais.")
    
    # Inicializar analyzer
    analyzer = SimpleForexAnalyzer()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configurações")
    
    # Seleção de par
    selected_pair = st.sidebar.selectbox(
        "Selecionar Par",
        options=list(analyzer.pairs.keys()),
        index=0
    )
    
    # Seleção de timeframe
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=['daily', '60min', '30min', '15min', '5min'],
        index=0
    )
    
    # Botão de análise
    if st.sidebar.button("🚀 Executar Análise", type="primary"):
        with st.spinner(f"Analisando {selected_pair}..."):
            analysis = analyzer.analyze_pair(selected_pair, timeframe)
            
            if analysis:
                st.session_state['current_analysis'] = analysis
                st.success("Análise concluída!")
                st.rerun()
            else:
                st.error("Falha ao obter dados. Verifique a conexão e API key.")
    
    # Sistema Multi-Pares
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Sistema Multi-Pares")
    if st.sidebar.button("🚀 Abrir Multi-Pares", type="secondary"):
        st.info("**Sistema Multi-Pares está rodando na porta 5001**")
        st.markdown("Acesse: http://localhost:5001")
    
    # Exibir análise se disponível
    if 'current_analysis' in st.session_state and st.session_state['current_analysis']:
        analysis = st.session_state['current_analysis']
        
        # Métricas principais
        st.markdown("## 📊 Análise Principal")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Preço Atual", f"{analysis['current_price']:.4f}")
        
        with col2:
            rsi_delta = f"{analysis['rsi']:.1f}"
            rsi_color = "normal" if 30 <= analysis['rsi'] <= 70 else "inverse"
            st.metric("RSI", rsi_delta, delta_color=rsi_color)
        
        with col3:
            trend_icon = "📈" if analysis['ema_trend'] == 'Bullish' else "📉"
            st.metric("Tendência EMA", f"{trend_icon} {analysis['ema_trend']}")
        
        with col4:
            macd_icon = "🟢" if analysis['macd_trend'] == 'Bullish' else "🔴"
            st.metric("MACD", f"{macd_icon} {analysis['macd_trend']}")
        
        with col5:
            score_color = "normal" if analysis['score'] >= 50 else "inverse"
            st.metric("Score", f"{analysis['score']:.0f}/100", delta_color=score_color)
        
        # Recomendação
        rec = analysis['recommendation']
        if rec == 'BUY':
            st.success(f"🟢 **RECOMENDAÇÃO: {rec}**")
        elif rec == 'SELL':
            st.error(f"🔴 **RECOMENDAÇÃO: {rec}**")
        else:
            st.warning(f"🟡 **RECOMENDAÇÃO: {rec}**")
        
        # Gráficos
        st.markdown("## 📈 Gráficos Técnicos")
        fig = create_charts(analysis)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detalhes técnicos
        st.markdown("## 📋 Detalhes Técnicos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Indicadores Técnicos:**
            - RSI (14): {analysis['rsi']:.2f} - {analysis['rsi_condition']}
            - EMA 12: {analysis['ema_12']:.4f}
            - EMA 26: {analysis['ema_26']:.4f}
            - Tendência EMA: {analysis['ema_trend']}
            """)
        
        with col2:
            st.markdown(f"""
            **MACD:**
            - MACD Line: {analysis['macd']:.6f}
            - Signal Line: {analysis['signal']:.6f}
            - Tendência: {analysis['macd_trend']}
            - Volatilidade: {analysis['volatility']:.3f}%
            """)
    
    else:
        # Instruções
        st.markdown("## 🚀 Como Usar")
        st.markdown("""
        1. **Selecione um par de moedas** na barra lateral
        2. **Escolha o timeframe** para análise
        3. **Clique em "Executar Análise"** 
        4. **Analise os resultados** e gráficos técnicos
        
        ### 📊 Indicadores Incluídos:
        - **RSI (14)**: Índice de Força Relativa
        - **EMA (12/26)**: Médias Móveis Exponenciais
        - **MACD**: Convergência/Divergência de Médias
        - **Score Técnico**: Combinação ponderada dos indicadores
        
        ### 🌍 Sistema Multi-Pares:
        Para análise simultânea de múltiplos pares, use o Sistema Multi-Pares na porta 5001.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        💱 Plataforma Forex | Dados Alpha Vantage | Análise Técnica Profissional
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()