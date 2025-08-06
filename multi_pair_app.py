"""
Aplicação Simplificada Multi-Pares Forex para Replit
Sistema otimizado com dados reais e sem dependências complexas
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Configuração da página
st.set_page_config(
    page_title="Forex Multi-Análise",
    page_icon="💱",
    layout="wide"
)

# CSS para design moderno
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f24 100%);
    }
    .metric-card {
        background: linear-gradient(145deg, #1e2328 0%, #2a2f36 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3a4046;
        margin-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #5b8def 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
    }
    .status-bullish { color: #00ff88; }
    .status-bearish { color: #ff4757; }
    .status-neutral { color: #ffa502; }
</style>
""", unsafe_allow_html=True)

class SimpleForexAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
    def fetch_data(self, pair: str) -> pd.DataFrame:
        """Obter dados simples via Alpha Vantage"""
        try:
            url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&outputsize=compact&apikey={self.api_key}'
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'Time Series FX (Daily)' not in data:
                return None
            
            df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            df.columns = ['open', 'high', 'low', 'close']
            
            return df
            
        except Exception:
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcular RSI simples"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calcular EMA simples"""
        try:
            ema = prices.ewm(span=period).mean()
            return ema.iloc[-1]
        except:
            return prices.iloc[-1]
    
    def analyze_pair(self, pair: str) -> dict:
        """Análise completa de um par"""
        data = self.fetch_data(pair)
        
        if data is None or len(data) < 30:
            return None
        
        # Análise técnica básica
        current_price = data['close'].iloc[-1]
        rsi = self.calculate_rsi(data['close'])
        ema_12 = self.calculate_ema(data['close'], 12)
        ema_26 = self.calculate_ema(data['close'], 26)
        
        # Determinar direção
        ema_signal = 'Bullish' if ema_12 > ema_26 else 'Bearish'
        rsi_signal = 'Bullish' if rsi > 50 else 'Bearish' if rsi < 50 else 'Neutral'
        
        # Volatilidade
        volatility = data['close'].pct_change().std() * 100
        
        # Score final simples
        score = 50
        if ema_signal == 'Bullish':
            score += 15
        if rsi > 60:
            score += 10
        elif rsi < 40:
            score -= 10
        
        # Calcular DD estimado
        returns = data['close'].pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min() * 100)
        
        # Win rate estimado
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        return {
            'pair': pair,
            'current_price': current_price,
            'rsi': rsi,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'trend_direction': ema_signal,
            'rsi_signal': rsi_signal,
            'volatility': volatility,
            'score': score,
            'max_drawdown': max_dd,
            'estimated_win_rate': win_rate,
            'recommendation': 'BUY' if score > 65 else 'SELL' if score < 35 else 'HOLD'
        }

def main():
    """Função principal"""
    
    # Header
    st.markdown('<h1 class="main-header">💱 Forex Multi-Análise Simplificada</h1>', unsafe_allow_html=True)
    st.markdown("### Sistema de Análise Multi-Pares com Dados Reais Alpha Vantage")
    
    # Inicializar analyzer
    analyzer = SimpleForexAnalyzer()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configurações")
    
    # Seleção de pares
    selected_pairs = st.sidebar.multiselect(
        "Selecionar Pares",
        analyzer.pairs,
        default=analyzer.pairs,
        help="Escolha os pares para análise"
    )
    
    # Botão de análise
    if st.sidebar.button("🚀 Executar Análise", type="primary"):
        if not selected_pairs:
            st.error("Selecione pelo menos um par para análise")
            return
        
        # Executar análise
        with st.spinner("Analisando pares selecionados..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, pair in enumerate(selected_pairs):
                st.text(f"Analisando {pair}...")
                
                result = analyzer.analyze_pair(pair)
                if result:
                    results.append(result)
                
                # Delay para API
                time.sleep(12)  # Alpha Vantage rate limit
                progress_bar.progress((i + 1) / len(selected_pairs))
            
            progress_bar.progress(1.0)
            st.success(f"Análise concluída! {len(results)} pares analisados.")
            
            # Armazenar resultados
            st.session_state['analysis_results'] = results
    
    # Exibir resultados se disponíveis
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        results = st.session_state['analysis_results']
        
        # Métricas gerais
        st.markdown("## 📊 Resumo Geral")
        
        total_pairs = len(results)
        bullish_count = sum(1 for r in results if r['trend_direction'] == 'Bullish')
        bearish_count = sum(1 for r in results if r['trend_direction'] == 'Bearish')
        avg_win_rate = np.mean([r['estimated_win_rate'] for r in results])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pares", total_pairs)
        with col2:
            st.metric("Bullish", bullish_count, f"{bullish_count/total_pairs*100:.1f}%")
        with col3:
            st.metric("Bearish", bearish_count, f"{bearish_count/total_pairs*100:.1f}%")
        with col4:
            st.metric("Win Rate Médio", f"{avg_win_rate:.1f}%")
        
        # Tabela de resultados
        st.markdown("## 📋 Resultados Detalhados")
        
        # Preparar dados para tabela
        table_data = []
        for result in results:
            table_data.append({
                'Par': result['pair'],
                'Preço': f"{result['current_price']:.4f}",
                'RSI': f"{result['rsi']:.1f}",
                'Direção': result['trend_direction'],
                'Score': f"{result['score']:.0f}",
                'DD Máx (%)': f"{result['max_drawdown']:.1f}",
                'Win Rate (%)': f"{result['estimated_win_rate']:.1f}",
                'Recomendação': result['recommendation']
            })
        
        df = pd.DataFrame(table_data)
        
        # Estilizar tabela
        def color_direction(val):
            if val == 'Bullish':
                return 'background-color: #00ff88; color: white'
            elif val == 'Bearish':
                return 'background-color: #ff4757; color: white'
            else:
                return 'background-color: #ffa502; color: white'
        
        def color_recommendation(val):
            if val == 'BUY':
                return 'background-color: #00ff88; color: white'
            elif val == 'SELL':
                return 'background-color: #ff4757; color: white'
            else:
                return 'background-color: #ffa502; color: white'
        
        styled_df = df.style.applymap(color_direction, subset=['Direção']) \
                           .applymap(color_recommendation, subset=['Recomendação'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Gráfico de distribuição
        st.markdown("## 📈 Análise Visual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras - Direções
            directions = [r['trend_direction'] for r in results]
            direction_counts = pd.Series(directions).value_counts()
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=direction_counts.index,
                    y=direction_counts.values,
                    marker_color=['#00ff88' if x == 'Bullish' else '#ff4757' for x in direction_counts.index],
                    text=direction_counts.values,
                    textposition='auto'
                )
            ])
            
            fig_bar.update_layout(
                title="Distribuição de Tendências",
                xaxis_title="Direção",
                yaxis_title="Quantidade",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Gráfico de dispersão - Score vs Win Rate
            scores = [r['score'] for r in results]
            win_rates = [r['estimated_win_rate'] for r in results]
            pair_names = [r['pair'] for r in results]
            
            fig_scatter = go.Figure(data=go.Scatter(
                x=scores,
                y=win_rates,
                mode='markers+text',
                text=pair_names,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig_scatter.update_layout(
                title="Score vs Win Rate",
                xaxis_title="Score",
                yaxis_title="Win Rate (%)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Alertas
        st.markdown("## 🚨 Alertas e Oportunidades")
        
        high_score_pairs = [r for r in results if r['score'] > 70]
        high_risk_pairs = [r for r in results if r['max_drawdown'] > 15]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Oportunidades (Score > 70)")
            if high_score_pairs:
                for result in high_score_pairs:
                    st.success(f"**{result['pair']}** - {result['trend_direction']}\n"
                              f"Score: {result['score']:.0f} | RSI: {result['rsi']:.1f}")
            else:
                st.info("Nenhuma oportunidade de alto score encontrada")
        
        with col2:
            st.markdown("### ⚠️ Alto Risco (DD > 15%)")
            if high_risk_pairs:
                for result in high_risk_pairs:
                    st.warning(f"**{result['pair']}** - DD: {result['max_drawdown']:.1f}%\n"
                              f"Volatilidade: {result['volatility']:.2f}%")
            else:
                st.info("Nenhum par de alto risco detectado")
    
    else:
        # Instruções iniciais
        st.markdown("## 🚀 Como Usar")
        st.markdown("""
        1. **Selecione os pares** na barra lateral
        2. **Clique em "Executar Análise"** para iniciar
        3. **Aguarde** o processamento (12s por par devido ao rate limit da API)
        4. **Analise os resultados** nas tabelas e gráficos
        
        ### 📊 Métricas Incluídas:
        - **RSI**: Índice de Força Relativa (14 períodos)
        - **EMA**: Médias móveis exponenciais (12 e 26)
        - **Score**: Pontuação técnica combinada (0-100)
        - **DD Máximo**: Drawdown máximo estimado
        - **Win Rate**: Taxa de acerto estimada
        
        ### ⚠️ Importante:
        - Sistema usa dados reais da Alpha Vantage API
        - Rate limit: 5 calls por minuto (delay automático)
        - Requer chave API válida configurada
        """)

if __name__ == "__main__":
    main()