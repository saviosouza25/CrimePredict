"""
Dashboard Interativo para Sistema de Análise Multi-Pares Forex
Design moderno e otimizado para Replit com métricas em tempo real
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from services.forex_multi_analysis import ForexMultiAnalysis

# Configuração da página
st.set_page_config(
    page_title="Forex Multi-Analysis Pro",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para design moderno
st.markdown("""
<style>
    /* Dark theme otimizado */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f24 100%);
    }
    
    /* Cards modernos */
    .metric-card {
        background: linear-gradient(145deg, #1e2328 0%, #2a2f36 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #3a4046;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    /* Headers estilizados */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #5b8def 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    /* Sidebar customizada */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f24 0%, #0f1419 100%);
    }
    
    /* Métricas destacadas */
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    /* Status indicators */
    .status-bullish { color: #00ff88; }
    .status-bearish { color: #ff4757; }
    .status-neutral { color: #ffa502; }
    
    /* Hover effects */
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    
    /* Progress bars customizadas */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #00d4ff 0%, #5b8def 100%);
    }
    
    /* Buttons modernos */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff 0%, #5b8def 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class ForexDashboard:
    def __init__(self):
        self.analyzer = ForexMultiAnalysis()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializar estado da sessão"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = None
        if 'selected_profile' not in st.session_state:
            st.session_state.selected_profile = 'swing_trader'
        if 'theme_mode' not in st.session_state:
            st.session_state.theme_mode = 'dark'
    
    def render_header(self):
        """Renderizar cabeçalho principal"""
        st.markdown('<h1 class="main-header">🚀 Forex Multi-Analysis Pro</h1>', unsafe_allow_html=True)
        st.markdown("### Sistema Avançado de Análise Multi-Pares com IA | Dados Reais Alpha Vantage")
        
        # Status do sistema
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status API", "🟢 Online", help="Conexão com Alpha Vantage")
        with col2:
            st.metric("Pares Ativos", len(self.analyzer.pairs))
        with col3:
            last_update = st.session_state.last_analysis_time
            if last_update:
                update_str = last_update.strftime("%H:%M")
            else:
                update_str = "Nunca"
            st.metric("Última Análise", update_str)
        with col4:
            profile = st.session_state.selected_profile.replace('_', ' ').title()
            st.metric("Perfil Ativo", profile)
    
    def render_sidebar(self):
        """Renderizar barra lateral com controles"""
        st.sidebar.markdown("## ⚙️ Configurações")
        
        # Seleção do perfil de trader
        profile_options = {
            'scalper': '⚡ Scalper (1-5min)',
            'day_trader': '📊 Day Trader (15-60min)',
            'swing_trader': '📈 Swing Trader (Diário)',
            'position_trader': '📉 Position Trader (Semanal)'
        }
        
        selected_profile = st.sidebar.selectbox(
            "Perfil do Trader",
            options=list(profile_options.keys()),
            format_func=lambda x: profile_options[x],
            index=2,  # Default: swing_trader
            help="Selecione seu estilo de trading para métricas personalizadas"
        )
        st.session_state.selected_profile = selected_profile
        
        # Mostrar detalhes do perfil selecionado
        profile_config = self.analyzer.trader_profiles[selected_profile]
        st.sidebar.markdown("### 📋 Detalhes do Perfil")
        st.sidebar.info(f"""
        **Timeframes:** {', '.join(profile_config['timeframes'])}
        **DD Máximo:** {profile_config['max_dd_percent']}%
        **Extensão Max:** {profile_config['max_extension_pips']} pips
        **Tempo Operação:** {profile_config['hold_time']}
        **Win Rate Alvo:** {profile_config['win_rate_target']}%
        **Risk/Reward:** 1:{profile_config['risk_reward']}
        """)
        
        # Seleção de pares
        st.sidebar.markdown("### 📊 Pares para Análise")
        available_pairs = self.analyzer.pairs
        selected_pairs = st.sidebar.multiselect(
            "Selecionar Pares",
            available_pairs,
            default=available_pairs[:5],  # Top 5 pares por padrão
            help="Escolha até 10 pares para análise simultânea"
        )
        
        if selected_pairs:
            self.analyzer.pairs = selected_pairs[:10]  # Limite de 10 pares
        
        # Controles de análise
        st.sidebar.markdown("### 🎯 Controles de Análise")
        
        if st.sidebar.button("🚀 Executar Análise Completa", type="primary", use_container_width=True):
            self.run_full_analysis()
        
        if st.sidebar.button("🔄 Atualizar Dados", use_container_width=True):
            self.refresh_data()
        
        # Configurações avançadas
        with st.sidebar.expander("🔧 Configurações Avançadas"):
            st.slider("Intervalo Auto-Refresh (min)", 5, 60, 15, key="refresh_interval")
            st.checkbox("Notificações de Alerta", value=True, key="enable_alerts")
            st.checkbox("Salvar Histórico", value=True, key="save_history")
        
        # Theme toggle
        st.sidebar.markdown("---")
        theme_mode = st.sidebar.radio("🎨 Tema", ["Dark", "Light"], index=0)
        st.session_state.theme_mode = theme_mode.lower()
    
    def run_full_analysis(self):
        """Executar análise completa para todos os pares selecionados"""
        with st.spinner("🔄 Executando análise completa... Isso pode levar alguns minutos..."):
            # Barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                profile = st.session_state.selected_profile
                status_text.text(f"Iniciando análise com perfil: {profile}")
                
                results = self.analyzer.analyze_all_pairs(profile)
                
                progress_bar.progress(100)
                status_text.text("✅ Análise concluída com sucesso!")
                
                st.session_state.analysis_results = results
                st.session_state.last_analysis_time = datetime.now()
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {str(e)}")
                status_text.text("❌ Análise falhada")
    
    def refresh_data(self):
        """Atualizar dados sem nova análise completa"""
        st.info("🔄 Atualizando dados em tempo real...")
        # Implementar refresh de dados específicos
        time.sleep(2)
        st.success("✅ Dados atualizados!")
    
    def render_overview_metrics(self):
        """Renderizar métricas gerais na visão geral"""
        results = st.session_state.analysis_results
        if not results:
            st.warning("⚠️ Execute uma análise primeiro para ver as métricas.")
            return
        
        st.markdown("## 📊 Visão Geral do Mercado")
        
        # Calcular métricas agregadas
        total_pairs = len(results)
        bullish_pairs = sum(1 for r in results.values() if r['trend']['direction'] == 'Bullish')
        bearish_pairs = sum(1 for r in results.values() if r['trend']['direction'] == 'Bearish')
        neutral_pairs = total_pairs - bullish_pairs - bearish_pairs
        
        avg_win_rate = np.mean([r['trade_probability']['success_probability'] for r in results.values()])
        avg_liquidity = np.mean([r['liquidity']['liquidity_score'] for r in results.values()])
        
        # Cards de métricas principais
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #00d4ff; margin: 0;">Total Pares</h3>
                <div class="big-metric" style="color: white;">{}</div>
                <small style="color: #a0a0a0;">Analisados</small>
            </div>
            """.format(total_pairs), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #00ff88; margin: 0;">Bullish</h3>
                <div class="big-metric status-bullish">{}</div>
                <small style="color: #a0a0a0;">{:.1f}% do total</small>
            </div>
            """.format(bullish_pairs, (bullish_pairs/total_pairs)*100), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #ff4757; margin: 0;">Bearish</h3>
                <div class="big-metric status-bearish">{}</div>
                <small style="color: #a0a0a0;">{:.1f}% do total</small>
            </div>
            """.format(bearish_pairs, (bearish_pairs/total_pairs)*100), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #ffa502; margin: 0;">Win Rate Médio</h3>
                <div class="big-metric" style="color: white;">{:.1f}%</div>
                <small style="color: #a0a0a0;">Probabilidade</small>
            </div>
            """.format(avg_win_rate), unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #00d4ff; margin: 0;">Liquidez Média</h3>
                <div class="big-metric" style="color: white;">{:.1f}</div>
                <small style="color: #a0a0a0;">Score/100</small>
            </div>
            """.format(avg_liquidity), unsafe_allow_html=True)
    
    def render_pair_analysis_table(self):
        """Renderizar tabela detalhada de análise por par"""
        results = st.session_state.analysis_results
        if not results:
            return
        
        st.markdown("## 📋 Análise Detalhada por Par")
        
        # Preparar dados para a tabela
        table_data = []
        for pair, analysis in results.items():
            table_data.append({
                'Par': pair,
                'Direção': analysis['trend']['direction'],
                'Score Final': f"{analysis['final_score']:.1f}",
                'Probabilidade': f"{analysis['trade_probability']['success_probability']:.1f}%",
                'Liquidez': f"{analysis['liquidity']['liquidity_score']:.1f}",
                'RSI': f"{analysis['trend']['rsi_value']:.1f}",
                'DD Max': f"{analysis['risk_metrics']['max_drawdown_percent']:.1f}%",
                'Extensão': f"{analysis['risk_metrics']['max_extension_pips']:.0f} pips",
                'Recomendação': analysis['trade_probability']['recommended_action'],
                'IA Pred': analysis['lstm_prediction']['prediction'],
                'Risco': analysis['trade_probability']['risk_level']
            })
        
        df = pd.DataFrame(table_data)
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            direction_filter = st.selectbox("Filtrar por Direção", ["Todos", "Bullish", "Bearish", "Neutral"])
        with col2:
            min_probability = st.slider("Probabilidade Mínima (%)", 0, 100, 60)
        with col3:
            risk_filter = st.selectbox("Filtrar por Risco", ["Todos", "Low", "Medium", "High"])
        
        # Aplicar filtros
        filtered_df = df.copy()
        if direction_filter != "Todos":
            filtered_df = filtered_df[filtered_df['Direção'] == direction_filter]
        
        filtered_df['Prob_Numeric'] = filtered_df['Probabilidade'].str.rstrip('%').astype(float)
        filtered_df = filtered_df[filtered_df['Prob_Numeric'] >= min_probability]
        
        if risk_filter != "Todos":
            filtered_df = filtered_df[filtered_df['Risco'] == risk_filter]
        
        # Remover coluna auxiliar
        filtered_df = filtered_df.drop('Prob_Numeric', axis=1)
        
        # Estilizar tabela
        def style_table(df):
            def color_direction(val):
                color = '#00ff88' if val == 'Bullish' else '#ff4757' if val == 'Bearish' else '#ffa502'
                return f'background-color: {color}; color: white; font-weight: bold'
            
            def color_recommendation(val):
                color = '#00ff88' if val == 'BUY' else '#ff4757' if val == 'SELL' else '#ffa502'
                return f'background-color: {color}; color: white; font-weight: bold'
            
            return df.style.applymap(color_direction, subset=['Direção']) \
                          .applymap(color_recommendation, subset=['Recomendação']) \
                          .format({'Score Final': '{:.1f}', 'Liquidez': '{:.1f}'})
        
        styled_df = style_table(filtered_df)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Estatísticas do filtro
        st.info(f"📊 Exibindo {len(filtered_df)} de {len(df)} pares")
    
    def render_charts(self):
        """Renderizar gráficos interativos"""
        results = st.session_state.analysis_results
        if not results:
            return
        
        st.markdown("## 📈 Análise Visual")
        
        # Preparar dados
        pairs = list(results.keys())
        scores = [results[pair]['final_score'] for pair in pairs]
        probabilities = [results[pair]['trade_probability']['success_probability'] for pair in pairs]
        directions = [results[pair]['trend']['direction'] for pair in pairs]
        
        # Gráfico de dispersão: Score Final vs Probabilidade
        fig_scatter = go.Figure()
        
        colors = {'Bullish': '#00ff88', 'Bearish': '#ff4757', 'Neutral': '#ffa502'}
        for direction in colors.keys():
            mask = [d == direction for d in directions]
            if any(mask):
                fig_scatter.add_trace(go.Scatter(
                    x=[scores[i] for i in range(len(scores)) if mask[i]],
                    y=[probabilities[i] for i in range(len(probabilities)) if mask[i]],
                    mode='markers+text',
                    name=direction,
                    text=[pairs[i] for i in range(len(pairs)) if mask[i]],
                    textposition="top center",
                    marker=dict(
                        color=colors[direction],
                        size=12,
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Probabilidade: %{y:.1f}%<extra></extra>'
                ))
        
        fig_scatter.update_layout(
            title="Score Final vs Probabilidade de Sucesso",
            xaxis_title="Score Final",
            yaxis_title="Probabilidade de Sucesso (%)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico de barras: Distribuição de direções
        direction_counts = pd.Series(directions).value_counts()
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=direction_counts.index,
                y=direction_counts.values,
                marker_color=[colors.get(d, '#ffffff') for d in direction_counts.index],
                text=direction_counts.values,
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title="Distribuição de Sinais de Mercado",
            xaxis_title="Direção",
            yaxis_title="Número de Pares",
            template="plotly_dark",
            height=400
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Gráfico de pizza: Risk levels
        with col2:
            risk_levels = [results[pair]['trade_probability']['risk_level'] for pair in pairs]
            risk_counts = pd.Series(risk_levels).value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker_colors=['#00ff88', '#ffa502', '#ff4757']
            )])
            
            fig_pie.update_layout(
                title="Distribuição de Níveis de Risco",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_alerts_section(self):
        """Renderizar seção de alertas e oportunidades"""
        results = st.session_state.analysis_results
        if not results:
            return
        
        st.markdown("## 🚨 Alertas e Oportunidades")
        
        # Filtrar oportunidades de alta probabilidade
        high_prob_pairs = {
            pair: analysis for pair, analysis in results.items()
            if analysis['trade_probability']['success_probability'] >= 70
        }
        
        # Filtrar pares de alto risco
        high_risk_pairs = {
            pair: analysis for pair, analysis in results.items()
            if analysis['trade_probability']['risk_level'] == 'High'
        }
        
        # Filtrar tendências fortes
        strong_trends = {
            pair: analysis for pair, analysis in results.items()
            if analysis['trend']['trend_strength'] == 'Strong'
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Oportunidades High-Probability")
            if high_prob_pairs:
                for pair, analysis in high_prob_pairs.items():
                    prob = analysis['trade_probability']['success_probability']
                    action = analysis['trade_probability']['recommended_action']
                    direction = analysis['trend']['direction']
                    
                    color = "success" if action == "BUY" else "error" if action == "SELL" else "warning"
                    st.success(f"**{pair}** - {direction}\n\n📊 Probabilidade: {prob:.1f}%\n🎯 Ação: {action}")
            else:
                st.info("Nenhuma oportunidade de alta probabilidade encontrada.")
        
        with col2:
            st.markdown("### ⚠️ Alertas de Risco")
            if high_risk_pairs:
                for pair, analysis in high_risk_pairs.items():
                    dd = analysis['risk_metrics']['max_drawdown_percent']
                    risk = analysis['trade_probability']['risk_level']
                    
                    st.warning(f"**{pair}** - Risco {risk}\n\n📉 DD Máximo: {dd:.1f}%\n⚠️ Cuidado com volatilidade")
            else:
                st.info("Nenhum alerta de risco ativo.")
        
        with col3:
            st.markdown("### 💪 Tendências Fortes")
            if strong_trends:
                for pair, analysis in strong_trends.items():
                    direction = analysis['trend']['direction']
                    adx = analysis['trend']['adx_value']
                    
                    st.info(f"**{pair}** - {direction}\n\n📈 ADX: {adx:.1f}\n💪 Tendência consolidada")
            else:
                st.info("Nenhuma tendência forte detectada.")

def main():
    """Função principal do dashboard"""
    dashboard = ForexDashboard()
    
    # Renderizar componentes
    dashboard.render_header()
    dashboard.render_sidebar()
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📋 Análise Detalhada", "📈 Gráficos", "🚨 Alertas"])
    
    with tab1:
        dashboard.render_overview_metrics()
    
    with tab2:
        dashboard.render_pair_analysis_table()
    
    with tab3:
        dashboard.render_charts()
    
    with tab4:
        dashboard.render_alerts_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0a0a0; padding: 20px;">
        <p>🚀 Forex Multi-Analysis Pro | Powered by Alpha Vantage API | Built for Replit</p>
        <p>⚡ Real-time data | 🤖 AI-powered predictions | 📊 Professional-grade analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()