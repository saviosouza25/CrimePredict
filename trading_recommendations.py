import streamlit as st

def display_trading_recommendations(results):
    """Exibe recomendações detalhadas de trading para os melhores pares"""
    
    if not results:
        st.warning("Nenhuma recomendação disponível.")
        return
    
    st.markdown("### 💼 Recomendações Profissionais de Trading")
    st.markdown("**Baseado em:** Análise multi-timeframe + Liquidez real + Sentimento + IA/LSTM")
    
    # Filtrar apenas os melhores 10 pares com recomendações
    top_pairs = []
    for result in results[:15]:
        trading_rec = result.get('trading_recommendation', {})
        if trading_rec.get('action') != 'AGUARDAR':
            top_pairs.append(result)
    
    if not top_pairs:
        st.info("📊 Nenhuma oportunidade de trading identificada no momento. Todos os pares estão em status 'AGUARDAR'.")
        return
    
    st.success(f"🎯 {len(top_pairs)} oportunidades de trading identificadas!")
    
    # Exibir recomendações detalhadas
    for i, result in enumerate(top_pairs[:10]):  # Top 10 oportunidades
        pair = result['pair']
        current_price = result['current_price']
        trading_rec = result['trading_recommendation']
        overall_analysis = result.get('overall_analysis', {})
        
        # Cores baseadas na ação
        action = trading_rec['action']
        if action == 'COMPRAR':
            color = "#00C851"
            icon = "📈"
            action_text = "COMPRA"
        elif action == 'VENDER':
            color = "#F44336"
            icon = "📉"
            action_text = "VENDA"
        else:
            color = "#FF9800"
            icon = "⏸️"
            action_text = "AGUARDAR"
        
        # Confiança da recomendação
        confidence = trading_rec['confidence']
        confidence_color = "#00C851" if confidence == 'Alta' else "#FF9800" if confidence == 'Média' else "#F44336"
        
        # Exibir recomendação em card
        with st.container():
            st.markdown(f"""
            <div style="
                border: 2px solid {color}; 
                border-radius: 12px; 
                padding: 1.5rem; 
                margin: 1rem 0;
                background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: {color};">#{i+1} {pair} {icon}</h3>
                    <div style="text-align: right;">
                        <div style="
                            background: {color}; 
                            color: white; 
                            padding: 0.5rem 1rem; 
                            border-radius: 25px; 
                            font-weight: bold;
                            font-size: 1.1rem;
                        ">
                            {action_text}
                        </div>
                        <div style="
                            color: {confidence_color}; 
                            font-weight: bold; 
                            margin-top: 0.3rem;
                            font-size: 0.9rem;
                        ">
                            Confiança: {confidence}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalhes da recomendação em colunas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Detalhes Financeiros**")
                st.write(f"• **Preço Atual:** {current_price:.5f}")
                st.write(f"• **Probabilidade:** {trading_rec['probability']:.1f}%")
                st.write(f"• **Gestão de Risco:** {trading_rec['risk_level']}")
                
            with col2:
                st.markdown("**⏰ Timing e Execução**")
                st.write(f"• **Timing:** {trading_rec['timing']}")
                st.write(f"• **Força Técnica:** {trading_rec['technical_strength']}")
                st.write(f"• **Liquidez:** {trading_rec['liquidity_impact']}")
                
            with col3:
                st.markdown("**✅ Fatores de Confirmação**")
                confirmations = trading_rec.get('confirmations', [])
                if confirmations:
                    for conf in confirmations[:3]:  # Máximo 3
                        st.write(f"• {conf}")
                else:
                    st.write("• Sinais limitados")
            
            # Alertas importantes
            alerts = trading_rec.get('alerts', [])
            if alerts:
                st.markdown("**⚠️ Alertas Importantes:**")
                for alert in alerts:
                    st.warning(alert)
            
            # Análise do consenso multi-timeframe
            with st.expander(f"📊 Análise Multi-Timeframe - {pair}"):
                timeframe_analysis = result.get('timeframe_analysis', {})
                
                tf_cols = st.columns(4)
                for idx, (tf_name, col) in enumerate(zip(['M5', 'M15', 'H1', 'D1'], tf_cols)):
                    if tf_name in timeframe_analysis:
                        tf_data = timeframe_analysis[tf_name]
                        signal = tf_data.get('trend_signal', 'NEUTRO')
                        prob = tf_data.get('probability', 50)
                        
                        # Cor do sinal
                        if 'COMPRA' in signal:
                            tf_color = "#00C851"
                            tf_icon = "🟢"
                        elif 'VENDA' in signal:
                            tf_color = "#F44336"
                            tf_icon = "🔴"
                        else:
                            tf_color = "#808080"
                            tf_icon = "⚫"
                        
                        with col:
                            st.markdown(f"""
                            <div style="
                                text-align: center; 
                                padding: 1rem; 
                                border: 1px solid {tf_color}; 
                                border-radius: 8px;
                                background: rgba(255,255,255,0.8);
                            ">
                                <h4 style="margin: 0; color: {tf_color};">{tf_name}</h4>
                                <div style="font-size: 2rem; margin: 0.5rem 0;">{tf_icon}</div>
                                <div style="color: {tf_color}; font-weight: bold;">{signal}</div>
                                <div style="color: #666; font-size: 0.9rem;">{prob:.0f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with col:
                            st.markdown(f"""
                            <div style="
                                text-align: center; 
                                padding: 1rem; 
                                border: 1px solid #ddd; 
                                border-radius: 8px;
                                background: rgba(248,248,248,0.8);
                            ">
                                <h4 style="margin: 0; color: #999;">{tf_name}</h4>
                                <div style="font-size: 2rem; margin: 0.5rem 0;">❌</div>
                                <div style="color: #999;">Sem dados</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Resumo estratégico
    st.markdown("### 📋 Resumo Estratégico")
    
    # Contar tipos de recomendações
    buy_count = sum(1 for r in top_pairs if r['trading_recommendation']['action'] == 'COMPRAR')
    sell_count = sum(1 for r in top_pairs if r['trading_recommendation']['action'] == 'VENDER')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 Oportunidades de Compra", buy_count)
    
    with col2:
        st.metric("🔴 Oportunidades de Venda", sell_count)
    
    with col3:
        high_confidence = sum(1 for r in top_pairs if r['trading_recommendation']['confidence'] == 'Alta')
        st.metric("⭐ Alta Confiança", high_confidence)
    
    # Recomendações gerais
    st.markdown("### 💡 Estratégia Recomendada")
    
    if buy_count > sell_count * 1.5:
        market_bias = "COMPRADOR"
        market_color = "#00C851"
        strategy = "Foque em posições de compra com boa gestão de risco. Mercado apresenta viés positivo."
    elif sell_count > buy_count * 1.5:
        market_bias = "VENDEDOR"
        market_color = "#F44336"
        strategy = "Considere posições de venda com stops ajustados. Mercado apresenta viés negativo."
    else:
        market_bias = "NEUTRO"
        market_color = "#FF9800"
        strategy = "Mercado equilibrado. Opere apenas oportunidades de alta confiança."
    
    st.markdown(f"""
    <div style="
        padding: 1rem; 
        border-radius: 8px; 
        background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
        border-left: 4px solid {market_color};
    ">
        <h4 style="color: {market_color}; margin: 0;">Viés de Mercado: {market_bias}</h4>
        <p style="margin: 0.5rem 0; color: #444;">{strategy}</p>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            <strong>Lembre-se:</strong> Sempre use stop loss, gerencie o risco e não arrisque mais que 2-3% da banca por operação.
        </p>
    </div>
    """, unsafe_allow_html=True)