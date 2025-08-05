#!/usr/bin/env python3
"""
🔧 DEBUG EXATO: Comparação AI/LSTM bit-a-bit
"""

import streamlit as st
import pandas as pd
import numpy as np

def debug_ai_exact_values(df_with_indicators, current_price):
    """Debug exato dos valores AI/LSTM"""
    
    st.markdown("### 🔍 DEBUG EXATO: AI/LSTM")
    
    lookback_period = 20
    epochs = 50
    
    # === ANÁLISE INDIVIDUAL ===
    st.markdown("#### 📊 INDIVIDUAL")
    
    recent_prices_ind = df_with_indicators['close'].tail(lookback_period).values
    risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
    
    st.write(f"**Array recent_prices (len={len(recent_prices_ind)}):** {recent_prices_ind}")
    
    short_trend_ind = (recent_prices_ind[-1] - recent_prices_ind[-5]) / recent_prices_ind[-5] if len(recent_prices_ind) >= 5 else 0
    long_trend_ind = (recent_prices_ind[-1] - recent_prices_ind[0]) / recent_prices_ind[0]
    volatility_ind = np.std(recent_prices_ind) / np.mean(recent_prices_ind)
    
    st.write(f"**Short trend:** {short_trend_ind:.10f}")
    st.write(f"**Long trend:** {long_trend_ind:.10f}")
    st.write(f"**Volatility:** {volatility_ind:.10f}")
    
    base_learning_factor = min(1.0, epochs / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    st.write(f"**Learning factor:** {learning_factor:.10f}")
    
    trend_signal_ind = np.tanh(long_trend_ind * 10) * 0.020 * risk_config['signal_damping']
    momentum_signal_ind = np.tanh(short_trend_ind * 15) * 0.015 * risk_config['signal_damping']
    volatility_signal_ind = (0.02 - volatility_ind) * 0.010
    
    st.write(f"**Trend signal:** {trend_signal_ind:.10f}")
    st.write(f"**Momentum signal:** {momentum_signal_ind:.10f}")
    st.write(f"**Volatility signal (antes):** {volatility_signal_ind:.10f}")
    
    if volatility_ind > 0.015:
        volatility_signal_ind *= 0.8
        st.write(f"**Volatility signal (após):** {volatility_signal_ind:.10f}")
    
    combined_signal_ind = (trend_signal_ind * 0.5 + momentum_signal_ind * 0.3 + volatility_signal_ind * 0.2) * learning_factor
    
    st.write(f"**Combined signal:** {combined_signal_ind:.10f}")
    st.write(f"**Direção:** {'COMPRA' if combined_signal_ind > 0.001 else 'VENDA' if combined_signal_ind < -0.001 else 'NEUTRO'}")
    
    # === ANÁLISE UNIFICADA ===
    st.markdown("#### 🧠 UNIFICADA")
    
    prices = df_with_indicators['close'].values
    recent_prices_uni = prices[-lookback_period:] if len(prices) >= lookback_period else prices
    
    st.write(f"**Array prices total (len={len(prices)}):** {prices}")
    st.write(f"**Array recent_prices_unified (len={len(recent_prices_uni)}):** {recent_prices_uni}")
    st.write(f"**Arrays iguais?** {np.array_equal(recent_prices_ind, recent_prices_uni)}")
    
    if len(prices) >= lookback_period:
        short_trend_uni = (recent_prices_uni[-1] - recent_prices_uni[-5]) / recent_prices_uni[-5] if len(recent_prices_uni) >= 5 else 0
        long_trend_uni = (recent_prices_uni[-1] - recent_prices_uni[0]) / recent_prices_uni[0]
        volatility_ai_uni = np.std(recent_prices_uni) / np.mean(recent_prices_uni)
        
        st.write(f"**Short trend:** {short_trend_uni:.10f}")
        st.write(f"**Long trend:** {long_trend_uni:.10f}")
        st.write(f"**Volatility:** {volatility_ai_uni:.10f}")
        
        learning_factor_uni = learning_factor  # Mesmo valor
        
        trend_signal_uni = np.tanh(long_trend_uni * 10) * 0.020 * risk_config['signal_damping']
        momentum_signal_uni = np.tanh(short_trend_uni * 15) * 0.015 * risk_config['signal_damping']
        volatility_signal_uni = (0.02 - volatility_ai_uni) * 0.010
        
        st.write(f"**Trend signal:** {trend_signal_uni:.10f}")
        st.write(f"**Momentum signal:** {momentum_signal_uni:.10f}")
        st.write(f"**Volatility signal (antes):** {volatility_signal_uni:.10f}")
        
        if volatility_ai_uni > 0.015:
            volatility_signal_uni *= 0.8
            st.write(f"**Volatility signal (após):** {volatility_signal_uni:.10f}")
        
        lstm_signal_uni = (trend_signal_uni * 0.5 + momentum_signal_uni * 0.3 + volatility_signal_uni * 0.2) * learning_factor_uni
        
        st.write(f"**LSTM signal:** {lstm_signal_uni:.10f}")
        st.write(f"**Direção:** {'COMPRA' if lstm_signal_uni > 0.001 else 'VENDA' if lstm_signal_uni < -0.001 else 'NEUTRO'}")
        
        # === DIFERENÇAS ===
        st.markdown("#### ⚖️ DIFERENÇAS")
        
        st.write(f"**Diff Short Trend:** {abs(short_trend_ind - short_trend_uni):.12f}")
        st.write(f"**Diff Long Trend:** {abs(long_trend_ind - long_trend_uni):.12f}")
        st.write(f"**Diff Volatility:** {abs(volatility_ind - volatility_ai_uni):.12f}")
        st.write(f"**Diff Final Signal:** {abs(combined_signal_ind - lstm_signal_uni):.12f}")
        
        if abs(combined_signal_ind - lstm_signal_uni) > 0.000001:
            st.error("🚨 DIVERGÊNCIA DETECTADA!")
            st.write(f"Individual: {combined_signal_ind:.10f}")
            st.write(f"Unificada: {lstm_signal_uni:.10f}")
        else:
            st.success("✅ Sinais idênticos!")
    
    return {
        'individual_signal': combined_signal_ind,
        'unified_signal': lstm_signal_uni if len(prices) >= lookback_period else 0,
        'arrays_equal': np.array_equal(recent_prices_ind, recent_prices_uni) if len(prices) >= lookback_period else False
    }

if __name__ == "__main__":
    st.title("🔧 Debug Exato AI/LSTM")
    st.write("Comparação bit-a-bit dos cálculos")