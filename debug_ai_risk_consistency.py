#!/usr/bin/env python3
"""
🔧 DEBUG: CONSISTÊNCIA AI/LSTM E RISK
Ferramenta para verificar divergências entre análises individuais e unificadas
"""

import streamlit as st
import pandas as pd
import numpy as np

def debug_ai_lstm_consistency(df_with_indicators, current_price, lookback_period=20, epochs=50):
    """
    Comparar análise AI/LSTM individual vs unificada
    """
    st.markdown("### 🤖 Debug: Consistência IA/LSTM")
    
    # === ANÁLISE INDIVIDUAL ===
    st.markdown("#### 📊 Análise Individual AI/LSTM")
    
    # Parâmetros da análise individual
    risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
    recent_prices = df_with_indicators['close'].tail(lookback_period).values
    
    # Cálculos individuais
    short_trend_individual = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
    long_trend_individual = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    volatility_individual = np.std(recent_prices) / np.mean(recent_prices)
    
    base_learning_factor = min(1.0, epochs / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    # Sinal individual combinado
    trend_signal = long_trend_individual * 0.6 * risk_config['signal_damping']
    momentum_signal = short_trend_individual * 0.4 * risk_config['signal_damping']
    volatility_signal = max(-0.02, min(0.02, (0.015 - volatility_individual) * 0.5))
    
    if volatility_individual > 0.015:
        volatility_signal *= 0.8
    
    combined_signal_individual = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
    
    direction_individual = 'COMPRA' if combined_signal_individual > 0.001 else 'VENDA' if combined_signal_individual < -0.001 else 'NEUTRO'
    
    st.write(f"**Long Trend:** {long_trend_individual:.6f}")
    st.write(f"**Short Trend:** {short_trend_individual:.6f}")
    st.write(f"**Volatilidade:** {volatility_individual:.6f}")
    st.write(f"**Learning Factor:** {learning_factor:.3f}")
    st.write(f"**Sinal Combinado:** {combined_signal_individual:.6f}")
    st.write(f"**Direção:** {direction_individual}")
    
    # === ANÁLISE UNIFICADA ===
    st.markdown("#### 🧠 Análise Unificada AI/LSTM")
    
    prices = df_with_indicators['close'].values
    
    # Cálculos unificados
    lstm_signal = 0
    if len(prices) >= 20:
        long_trend_unified = (prices[-1] - prices[-20]) / prices[-20]
        recent_volatility_unified = np.std(prices[-5:]) / prices[-1] if len(prices) >= 5 else 0
        
        if long_trend_unified > 0.005 and recent_volatility_unified < 0.01:
            lstm_signal = 0.6
        elif long_trend_unified > 0.002:
            lstm_signal = 0.3
        elif long_trend_unified < -0.005 and recent_volatility_unified < 0.01:
            lstm_signal = -0.6
        elif long_trend_unified < -0.002:
            lstm_signal = -0.3
        else:
            lstm_signal = long_trend_unified * 50
    
    # Normalizar
    lstm_norm = max(-1.0, min(1.0, lstm_signal))
    direction_unified = 'COMPRA' if lstm_norm > 0.1 else 'VENDA' if lstm_norm < -0.1 else 'NEUTRO'
    
    st.write(f"**Long Trend:** {long_trend_unified:.6f}")
    st.write(f"**Recent Volatility:** {recent_volatility_unified:.6f}")
    st.write(f"**LSTM Signal:** {lstm_signal:.6f}")
    st.write(f"**LSTM Normalizado:** {lstm_norm:.6f}")
    st.write(f"**Direção:** {direction_unified}")
    
    # === COMPARAÇÃO ===
    st.markdown("#### ⚖️ Comparação AI/LSTM")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Individual", direction_individual, f"{combined_signal_individual:.6f}")
    
    with col2:
        st.metric("Unificada", direction_unified, f"{lstm_norm:.6f}")
    
    with col3:
        consistent = direction_individual == direction_unified
        st.metric("Consistente", "✅ SIM" if consistent else "❌ NÃO", "")
    
    return {
        'individual': {'signal': combined_signal_individual, 'direction': direction_individual},
        'unified': {'signal': lstm_norm, 'direction': direction_unified},
        'consistent': consistent
    }

def debug_risk_consistency(current_price, df_with_indicators):
    """
    Comparar análise Risk individual vs unificada
    """
    st.markdown("### ⚖️ Debug: Consistência Risk")
    
    # === ANÁLISE INDIVIDUAL ===
    st.markdown("#### 📊 Análise Individual Risk")
    
    # Fatores de risco da análise individual
    factor = {'volatility': 0.012, 'confidence': 0.75, 'signal_range': 0.015}
    
    # Sinal individual (usando random - PROBLEMA!)
    signal_individual = np.random.uniform(-factor['signal_range'], factor['signal_range'])
    direction_individual = 'COMPRA' if signal_individual > 0.001 else 'VENDA' if signal_individual < -0.001 else 'NEUTRO'
    
    st.write(f"**Sinal Random:** {signal_individual:.6f}")
    st.write(f"**Range:** ±{factor['signal_range']:.3f}")
    st.write(f"**Direção:** {direction_individual}")
    st.warning("⚠️ Análise individual usa RANDOM - não determinística!")
    
    # === ANÁLISE UNIFICADA ===
    st.markdown("#### 🧠 Análise Unificada Risk")
    
    prices = df_with_indicators['close'].values
    price_changes = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else np.array([0])
    volatility = np.std(price_changes) if len(price_changes) > 0 else 0
    
    # Cálculo de risco unificado
    risk_score = 0
    if volatility > 0:
        if volatility > 0.02:
            risk_score = -0.4
        elif volatility > 0.01:
            risk_score = -0.2
        elif volatility < 0.005:
            risk_score = 0.3
        else:
            risk_score = 0.1
    
    # Normalizar
    risk_norm = max(-1.0, min(1.0, risk_score))
    direction_unified = 'COMPRA' if risk_norm > 0.1 else 'VENDA' if risk_norm < -0.1 else 'NEUTRO'
    
    st.write(f"**Volatilidade:** {volatility:.6f}")
    st.write(f"**Risk Score:** {risk_score:.6f}")
    st.write(f"**Risk Normalizado:** {risk_norm:.6f}")
    st.write(f"**Direção:** {direction_unified}")
    
    # === COMPARAÇÃO ===
    st.markdown("#### ⚖️ Comparação Risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Individual", direction_individual, f"{signal_individual:.6f}")
    
    with col2:
        st.metric("Unificada", direction_unified, f"{risk_norm:.6f}")
    
    with col3:
        st.metric("Consistente", "❌ IMPOSSÍVEL", "Random vs Determinístico")
    
    return {
        'individual': {'signal': signal_individual, 'direction': direction_individual},
        'unified': {'signal': risk_norm, 'direction': direction_unified},
        'consistent': False,
        'issue': 'Individual usa random, unificada usa volatilidade determinística'
    }

if __name__ == "__main__":
    st.title("🔧 Debug: Consistência AI/LSTM e Risk")
    st.write("Ferramenta para verificar divergências entre análises individuais e unificadas")