#!/usr/bin/env python3
"""
🔧 DEBUG DETALHADO: AI/LSTM INDIVIDUAL VS UNIFICADA
Análise passo-a-passo dos cálculos para identificar divergência
"""

import streamlit as st
import pandas as pd
import numpy as np

def debug_ai_lstm_step_by_step(df_with_indicators, current_price, lookback_period=20, epochs=50):
    """
    Debug passo-a-passo dos cálculos AI/LSTM
    """
    st.markdown("### 🤖 DEBUG DETALHADO: AI/LSTM")
    
    # === ANÁLISE INDIVIDUAL ===
    st.markdown("#### 📊 Análise Individual (run_ai_analysis)")
    
    # Parâmetros individuais
    risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
    recent_prices = df_with_indicators['close'].tail(lookback_period).values
    
    st.write(f"**Lookback Period:** {lookback_period}")
    st.write(f"**Épocas:** {epochs}")
    st.write(f"**Preços recentes (últimos 5):** {recent_prices[-5:] if len(recent_prices) >= 5 else recent_prices}")
    
    # Cálculos step-by-step individual
    short_trend_ind = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
    long_trend_ind = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    volatility_ind = np.std(recent_prices) / np.mean(recent_prices)
    
    st.write(f"**Short Trend Individual:** {short_trend_ind:.8f}")
    st.write(f"**Long Trend Individual:** {long_trend_ind:.8f}")
    st.write(f"**Volatilidade Individual:** {volatility_ind:.8f}")
    
    # Learning factor
    base_learning_factor = min(1.0, epochs / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    st.write(f"**Base Learning Factor:** {base_learning_factor:.8f}")
    st.write(f"**Learning Factor Final:** {learning_factor:.8f}")
    
    # Sinais individuais
    trend_signal_ind = np.tanh(long_trend_ind * 10) * 0.020 * risk_config['signal_damping']
    momentum_signal_ind = np.tanh(short_trend_ind * 15) * 0.015 * risk_config['signal_damping']
    volatility_signal_ind = (0.02 - volatility_ind) * 0.010
    
    st.write(f"**Trend Signal Individual:** {trend_signal_ind:.8f}")
    st.write(f"**Momentum Signal Individual:** {momentum_signal_ind:.8f}")
    st.write(f"**Volatility Signal Individual (antes):** {volatility_signal_ind:.8f}")
    
    # Ajuste volatilidade
    if volatility_ind > 0.015:
        volatility_signal_ind *= 0.8
        st.write(f"**Volatility Signal Individual (após ajuste):** {volatility_signal_ind:.8f}")
    
    # Sinal final individual
    combined_signal_ind = (trend_signal_ind * 0.5 + momentum_signal_ind * 0.3 + volatility_signal_ind * 0.2) * learning_factor
    direction_ind = 'COMPRA' if combined_signal_ind > 0.001 else 'VENDA' if combined_signal_ind < -0.001 else 'NEUTRO'
    
    st.write(f"**Sinal Combinado Individual:** {combined_signal_ind:.8f}")
    st.write(f"**Direção Individual:** {direction_ind}")
    
    # === ANÁLISE UNIFICADA ===
    st.markdown("#### 🧠 Análise Unificada")
    
    prices = df_with_indicators['close'].values
    st.write(f"**Total de preços disponíveis:** {len(prices)}")
    st.write(f"**Preços para cálculo (últimos 20):** {prices[-20:] if len(prices) >= 20 else prices}")
    
    # Cálculos unificados
    if len(prices) >= lookback_period:
        recent_prices_unified = prices[-lookback_period:]
        
        st.write(f"**Recent prices unified igual individual?** {np.array_equal(recent_prices, recent_prices_unified)}")
        
        # Mesmos cálculos da unificada
        short_trend_uni = (recent_prices_unified[-1] - recent_prices_unified[-5]) / recent_prices_unified[-5] if len(recent_prices_unified) >= 5 else 0
        long_trend_uni = (recent_prices_unified[-1] - recent_prices_unified[0]) / recent_prices_unified[0]
        volatility_ai_uni = np.std(recent_prices_unified) / np.mean(recent_prices_unified)
        
        st.write(f"**Short Trend Unificada:** {short_trend_uni:.8f}")
        st.write(f"**Long Trend Unificada:** {long_trend_uni:.8f}")
        st.write(f"**Volatilidade Unificada:** {volatility_ai_uni:.8f}")
        
        # Learning factor unificado
        base_learning_factor_uni = min(1.0, epochs / 100)
        learning_factor_uni = base_learning_factor_uni * risk_config['volatility_tolerance']
        
        st.write(f"**Learning Factor Unificada:** {learning_factor_uni:.8f}")
        
        # Sinais unificados
        trend_signal_uni = np.tanh(long_trend_uni * 10) * 0.020 * risk_config['signal_damping']
        momentum_signal_uni = np.tanh(short_trend_uni * 15) * 0.015 * risk_config['signal_damping']
        volatility_signal_uni = (0.02 - volatility_ai_uni) * 0.010
        
        st.write(f"**Trend Signal Unificada:** {trend_signal_uni:.8f}")
        st.write(f"**Momentum Signal Unificada:** {momentum_signal_uni:.8f}")
        st.write(f"**Volatility Signal Unificada (antes):** {volatility_signal_uni:.8f}")
        
        # Ajuste volatilidade unificada
        if volatility_ai_uni > 0.015:
            volatility_signal_uni *= 0.8
            st.write(f"**Volatility Signal Unificada (após ajuste):** {volatility_signal_uni:.8f}")
        
        # Sinal final unificado
        lstm_signal_uni = (trend_signal_uni * 0.5 + momentum_signal_uni * 0.3 + volatility_signal_uni * 0.2) * learning_factor_uni
        
        # Normalizar
        lstm_norm = max(-1.0, min(1.0, lstm_signal_uni))
        direction_uni = 'COMPRA' if lstm_norm > 0.1 else 'VENDA' if lstm_norm < -0.1 else 'NEUTRO'
        
        st.write(f"**LSTM Signal Unificada:** {lstm_signal_uni:.8f}")
        st.write(f"**LSTM Normalizada:** {lstm_norm:.8f}")
        st.write(f"**Direção Unificada:** {direction_uni}")
    else:
        st.warning("Dados insuficientes para análise unificada")
        lstm_norm = 0
        direction_uni = "NEUTRO"
    
    # === COMPARAÇÃO DETALHADA ===
    st.markdown("#### ⚖️ Comparação Detalhada")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Individual", direction_ind, f"{combined_signal_ind:.8f}")
    
    with col2:
        st.metric("Unificada", direction_uni, f"{lstm_norm:.8f}")
    
    with col3:
        consistent = direction_ind == direction_uni
        st.metric("Consistente", "✅ SIM" if consistent else "❌ NÃO", "")
        
    # Diferenças específicas 
    if not consistent:
        st.markdown("#### 🔍 DIFERENÇAS IDENTIFICADAS:")
        
        if len(recent_prices) >= 5 and len(prices) >= lookback_period:
            st.write(f"**Diferença Short Trend:** {abs(short_trend_ind - short_trend_uni):.8f}")
            st.write(f"**Diferença Long Trend:** {abs(long_trend_ind - long_trend_uni):.8f}")
            st.write(f"**Diferença Volatilidade:** {abs(volatility_ind - volatility_ai_uni):.8f}")
            st.write(f"**Diferença Sinal Final:** {abs(combined_signal_ind - lstm_signal_uni):.8f}")
            
            # Verificar se os arrays são idênticos
            arrays_equal = np.array_equal(recent_prices, recent_prices_unified) if len(prices) >= lookback_period else False
            st.write(f"**Arrays de preços idênticos:** {'✅ SIM' if arrays_equal else '❌ NÃO'}")
    
    return {
        'individual': {
            'signal': combined_signal_ind,
            'direction': direction_ind,
            'components': {
                'trend': trend_signal_ind,
                'momentum': momentum_signal_ind,
                'volatility': volatility_signal_ind,
                'learning': learning_factor
            }
        },
        'unified': {
            'signal': lstm_norm,
            'direction': direction_uni,
            'raw_signal': lstm_signal_uni if len(prices) >= lookback_period else 0
        },
        'consistent': consistent
    }

if __name__ == "__main__":
    st.title("🔧 Debug Detalhado: AI/LSTM")
    st.write("Análise passo-a-passo para identificar divergências específicas")