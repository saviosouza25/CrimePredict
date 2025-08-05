#!/usr/bin/env python3
"""
🔧 DEBUG: PADRONIZAÇÃO DA ANÁLISE DE VOLUME
Ferramenta para verificar consistência entre análise individual e unificada
"""

import streamlit as st
import pandas as pd
import numpy as np

def debug_volume_analysis(df_with_indicators, current_price):
    """
    Comparar análise de volume individual vs unificada
    """
    st.markdown("### 🔧 Debug: Consistência de Volume")
    
    # === ANÁLISE INDIVIDUAL (ORIGINAL) ===
    st.markdown("#### 📊 Análise Individual de Volume")
    
    # Configuração da análise individual
    config = {'signal_factor': 1.0, 'volatility_threshold': 0.020, 'confidence': 0.70}
    
    # Volatilidade como proxy para volume (análise individual)
    volatility_individual = df_with_indicators['close'].tail(20).std() / current_price
    
    # Sinal individual
    base_signal_individual = (config['volatility_threshold'] - volatility_individual) * 0.015
    signal_individual = base_signal_individual * config['signal_factor']
    
    if volatility_individual > config['volatility_threshold']:
        signal_individual *= 0.8
    
    direction_individual = 'COMPRA' if signal_individual > 0.001 else 'VENDA' if signal_individual < -0.001 else 'NEUTRO'
    
    st.write(f"**Volatilidade:** {volatility_individual:.6f}")
    st.write(f"**Threshold:** {config['volatility_threshold']:.3f}")
    st.write(f"**Sinal Base:** {base_signal_individual:.6f}")
    st.write(f"**Sinal Final:** {signal_individual:.6f}")
    st.write(f"**Direção:** {direction_individual}")
    
    # === ANÁLISE UNIFICADA (PADRONIZADA) ===
    st.markdown("#### 🧠 Análise Unificada de Volume (Padronizada)")
    
    # Usar a mesma lógica da análise individual
    volume_volatility = df_with_indicators['close'].tail(20).std() / current_price
    volatility_threshold = 0.020
    signal_factor = 1.0
    
    base_volume_signal = (volatility_threshold - volume_volatility) * 0.015
    volume_confirmation = base_volume_signal * signal_factor
    
    if volume_volatility > volatility_threshold:
        volume_confirmation *= 0.8
    
    # Normalizar
    volume_norm = max(-1.0, min(1.0, volume_confirmation))
    direction_unified = 'COMPRA' if volume_norm > 0.1 else 'VENDA' if volume_norm < -0.1 else 'NEUTRO'
    
    st.write(f"**Volatilidade:** {volume_volatility:.6f}")
    st.write(f"**Threshold:** {volatility_threshold:.3f}")
    st.write(f"**Sinal Base:** {base_volume_signal:.6f}")
    st.write(f"**Sinal Final:** {volume_confirmation:.6f}")
    st.write(f"**Sinal Normalizado:** {volume_norm:.6f}")
    st.write(f"**Direção:** {direction_unified}")
    
    # === COMPARAÇÃO ===
    st.markdown("#### ⚖️ Comparação")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Individual", direction_individual, f"{signal_individual:.6f}")
    
    with col2:
        st.metric("Unificada", direction_unified, f"{volume_norm:.6f}")
    
    with col3:
        consistent = direction_individual == direction_unified
        st.metric("Consistente", "✅ SIM" if consistent else "❌ NÃO", "")
    
    return {
        'individual': {
            'signal': signal_individual,
            'direction': direction_individual,
            'volatility': volatility_individual
        },
        'unified': {
            'signal': volume_norm,
            'direction': direction_unified,
            'volatility': volume_volatility
        },
        'consistent': consistent
    }

if __name__ == "__main__":
    st.title("🔧 Debug: Padronização de Volume")
    st.write("Esta ferramenta verifica a consistência entre análises individual e unificada de volume")