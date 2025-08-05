#!/usr/bin/env python3
"""
🔧 DEBUG COMPLETO: Análise de TODOS os componentes
Comparação bit-a-bit entre análises individuais e unificada
"""

import streamlit as st
import pandas as pd
import numpy as np
from services.alpha_vantage_service import AlphaVantageService
from services.sentiment_service import SentimentService

def debug_all_components_consistency(pair, interval, api_key):
    """Debug completo de todos os 6 componentes"""
    
    st.markdown("### 🔍 DEBUG COMPLETO: TODOS OS COMPONENTES")
    
    # Obter dados
    alpha_service = AlphaVantageService(api_key)
    sentiment_service = SentimentService(api_key)
    
    try:
        # Dados básicos
        df = alpha_service.get_forex_data(pair, interval)
        current_price = df['close'].iloc[-1]
        
        # Calcular indicadores técnicos
        df_with_indicators = alpha_service.calculate_technical_indicators(df)
        
        st.write(f"**Par:** {pair}")
        st.write(f"**Preço atual:** {current_price:.5f}")
        st.write(f"**Dados disponíveis:** {len(df)} períodos")
        
        # === 1. ANÁLISE TÉCNICA ===
        st.markdown("#### 📊 1. ANÁLISE TÉCNICA")
        
        # Individual
        tech_individual = run_technical_analysis_debug(current_price, df_with_indicators)
        
        # Cálculo unificado (extraído do código principal)
        rsi = df_with_indicators['rsi'].iloc[-1]
        macd = df_with_indicators['macd'].iloc[-1]
        bb_position = (current_price - df_with_indicators['bb_lower'].iloc[-1]) / (df_with_indicators['bb_upper'].iloc[-1] - df_with_indicators['bb_lower'].iloc[-1])
        
        # Forças dos sinais técnicos
        rsi_signal = 0.5 - (rsi / 100)  # RSI invertido (alta = negativo)
        macd_signal = macd * 50  # MACD amplificado
        bb_signal = (bb_position - 0.5) * 0.4  # Bollinger normalizado
        
        technical_strength_unified = (rsi_signal * 0.4 + macd_signal * 0.4 + bb_signal * 0.2)
        
        st.write(f"**INDIVIDUAL:** {tech_individual['technical_strength']:.10f}")
        st.write(f"**UNIFICADA:** {technical_strength_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(tech_individual['technical_strength'] - technical_strength_unified):.12f}")
        
        individual_dir = 'COMPRA' if tech_individual['technical_strength'] > 0.001 else 'VENDA' if tech_individual['technical_strength'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if technical_strength_unified > 0.001 else 'VENDA' if technical_strength_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA TÉCNICA!")
        else:
            st.success("✅ Técnica consistente")
        
        # === 2. ANÁLISE DE TENDÊNCIA ===
        st.markdown("#### 📈 2. ANÁLISE DE TENDÊNCIA")
        
        # Individual
        trend_individual = run_trend_analysis_debug(current_price, df_with_indicators)
        
        # Unificado
        prices = df_with_indicators['close'].values
        trend_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        trend_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        
        trend_alignment_unified = (trend_5 * 0.5 + trend_10 * 0.3 + trend_20 * 0.2)
        
        st.write(f"**INDIVIDUAL:** {trend_individual['trend_alignment']:.10f}")
        st.write(f"**UNIFICADA:** {trend_alignment_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(trend_individual['trend_alignment'] - trend_alignment_unified):.12f}")
        
        individual_dir = 'COMPRA' if trend_individual['trend_alignment'] > 0.001 else 'VENDA' if trend_individual['trend_alignment'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if trend_alignment_unified > 0.001 else 'VENDA' if trend_alignment_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA TENDÊNCIA!")
        else:
            st.success("✅ Tendência consistente")
        
        # === 3. ANÁLISE DE VOLUME ===
        st.markdown("#### 📊 3. ANÁLISE DE VOLUME")
        
        # Individual
        volume_individual = run_volume_analysis_debug(current_price, df_with_indicators)
        
        # Unificado (usar o mesmo cálculo padronizado)
        volatility_threshold = 0.020
        volume_volatility_unified = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.015
        volume_confirmation_unified = 0.5 if volume_volatility_unified > volatility_threshold else -0.3
        
        st.write(f"**INDIVIDUAL:** {volume_individual['volume_confirmation']:.10f}")
        st.write(f"**UNIFICADA:** {volume_confirmation_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(volume_individual['volume_confirmation'] - volume_confirmation_unified):.12f}")
        
        individual_dir = 'COMPRA' if volume_individual['volume_confirmation'] > 0.001 else 'VENDA' if volume_individual['volume_confirmation'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if volume_confirmation_unified > 0.001 else 'VENDA' if volume_confirmation_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA VOLUME!")
        else:
            st.success("✅ Volume consistente")
        
        # === 4. ANÁLISE DE SENTIMENTO ===
        st.markdown("#### 💬 4. ANÁLISE DE SENTIMENTO")
        
        # Individual
        sentiment_individual = run_sentiment_analysis_debug(pair, sentiment_service)
        
        # Unificado (usar mesmo cálculo)
        sentiment_score_unified = sentiment_individual['sentiment_score']
        if sentiment_score_unified > 0.05:
            sentiment_impact_unified = sentiment_score_unified * 0.8
        elif sentiment_score_unified < -0.05:
            sentiment_impact_unified = sentiment_score_unified * 0.6
        else:
            sentiment_impact_unified = sentiment_score_unified * 0.2
        
        st.write(f"**INDIVIDUAL:** {sentiment_individual['sentiment_impact']:.10f}")
        st.write(f"**UNIFICADA:** {sentiment_impact_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(sentiment_individual['sentiment_impact'] - sentiment_impact_unified):.12f}")
        
        individual_dir = 'COMPRA' if sentiment_individual['sentiment_impact'] > 0.001 else 'VENDA' if sentiment_individual['sentiment_impact'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if sentiment_impact_unified > 0.001 else 'VENDA' if sentiment_impact_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA SENTIMENTO!")
        else:
            st.success("✅ Sentimento consistente")
        
        # === 5. ANÁLISE AI/LSTM ===
        st.markdown("#### 🤖 5. ANÁLISE AI/LSTM")
        
        # Individual
        ai_individual = run_ai_analysis_debug(current_price, 20, 50, df_with_indicators)
        
        # Unificado (extrair do código)
        lstm_signal_unified = calculate_lstm_unified(prices, 20)
        
        st.write(f"**INDIVIDUAL:** {ai_individual['combined_signal']:.10f}")
        st.write(f"**UNIFICADA:** {lstm_signal_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(ai_individual['combined_signal'] - lstm_signal_unified):.12f}")
        
        individual_dir = 'COMPRA' if ai_individual['combined_signal'] > 0.001 else 'VENDA' if ai_individual['combined_signal'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if lstm_signal_unified > 0.001 else 'VENDA' if lstm_signal_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA AI/LSTM!")
        else:
            st.success("✅ AI/LSTM consistente")
        
        # === 6. ANÁLISE DE RISCO ===
        st.markdown("#### ⚠️ 6. ANÁLISE DE RISCO")
        
        # Individual
        risk_individual = run_risk_analysis_debug(current_price, df_with_indicators)
        
        # Unificado
        volatility_unified = np.std(np.diff(prices[-20:]) / prices[-20:-1]) if len(prices) >= 20 else 0.012
        risk_score_unified = 0
        if volatility_unified > 0.02:
            risk_score_unified = -0.4
        elif volatility_unified > 0.01:
            risk_score_unified = -0.2
        elif volatility_unified < 0.005:
            risk_score_unified = 0.3
        else:
            risk_score_unified = 0.1
        
        st.write(f"**INDIVIDUAL:** {risk_individual['risk_score']:.10f}")
        st.write(f"**UNIFICADA:** {risk_score_unified:.10f}")
        st.write(f"**DIFERENÇA:** {abs(risk_individual['risk_score'] - risk_score_unified):.12f}")
        
        individual_dir = 'COMPRA' if risk_individual['risk_score'] > 0.001 else 'VENDA' if risk_individual['risk_score'] < -0.001 else 'NEUTRO'
        unified_dir = 'COMPRA' if risk_score_unified > 0.001 else 'VENDA' if risk_score_unified < -0.001 else 'NEUTRO'
        
        st.write(f"**Direção Individual:** {individual_dir}")
        st.write(f"**Direção Unificada:** {unified_dir}")
        
        if individual_dir != unified_dir:
            st.error("🚨 DIVERGÊNCIA RISCO!")
        else:
            st.success("✅ Risco consistente")
        
        # === RESUMO FINAL ===
        st.markdown("#### 📋 RESUMO FINAL")
        
        components = ['Técnica', 'Tendência', 'Volume', 'Sentimento', 'AI/LSTM', 'Risco']
        individual_dirs = [
            'COMPRA' if tech_individual['technical_strength'] > 0.001 else 'VENDA' if tech_individual['technical_strength'] < -0.001 else 'NEUTRO',
            'COMPRA' if trend_individual['trend_alignment'] > 0.001 else 'VENDA' if trend_individual['trend_alignment'] < -0.001 else 'NEUTRO',
            'COMPRA' if volume_individual['volume_confirmation'] > 0.001 else 'VENDA' if volume_individual['volume_confirmation'] < -0.001 else 'NEUTRO',
            'COMPRA' if sentiment_individual['sentiment_impact'] > 0.001 else 'VENDA' if sentiment_individual['sentiment_impact'] < -0.001 else 'NEUTRO',
            'COMPRA' if ai_individual['combined_signal'] > 0.001 else 'VENDA' if ai_individual['combined_signal'] < -0.001 else 'NEUTRO',
            'COMPRA' if risk_individual['risk_score'] > 0.001 else 'VENDA' if risk_individual['risk_score'] < -0.001 else 'NEUTRO'
        ]
        
        unified_dirs = [
            'COMPRA' if technical_strength_unified > 0.001 else 'VENDA' if technical_strength_unified < -0.001 else 'NEUTRO',
            'COMPRA' if trend_alignment_unified > 0.001 else 'VENDA' if trend_alignment_unified < -0.001 else 'NEUTRO',
            'COMPRA' if volume_confirmation_unified > 0.001 else 'VENDA' if volume_confirmation_unified < -0.001 else 'NEUTRO',
            'COMPRA' if sentiment_impact_unified > 0.001 else 'VENDA' if sentiment_impact_unified < -0.001 else 'NEUTRO',
            'COMPRA' if lstm_signal_unified > 0.001 else 'VENDA' if lstm_signal_unified < -0.001 else 'NEUTRO',
            'COMPRA' if risk_score_unified > 0.001 else 'VENDA' if risk_score_unified < -0.001 else 'NEUTRO'
        ]
        
        inconsistent_count = 0
        for i, comp in enumerate(components):
            if individual_dirs[i] != unified_dirs[i]:
                inconsistent_count += 1
                st.error(f"❌ {comp}: Individual={individual_dirs[i]} vs Unificada={unified_dirs[i]}")
            else:
                st.success(f"✅ {comp}: {individual_dirs[i]} (consistente)")
        
        if inconsistent_count == 0:
            st.success("🎉 TODOS OS COMPONENTES CONSISTENTES!")
        else:
            st.error(f"🚨 {inconsistent_count}/6 COMPONENTES INCONSISTENTES!")
        
    except Exception as e:
        st.error(f"Erro no debug: {str(e)}")

def run_technical_analysis_debug(current_price, df_with_indicators):
    """Reproduzir análise técnica individual"""
    rsi = df_with_indicators['rsi'].iloc[-1]
    macd = df_with_indicators['macd'].iloc[-1]
    bb_position = (current_price - df_with_indicators['bb_lower'].iloc[-1]) / (df_with_indicators['bb_upper'].iloc[-1] - df_with_indicators['bb_lower'].iloc[-1])
    
    rsi_signal = 0.5 - (rsi / 100)
    macd_signal = macd * 50
    bb_signal = (bb_position - 0.5) * 0.4
    
    technical_strength = (rsi_signal * 0.4 + macd_signal * 0.4 + bb_signal * 0.2)
    
    return {'technical_strength': technical_strength}

def run_trend_analysis_debug(current_price, df_with_indicators):
    """Reproduzir análise de tendência individual"""
    prices = df_with_indicators['close'].values
    trend_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
    trend_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
    trend_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
    
    trend_alignment = (trend_5 * 0.5 + trend_10 * 0.3 + trend_20 * 0.2)
    
    return {'trend_alignment': trend_alignment}

def run_volume_analysis_debug(current_price, df_with_indicators):
    """Reproduzir análise de volume individual"""
    prices = df_with_indicators['close'].values
    volatility_threshold = 0.020
    volume_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.015
    volume_confirmation = 0.5 if volume_volatility > volatility_threshold else -0.3
    
    return {'volume_confirmation': volume_confirmation}

def run_sentiment_analysis_debug(pair, sentiment_service):
    """Reproduzir análise de sentimento individual"""
    sentiment_score = sentiment_service.fetch_news_sentiment(pair)
    
    if sentiment_score > 0.05:
        sentiment_impact = sentiment_score * 0.8
    elif sentiment_score < -0.05:
        sentiment_impact = sentiment_score * 0.6
    else:
        sentiment_impact = sentiment_score * 0.2
    
    return {'sentiment_score': sentiment_score, 'sentiment_impact': sentiment_impact}

def run_ai_analysis_debug(current_price, lookback_period, epochs, df_with_indicators):
    """Reproduzir análise AI/LSTM individual"""
    risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
    
    recent_prices = df_with_indicators['close'].tail(lookback_period).values
    
    short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
    long_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    volatility = np.std(recent_prices) / np.mean(recent_prices)
    
    base_learning_factor = min(1.0, epochs / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    trend_signal = np.tanh(long_trend * 10) * 0.020 * risk_config['signal_damping']
    momentum_signal = np.tanh(short_trend * 15) * 0.015 * risk_config['signal_damping']
    volatility_signal = (0.02 - volatility) * 0.010
    
    if volatility > 0.015:
        volatility_signal *= 0.8
    
    combined_signal = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
    
    return {'combined_signal': combined_signal}

def run_risk_analysis_debug(current_price, df_with_indicators):
    """Reproduzir análise de risco individual"""
    prices = df_with_indicators['close'].values
    price_changes = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else np.array([0])
    volatility = np.std(price_changes) if len(price_changes) > 0 else 0.012
    
    risk_score = 0
    if volatility > 0.02:
        risk_score = -0.4
    elif volatility > 0.01:
        risk_score = -0.2
    elif volatility < 0.005:
        risk_score = 0.3
    else:
        risk_score = 0.1
    
    return {'risk_score': risk_score}

def calculate_lstm_unified(prices, lookback_period):
    """Calcular LSTM unificado exatamente como no código principal"""
    if len(prices) < lookback_period:
        return 0
    
    risk_config = {'volatility_tolerance': 1.0, 'signal_damping': 1.0, 'min_confidence': 0.65}
    recent_prices = prices[-lookback_period:]
    
    short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
    long_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    volatility_ai = np.std(recent_prices) / np.mean(recent_prices)
    
    base_learning_factor = min(1.0, 50 / 100)
    learning_factor = base_learning_factor * risk_config['volatility_tolerance']
    
    trend_signal = np.tanh(long_trend * 10) * 0.020 * risk_config['signal_damping']
    momentum_signal = np.tanh(short_trend * 15) * 0.015 * risk_config['signal_damping']
    volatility_signal = (0.02 - volatility_ai) * 0.010
    
    if volatility_ai > 0.015:
        volatility_signal *= 0.8
    
    lstm_signal = (trend_signal * 0.5 + momentum_signal * 0.3 + volatility_signal * 0.2) * learning_factor
    
    return lstm_signal

if __name__ == "__main__":
    st.title("🔧 Debug Completo de Consistência")
    st.write("Análise detalhada de todos os 6 componentes")