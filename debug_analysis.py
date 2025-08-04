#!/usr/bin/env python3
"""Debug script para identificar o erro 'argument of type float is not iterable'"""

import pandas as pd
import numpy as np
import traceback

def debug_unified_analysis():
    """Teste isolado da análise unificada"""
    try:
        # Simular dados de entrada
        pair = "EUR/USD"
        current_price = 1.0850
        sentiment_score = 0.1
        confidence = 0.65
        
        # Simular dataframe com indicadores
        df_data = {
            'close': [1.0840, 1.0845, 1.0850],
            'high': [1.0845, 1.0850, 1.0855],
            'low': [1.0835, 1.0840, 1.0845],
            'rsi': [55.0, 56.0, 57.0],
            'macd': [0.0001, 0.0002, 0.0003],
            'sma_20': [1.0845, 1.0846, 1.0847],
            'sma_5': [1.0848, 1.0849, 1.0850],
            'sma_10': [1.0846, 1.0847, 1.0848]
        }
        df_with_indicators = pd.DataFrame(df_data)
        
        print("Testando cálculos isolados...")
        
        # Teste 1: Cálculos de tendência
        prices = df_with_indicators['close'].values
        
        # Tendências multi-timeframe
        trend_5 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
        trend_10 = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        trend_20 = (prices[-1] - prices[0]) / prices[0] if len(prices) >= 1 else 0
        
        print(f"Tendências: 5p={trend_5}, 10p={trend_10}, 20p={trend_20}")
        
        # Teste 2: Indicadores técnicos
        rsi = df_with_indicators['rsi'].iloc[-1]
        macd = df_with_indicators['macd'].iloc[-1] 
        sma_20 = df_with_indicators['sma_20'].iloc[-1]
        
        print(f"Indicadores: RSI={rsi}, MACD={macd}, SMA20={sma_20}")
        
        # Teste 3: Sinais técnicos
        rsi_signal = (70 - rsi) / 50 if rsi > 70 else (30 - rsi) / 50 if rsi < 30 else 0
        macd_signal = max(-0.5, min(0.5, macd * 5000))
        sma_signal = (current_price - sma_20) / sma_20
        
        print(f"Sinais: RSI={rsi_signal}, MACD={macd_signal}, SMA={sma_signal}")
        
        # Teste 4: Análise de tendência
        if trend_5 > 0.001 and trend_10 > 0.001 and trend_20 > 0.001:
            trend_alignment = 0.9
        elif trend_5 > 0.0005 and trend_10 > 0.0005:
            trend_alignment = 0.6
        elif trend_5 > 0 and trend_10 > 0:
            trend_alignment = 0.3
        elif trend_5 < -0.001 and trend_10 < -0.001 and trend_20 < -0.001:
            trend_alignment = -0.9
        elif trend_5 < -0.0005 and trend_10 < -0.0005:
            trend_alignment = -0.6
        elif trend_5 < 0 and trend_10 < 0:
            trend_alignment = -0.3
        else:
            trend_alignment = 0
            
        print(f"Alinhamento de tendência: {trend_alignment}")
        
        # Teste 5: Sinal unificado
        technical_strength = (rsi_signal * 0.4 + macd_signal * 0.35 + sma_signal * 0.25) * 1.2
        
        unified_signal = (
            technical_strength * 0.35 +
            trend_alignment * 0.30 +
            sentiment_score * 0.20 * 0.8
        )
        
        print(f"Sinal técnico: {technical_strength}")
        print(f"Sinal unificado: {unified_signal}")
        
        # Teste 6: Direção (AQUI PODE ESTAR O PROBLEMA)
        unified_signal = float(unified_signal)  # Converter para float
        
        if unified_signal > 0.4:
            direction = "COMPRA FORTE"
            probability = min(85, 65 + (unified_signal * 25))
        elif unified_signal > 0.15:
            direction = "COMPRA"
            probability = min(75, 55 + (unified_signal * 35))
        elif unified_signal < -0.4:
            direction = "VENDA FORTE"  
            probability = min(85, 65 + (abs(unified_signal) * 25))
        elif unified_signal < -0.15:
            direction = "VENDA"
            probability = min(75, 55 + (abs(unified_signal) * 35))
        else:
            direction = "LATERAL/NEUTRO"
            probability = 50
            
        print(f"Direção: {direction} (tipo: {type(direction)})")
        print(f"Probabilidade: {probability} (tipo: {type(probability)})")
        
        # Teste 7: Verificação de strings (AQUI PODE ESTAR O PROBLEMA)
        direction_str = str(direction)  # Garantir que é string
        print(f"Direção como string: {direction_str}")
        
        # Teste das condições que podem dar erro
        print("\nTestando condições de string:")
        print(f"'COMPRA FORTE' in '{direction_str}': {'COMPRA FORTE' in direction_str}")
        print(f"'COMPRA' in '{direction_str}': {'COMPRA' in direction_str}")
        print(f"'VENDA FORTE' in '{direction_str}': {'VENDA FORTE' in direction_str}")
        print(f"'VENDA' in '{direction_str}': {'VENDA' in direction_str}")
        
        print("\n✅ TESTE PASSOU - Nenhum erro encontrado!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ENCONTRADO: {e}")
        print(f"Tipo do erro: {type(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_unified_analysis()