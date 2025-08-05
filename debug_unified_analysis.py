#!/usr/bin/env python3
"""
Debug da Análise Unificada - Identificar cenários problemáticos
"""

def debug_unified_analysis_scenarios():
    """Analisar cenários onde componentes individuais não correspondem ao resultado final"""
    
    print("🔍 ANÁLISE DE CENÁRIOS PROBLEMÁTICOS NA UNIFICAÇÃO")
    print("=" * 60)
    
    # Cenário 1: Maioria positiva com resultado negativo
    print("\n📊 CENÁRIO 1: 4 COMPRA vs VENDA FORTE")
    print("-" * 40)
    
    # Simulação de componentes individuais
    technical_norm = 0.6    # COMPRA
    trend_norm = 0.4        # COMPRA  
    volume_norm = 0.3       # COMPRA
    sentiment_norm = 0.2    # COMPRA
    
    # Pesos iguais
    weight = 0.25
    
    # Cálculo do sinal unificado
    unified_signal = (
        technical_norm * weight +
        trend_norm * weight +
        volume_norm * weight +
        sentiment_norm * weight
    )
    
    print(f"Técnica: {technical_norm} * {weight} = {technical_norm * weight:.3f}")
    print(f"Tendência: {trend_norm} * {weight} = {trend_norm * weight:.3f}")
    print(f"Volume: {volume_norm} * {weight} = {volume_norm * weight:.3f}")
    print(f"Sentimento: {sentiment_norm} * {weight} = {sentiment_norm * weight:.3f}")
    print(f"Sinal Unificado: {unified_signal:.3f}")
    
    # Contagem de sinais
    positive_signals = sum(1 for comp in [technical_norm, trend_norm, volume_norm, sentiment_norm] if comp > 0.1)
    negative_signals = sum(1 for comp in [technical_norm, trend_norm, volume_norm, sentiment_norm] if comp < -0.1)
    
    print(f"Sinais Positivos: {positive_signals}")
    print(f"Sinais Negativos: {negative_signals}")
    
    # Lógica de decisão
    if positive_signals >= 3:
        if unified_signal > 0.3:
            direction = "COMPRA FORTE"
        else:
            direction = "COMPRA"
    elif negative_signals >= 3:
        if unified_signal < -0.3:
            direction = "VENDA FORTE"
        else:
            direction = "VENDA"
    else:
        if unified_signal > 0.2:
            direction = "COMPRA MODERADA"
        elif unified_signal < -0.2:
            direction = "VENDA MODERADA"
        else:
            direction = "LATERAL/NEUTRO"
    
    print(f"Resultado: {direction}")
    print(f"✅ CORRETO: 4 sinais positivos resultaram em {direction}")
    
    print("\n" + "=" * 60)
    print("\n📊 CENÁRIO 2: PROBLEMA IDENTIFICADO - Valores Extremos")
    print("-" * 40)
    
    # Cenário problemático com valores extremos antes da normalização
    technical_strength = 2.5     # Valor extremo
    trend_alignment = -3.0       # Valor extremo negativo
    volume_confirmation = 0.4
    sentiment_impact = 0.3
    
    print("ANTES DA NORMALIZAÇÃO:")
    print(f"Técnica: {technical_strength}")
    print(f"Tendência: {trend_alignment}")  
    print(f"Volume: {volume_confirmation}")
    print(f"Sentimento: {sentiment_impact}")
    
    # Normalização
    def normalize_component(value, max_val=1.0):
        return max(-max_val, min(max_val, value))
    
    technical_norm = normalize_component(technical_strength)
    trend_norm = normalize_component(trend_alignment)
    volume_norm = normalize_component(volume_confirmation)
    sentiment_norm = normalize_component(sentiment_impact)
    
    print("\nAPÓS NORMALIZAÇÃO:")
    print(f"Técnica: {technical_norm}")
    print(f"Tendência: {trend_norm}")
    print(f"Volume: {volume_norm}")
    print(f"Sentimento: {sentiment_norm}")
    
    # Sinal unificado
    unified_signal = (
        technical_norm * weight +
        trend_norm * weight +
        volume_norm * weight +
        sentiment_norm * weight
    )
    
    print(f"\nSinal Unificado: {unified_signal:.3f}")
    
    # Contagem
    positive_signals = sum(1 for comp in [technical_norm, trend_norm, volume_norm, sentiment_norm] if comp > 0.1)
    negative_signals = sum(1 for comp in [technical_norm, trend_norm, volume_norm, sentiment_norm] if comp < -0.1)
    
    print(f"Sinais Positivos: {positive_signals}")
    print(f"Sinais Negativos: {negative_signals}")
    
    # Decisão
    if positive_signals >= 3:
        direction = "COMPRA" if unified_signal > 0.3 else "COMPRA"
    elif negative_signals >= 3:
        direction = "VENDA FORTE" if unified_signal < -0.3 else "VENDA"
    else:
        direction = "LATERAL/NEUTRO"
    
    print(f"Resultado: {direction}")
    
    print("\n" + "=" * 60)
    print("\n📊 CENÁRIO 3: CENÁRIOS CRÍTICOS A AVALIAR")
    print("-" * 40)
    
    critical_scenarios = [
        {
            "name": "Divergência Extrema",
            "technical": 0.9,
            "trend": -0.8,
            "volume": 0.3,
            "sentiment": 0.2,
            "description": "Técnica muito forte vs Tendência muito fraca"
        },
        {
            "name": "Sentimento Dominante",
            "technical": 0.2,
            "trend": 0.1,
            "volume": 0.1,
            "sentiment": -0.9,
            "description": "Sentimento extremamente negativo vs outros moderados"
        },
        {
            "name": "Volume Contraditório",
            "technical": 0.4,
            "trend": 0.3,
            "volume": -0.8,
            "sentiment": 0.2,
            "description": "Volume muito baixo contradizendo outros sinais"
        }
    ]
    
    for i, scenario in enumerate(critical_scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   Descrição: {scenario['description']}")
        
        components = [scenario['technical'], scenario['trend'], scenario['volume'], scenario['sentiment']]
        unified = sum(comp * 0.25 for comp in components)
        
        positive = sum(1 for comp in components if comp > 0.1)
        negative = sum(1 for comp in components if comp < -0.1)
        
        print(f"   Componentes: T={scenario['technical']:.1f}, Tr={scenario['trend']:.1f}, V={scenario['volume']:.1f}, S={scenario['sentiment']:.1f}")
        print(f"   Unificado: {unified:.3f}")
        print(f"   Contagem: {positive} POS, {negative} NEG")
        
        if positive >= 3:
            result = "COMPRA" if unified > 0.3 else "COMPRA"
        elif negative >= 3:
            result = "VENDA FORTE" if unified < -0.3 else "VENDA"
        else:
            result = "LATERAL"
            
        print(f"   Resultado: {result}")
        
        # Verificar inconsistência
        if positive > negative and unified < 0:
            print(f"   ⚠️  INCONSISTÊNCIA: Mais sinais positivos mas resultado negativo!")
        elif negative > positive and unified > 0:
            print(f"   ⚠️  INCONSISTÊNCIA: Mais sinais negativos mas resultado positivo!")
    
    print("\n" + "=" * 60)
    print("\n🎯 RECOMENDAÇÕES PARA CORREÇÃO:")
    print("-" * 40)
    print("1. Verificar se normalização está sendo aplicada corretamente")
    print("2. Implementar threshold mínimo para sinais extremos")
    print("3. Adicionar weight adjustment baseado em confiança")
    print("4. Criar override quando consenso é muito claro (4/4 ou 3/4)")
    print("5. Implementar debug logging em tempo real")

if __name__ == "__main__":
    debug_unified_analysis_scenarios()