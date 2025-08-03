"""
Serviço de IA Unificada para Análise Forex
Combina análise histórica, sentimento e probabilidades com parâmetros separados
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class AIAnalysisComponents:
    """Componentes individuais da análise de IA"""
    historical_analysis: Dict
    sentiment_analysis: Dict
    probability_analysis: Dict
    technical_analysis: Dict
    unified_interpretation: Dict

class AIUnifiedService:
    """Serviço unificado de análise de IA com parâmetros separados"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Import temporal parameters
        try:
            from config.settings import TEMPORAL_AI_PARAMETERS, PAIR_AI_ADJUSTMENTS
            self.temporal_params = TEMPORAL_AI_PARAMETERS
            self.pair_adjustments = PAIR_AI_ADJUSTMENTS
        except ImportError:
            self.temporal_params = {}
            self.pair_adjustments = {}
    
    def analyze_historical_patterns(self, price_data: pd.DataFrame, profile: Dict, 
                                   horizon: str = '1 Hora', pair: str = 'EUR/USD') -> Dict:
        """
        Análise de padrões históricos com parâmetros específicos
        """
        try:
            # Obter parâmetros específicos para o horizonte temporal
            temporal_params = self.temporal_params.get(horizon, {})
            pair_adjustments = self.pair_adjustments.get(pair, {'volatility_multiplier': 1.0, 'prediction_confidence_boost': 1.0})
            
            # Ajustar períodos baseado na estratégia temporal
            periods = temporal_params.get('ai_historical_periods', profile['ai_historical_periods'])
            confidence_threshold = profile['ai_confidence_threshold'] * pair_adjustments['prediction_confidence_boost']
            
            # Parâmetros específicos do horizonte
            volatility_sensitivity = temporal_params.get('ai_volatility_sensitivity', 1.0)
            momentum_periods = temporal_params.get('ai_momentum_periods', 10)
            trend_confirmation = temporal_params.get('ai_trend_confirmation', 5)
            
            # Calcular indicadores técnicos históricos
            prices = price_data['close'].values
            if len(prices) < periods:
                periods = len(prices) - 1
            
            # Análise de tendência histórica
            recent_prices = prices[-periods:]
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            trend_strength = abs(trend_slope) / np.std(recent_prices) if np.std(recent_prices) > 0 else 0
            
            # Análise de volatilidade histórica ajustada por par e horizonte
            returns = np.diff(prices) / prices[:-1]
            base_volatility = np.std(returns[-periods:]) if len(returns) >= periods else np.std(returns)
            historical_volatility = base_volatility * pair_adjustments['volatility_multiplier'] * volatility_sensitivity
            
            # Padrões de reversão/continuação
            price_changes = np.diff(recent_prices)
            bullish_periods = np.sum(price_changes > 0)
            bearish_periods = np.sum(price_changes < 0)
            
            # Momentum histórico ajustado para o horizonte temporal
            momentum_range = min(momentum_periods, len(recent_prices))
            if momentum_range > 1:
                momentum = (recent_prices[-1] - recent_prices[-momentum_range]) / recent_prices[-momentum_range] if recent_prices[-momentum_range] != 0 else 0
            else:
                momentum = 0
            
            # Níveis de suporte e resistência históricos
            highs = []
            lows = []
            for i in range(2, len(recent_prices) - 2):
                if (recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1] and
                    recent_prices[i] > recent_prices[i-2] and recent_prices[i] > recent_prices[i+2]):
                    highs.append(recent_prices[i])
                elif (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1] and
                      recent_prices[i] < recent_prices[i-2] and recent_prices[i] < recent_prices[i+2]):
                    lows.append(recent_prices[i])
            
            resistance_level = np.mean(highs) if highs else recent_prices[-1] * 1.01
            support_level = np.mean(lows) if lows else recent_prices[-1] * 0.99
            
            # Confiança baseada na consistência dos padrões e parâmetros temporais
            trend_consistency = min(bullish_periods / len(price_changes), bearish_periods / len(price_changes)) if len(price_changes) > 0 else 0.5
            pattern_consistency = min(abs(trend_strength), 1.0) * profile['ai_volatility_adjustment']
            
            # Ajustar confiança baseado na confirmação de tendência específica do horizonte
            confirmation_strength = 1.0
            if len(price_changes) >= trend_confirmation:
                recent_changes = price_changes[-trend_confirmation:]
                same_direction = np.sum(recent_changes > 0) if trend_slope > 0 else np.sum(recent_changes < 0)
                confirmation_strength = same_direction / trend_confirmation
            
            historical_confidence = min(
                pattern_consistency * confirmation_strength * pair_adjustments['prediction_confidence_boost'],
                confidence_threshold
            )
            
            return {
                'trend_direction': 'bullish' if trend_slope > 0 else 'bearish',
                'trend_strength': trend_strength,
                'historical_volatility': historical_volatility,
                'momentum': momentum,
                'bullish_periods_ratio': bullish_periods / len(price_changes) if len(price_changes) > 0 else 0.5,
                'bearish_periods_ratio': bearish_periods / len(price_changes) if len(price_changes) > 0 else 0.5,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'confidence': historical_confidence,
                'periods_analyzed': periods,
                'pattern_consistency': pattern_consistency,
                'temporal_horizon': horizon,
                'pair_adjustment': pair_adjustments['prediction_confidence_boost'],
                'volatility_adjustment': volatility_sensitivity,
                'trend_confirmation_strength': confirmation_strength,
                'momentum_periods_used': momentum_range
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise histórica: {e}")
            return {
                'trend_direction': 'neutral',
                'trend_strength': 0.0,
                'historical_volatility': 0.02,
                'momentum': 0.0,
                'bullish_periods_ratio': 0.5,
                'bearish_periods_ratio': 0.5,
                'support_level': price_data['close'].iloc[-1] * 0.99,
                'resistance_level': price_data['close'].iloc[-1] * 1.01,
                'confidence': 0.3,
                'periods_analyzed': 10,
                'pattern_consistency': 0.3
            }
    
    def analyze_market_sentiment(self, sentiment_data: Dict, profile: Dict, 
                                horizon: str = '1 Hora', pair: str = 'EUR/USD') -> Dict:
        """
        Análise de sentimento de mercado com parâmetros específicos
        """
        try:
            # Obter parâmetros específicos para o horizonte temporal
            temporal_params = self.temporal_params.get(horizon, {})
            pair_adjustments = self.pair_adjustments.get(pair, {'volatility_multiplier': 1.0, 'prediction_confidence_boost': 1.0})
            
            # Ajustar sensibilidade e decay baseado no horizonte
            sensitivity = profile['ai_sentiment_sensitivity']
            confidence_threshold = profile['ai_confidence_threshold'] * pair_adjustments['prediction_confidence_boost']
            sentiment_decay = temporal_params.get('ai_sentiment_decay', 0.7)
            news_impact_weight = temporal_params.get('ai_news_impact_weight', 0.6)
            
            # Processar dados de sentimento
            if not sentiment_data or 'overall_sentiment' not in sentiment_data:
                return {
                    'sentiment_score': 0.0,
                    'sentiment_direction': 'neutral',
                    'market_mood': 'uncertain',
                    'confidence': 0.3,
                    'news_impact': 'low',
                    'sentiment_strength': 0.0
                }
            
            # Extrair métricas de sentimento
            overall_sentiment = sentiment_data.get('overall_sentiment', 0.0)
            news_count = sentiment_data.get('news_count', 0)
            sentiment_consistency = sentiment_data.get('sentiment_consistency', 0.5)
            
            # Ajustar sentimento pela sensibilidade do perfil e decay temporal
            time_adjusted_sentiment = overall_sentiment * sentiment_decay  # Aplicar decay temporal
            adjusted_sentiment = time_adjusted_sentiment * sensitivity * news_impact_weight
            
            # Determinar direção do sentimento
            if adjusted_sentiment > 0.1:
                sentiment_direction = 'bullish'
                market_mood = 'optimistic'
            elif adjusted_sentiment < -0.1:
                sentiment_direction = 'bearish'
                market_mood = 'pessimistic'
            else:
                sentiment_direction = 'neutral'
                market_mood = 'uncertain'
            
            # Calcular força do sentimento
            sentiment_strength = min(abs(adjusted_sentiment), 1.0)
            
            # Determinar impacto das notícias
            if news_count > 10:
                news_impact = 'high'
            elif news_count > 5:
                news_impact = 'medium'
            else:
                news_impact = 'low'
            
            # Confiança baseada na consistência e quantidade de dados
            news_factor = min(news_count / 10, 1.0)  # Máximo 10 notícias para confiança total
            sentiment_confidence = min(
                sentiment_consistency * news_factor * sentiment_strength,
                confidence_threshold
            )
            
            return {
                'sentiment_score': adjusted_sentiment,
                'sentiment_direction': sentiment_direction,
                'market_mood': market_mood,
                'confidence': sentiment_confidence,
                'news_impact': news_impact,
                'sentiment_strength': sentiment_strength,
                'news_count': news_count,
                'consistency': sentiment_consistency
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de sentimento: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_direction': 'neutral',
                'market_mood': 'uncertain',
                'confidence': 0.3,
                'news_impact': 'low',
                'sentiment_strength': 0.0,
                'news_count': 0,
                'consistency': 0.5
            }
    
    def calculate_probability_metrics(self, prediction_data: Dict, profile: Dict,
                                    horizon: str = '1 Hora', pair: str = 'EUR/USD') -> Dict:
        """
        Cálculo de métricas de probabilidade com parâmetros específicos
        """
        try:
            # Obter parâmetros específicos para o horizonte temporal
            temporal_params = self.temporal_params.get(horizon, {})
            pair_adjustments = self.pair_adjustments.get(pair, {'volatility_multiplier': 1.0, 'prediction_confidence_boost': 1.0})
            
            # Ajustar thresholds baseado no horizonte e par
            confidence_threshold = profile['ai_confidence_threshold'] * pair_adjustments['prediction_confidence_boost']
            trend_strength_min = profile['ai_trend_strength_min']
            probability_threshold = temporal_params.get('ai_probability_threshold', 0.7)
            technical_weight = temporal_params.get('ai_technical_weight', 0.7)
            
            # Extrair dados de previsão
            predicted_price = prediction_data.get('predicted_price', 0.0)
            current_price = prediction_data.get('current_price', 0.0)
            model_confidence = prediction_data.get('confidence', 0.5)
            
            if current_price == 0:
                return {
                    'direction_probability': 0.5,
                    'magnitude_probability': 0.3,
                    'success_probability': 0.3,
                    'risk_probability': 0.7,
                    'confidence': 0.3,
                    'prediction_strength': 0.0
                }
            
            # Calcular direção e magnitude da previsão
            price_change = (predicted_price - current_price) / current_price
            prediction_strength = abs(price_change)
            
            # Probabilidade de direção ajustada por horizonte temporal e par
            base_threshold = trend_strength_min * pair_adjustments['volatility_multiplier']
            if prediction_strength > base_threshold:
                # Ajustar probabilidade baseado no peso técnico do horizonte
                raw_probability = min(0.5 + (prediction_strength * 2), 0.85)
                direction_probability = raw_probability * technical_weight + (0.5 * (1 - technical_weight))
            else:
                direction_probability = 0.5  # Neutro se previsão muito fraca
            
            # Probabilidade de magnitude (baseada na confiança do modelo)
            magnitude_probability = model_confidence * 0.8  # Máximo 80%
            
            # Probabilidade de sucesso (combinação das anteriores)
            success_probability = (direction_probability * 0.6 + magnitude_probability * 0.4)
            
            # Probabilidade de risco (inverso do sucesso)
            risk_probability = 1.0 - success_probability
            
            # Confiança ajustada pelo limiar do perfil e threshold temporal
            base_confidence = (direction_probability + magnitude_probability) / 2
            probability_confidence = min(
                base_confidence * (probability_threshold / 0.7),  # Normalizar para threshold base
                confidence_threshold
            )
            
            return {
                'direction_probability': direction_probability,
                'magnitude_probability': magnitude_probability,
                'success_probability': success_probability,
                'risk_probability': risk_probability,
                'confidence': probability_confidence,
                'prediction_strength': prediction_strength,
                'price_change_expected': price_change,
                'trend_strength_sufficient': prediction_strength >= trend_strength_min
            }
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de probabilidades: {e}")
            return {
                'direction_probability': 0.5,
                'magnitude_probability': 0.3,
                'success_probability': 0.3,
                'risk_probability': 0.7,
                'confidence': 0.3,
                'prediction_strength': 0.0,
                'price_change_expected': 0.0,
                'trend_strength_sufficient': False
            }
    
    def create_unified_interpretation(self, 
                                    historical: Dict, 
                                    sentiment: Dict, 
                                    probability: Dict,
                                    profile: Dict) -> Dict:
        """
        Interpretação unificada usando IA para combinar todas as análises
        """
        try:
            # Pesos dos componentes baseados no perfil
            hist_weight = profile['ai_historical_weight']
            sent_weight = profile['ai_sentiment_weight']
            prob_weight = profile['ai_probability_weight']
            
            # Normalizar pesos
            total_weight = hist_weight + sent_weight + prob_weight
            hist_weight /= total_weight
            sent_weight /= total_weight
            prob_weight /= total_weight
            
            # Combinar direções (bullish = 1, bearish = -1, neutral = 0)
            historical_direction = 1 if historical['trend_direction'] == 'bullish' else -1 if historical['trend_direction'] == 'bearish' else 0
            sentiment_direction = 1 if sentiment['sentiment_direction'] == 'bullish' else -1 if sentiment['sentiment_direction'] == 'bearish' else 0
            probability_direction = 1 if probability['direction_probability'] > 0.5 else -1
            
            # Direção unificada ponderada
            unified_direction_score = (
                historical_direction * hist_weight * historical['confidence'] +
                sentiment_direction * sent_weight * sentiment['confidence'] +
                probability_direction * prob_weight * probability['confidence']
            )
            
            # Determinar direção final
            if unified_direction_score > 0.1:
                unified_direction = 'bullish'
                direction_confidence = min(abs(unified_direction_score), 1.0)
            elif unified_direction_score < -0.1:
                unified_direction = 'bearish'
                direction_confidence = min(abs(unified_direction_score), 1.0)
            else:
                unified_direction = 'neutral'
                direction_confidence = 0.3
            
            # Força combinada da análise
            combined_strength = (
                historical['trend_strength'] * hist_weight +
                sentiment['sentiment_strength'] * sent_weight +
                probability['prediction_strength'] * prob_weight
            )
            
            # Confiança unificada
            unified_confidence = (
                historical['confidence'] * hist_weight +
                sentiment['confidence'] * sent_weight +
                probability['confidence'] * prob_weight
            )
            
            # Probabilidade de sucesso ajustada
            success_probability = (
                (historical['pattern_consistency'] if 'pattern_consistency' in historical else 0.5) * hist_weight +
                sentiment['sentiment_strength'] * sent_weight +
                probability['success_probability'] * prob_weight
            )
            
            # Análise de consenso (quantos componentes concordam)
            consensus_count = 0
            if historical_direction != 0 and historical_direction == (1 if unified_direction == 'bullish' else -1):
                consensus_count += 1
            if sentiment_direction != 0 and sentiment_direction == (1 if unified_direction == 'bullish' else -1):
                consensus_count += 1
            if probability_direction == (1 if unified_direction == 'bullish' else -1):
                consensus_count += 1
            
            consensus_strength = consensus_count / 3  # 0 a 1
            
            # Recomendação final baseada na interpretação unificada
            if unified_confidence > 0.7 and direction_confidence > 0.6 and consensus_strength >= 0.67:
                recommendation = 'strong_' + unified_direction
                recommendation_confidence = min(unified_confidence * 1.1, 0.95)
            elif unified_confidence > 0.5 and direction_confidence > 0.4:
                recommendation = 'moderate_' + unified_direction
                recommendation_confidence = unified_confidence * 0.9
            else:
                recommendation = 'weak_' + unified_direction if unified_direction != 'neutral' else 'hold'
                recommendation_confidence = unified_confidence * 0.7
            
            return {
                'unified_direction': unified_direction,
                'direction_confidence': direction_confidence,
                'combined_strength': combined_strength,
                'unified_confidence': unified_confidence,
                'success_probability': success_probability,
                'consensus_strength': consensus_strength,
                'consensus_count': consensus_count,
                'recommendation': recommendation,
                'recommendation_confidence': recommendation_confidence,
                'component_weights': {
                    'historical': hist_weight,
                    'sentiment': sent_weight,
                    'probability': prob_weight
                },
                'component_agreement': {
                    'historical_sentiment': historical_direction == sentiment_direction,
                    'historical_probability': historical_direction == probability_direction,
                    'sentiment_probability': sentiment_direction == probability_direction
                },
                'ai_interpretation': self._generate_ai_interpretation(
                    unified_direction, direction_confidence, consensus_strength, 
                    combined_strength, recommendation
                )
            }
            
        except Exception as e:
            self.logger.error(f"Erro na interpretação unificada: {e}")
            return {
                'unified_direction': 'neutral',
                'direction_confidence': 0.3,
                'combined_strength': 0.3,
                'unified_confidence': 0.3,
                'success_probability': 0.3,
                'consensus_strength': 0.0,
                'consensus_count': 0,
                'recommendation': 'hold',
                'recommendation_confidence': 0.3,
                'component_weights': {'historical': 0.33, 'sentiment': 0.33, 'probability': 0.34},
                'component_agreement': {'historical_sentiment': False, 'historical_probability': False, 'sentiment_probability': False},
                'ai_interpretation': 'Análise inconclusiva - dados insuficientes ou conflitantes.'
            }
    
    def _generate_ai_interpretation(self, direction: str, confidence: float, consensus: float, strength: float, recommendation: str) -> str:
        """
        Gerar interpretação textual da IA baseada nos resultados
        """
        # Determinar força da análise
        if strength > 0.7:
            strength_desc = "muito forte"
        elif strength > 0.5:
            strength_desc = "forte"
        elif strength > 0.3:
            strength_desc = "moderada"
        else:
            strength_desc = "fraca"
        
        # Determinar consenso
        if consensus >= 0.67:
            consensus_desc = "alto consenso entre os indicadores"
        elif consensus >= 0.33:
            consensus_desc = "consenso parcial"
        else:
            consensus_desc = "baixo consenso - sinais conflitantes"
        
        # Determinar confiança
        if confidence > 0.7:
            confidence_desc = "alta confiança"
        elif confidence > 0.5:
            confidence_desc = "confiança moderada"
        else:
            confidence_desc = "baixa confiança"
        
        # Gerar interpretação
        direction_text = "alta" if direction == "bullish" else "baixa" if direction == "bearish" else "lateral"
        
        interpretation = f"A IA identifica tendência {direction_text} com {strength_desc} intensidade. "
        interpretation += f"Há {consensus_desc} e {confidence_desc} na análise. "
        
        # Adicionar recomendação específica
        if "strong" in recommendation:
            interpretation += "Recomendação: Posição forte justificada pelos múltiplos indicadores alinhados."
        elif "moderate" in recommendation:
            interpretation += "Recomendação: Posição moderada com gestão de risco adequada."
        elif "weak" in recommendation:
            interpretation += "Recomendação: Posição cautelosa devido à incerteza nos sinais."
        else:
            interpretation += "Recomendação: Aguardar melhor definição do mercado."
        
        return interpretation
    
    def run_unified_analysis(self, 
                           price_data: pd.DataFrame,
                           sentiment_data: Dict,
                           prediction_data: Dict,
                           profile: Dict,
                           horizon: str = '1 Hora',
                           pair: str = 'EUR/USD') -> AIAnalysisComponents:
        """
        Executar análise unificada completa
        """
        try:
            # Executar cada componente de análise separadamente com parâmetros temporais
            historical = self.analyze_historical_patterns(price_data, profile, horizon, pair)
            sentiment = self.analyze_market_sentiment(sentiment_data, profile, horizon, pair)
            probability = self.calculate_probability_metrics(prediction_data, profile, horizon, pair)
            
            # Análise técnica (simplificada para este contexto)
            technical = {
                'support_levels': [historical.get('support_level', 0)],
                'resistance_levels': [historical.get('resistance_level', 0)],
                'trend_direction': historical.get('trend_direction', 'neutral'),
                'volatility': historical.get('historical_volatility', 0.02)
            }
            
            # Criar interpretação unificada
            unified = self.create_unified_interpretation(historical, sentiment, probability, profile)
            
            return AIAnalysisComponents(
                historical_analysis=historical,
                sentiment_analysis=sentiment,
                probability_analysis=probability,
                technical_analysis=technical,
                unified_interpretation=unified
            )
            
        except Exception as e:
            self.logger.error(f"Erro na análise unificada: {e}")
            # Retornar análise padrão em caso de erro
            return AIAnalysisComponents(
                historical_analysis={'trend_direction': 'neutral', 'confidence': 0.3},
                sentiment_analysis={'sentiment_direction': 'neutral', 'confidence': 0.3},
                probability_analysis={'success_probability': 0.3, 'confidence': 0.3},
                technical_analysis={'trend_direction': 'neutral', 'volatility': 0.02},
                unified_interpretation={'unified_direction': 'neutral', 'recommendation': 'hold', 'unified_confidence': 0.3}
            )