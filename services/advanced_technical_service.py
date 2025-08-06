import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from services.indicators import TechnicalIndicators

class AdvancedTechnicalService:
    """
    Análise técnica avançada com indicadores específicos:
    - RSI (período 14, níveis 50/70/30)
    - Médias móveis (SMA 50/200) 
    - ADX para força de tendência
    - Padrões de confirmação de tendência
    """
    
    # Configurações de indicadores
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_NEUTRAL = 50
    RSI_OVERBOUGHT = 70
    
    SMA_SHORT = 50
    SMA_LONG = 200
    
    ADX_PERIOD = 14
    ADX_STRONG_TREND = 25
    ADX_VERY_STRONG = 50
    
    @staticmethod
    def analyze_trend_confirmation(df: pd.DataFrame) -> Dict:
        """
        Análise completa de confirmação de tendência usando RSI, SMA e ADX
        
        Args:
            df: DataFrame com dados OHLC
            
        Returns:
            Dict com análise completa de tendência
        """
        try:
            if df is None or df.empty or len(df) < max(AdvancedTechnicalService.SMA_LONG, 30):
                return AdvancedTechnicalService._get_default_technical_analysis()
            
            # Preparar dados
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # 1. Análise RSI
            rsi_analysis = AdvancedTechnicalService._analyze_rsi(close)
            
            # 2. Análise de Médias Móveis
            sma_analysis = AdvancedTechnicalService._analyze_sma_crossover(close)
            
            # 3. Análise ADX (força de tendência)
            adx_analysis = AdvancedTechnicalService._analyze_adx_strength(high, low, close)
            
            # 4. Análise de suporte e resistência
            support_resistance = AdvancedTechnicalService._analyze_support_resistance(df)
            
            # 5. Padrões de velas
            candlestick_patterns = AdvancedTechnicalService._analyze_candlestick_patterns(df)
            
            # 6. Confirmação de tendência (consenso)
            trend_confirmation = AdvancedTechnicalService._calculate_trend_consensus(
                rsi_analysis, sma_analysis, adx_analysis
            )
            
            # 7. Sinais de entrada/saída
            entry_exit_signals = AdvancedTechnicalService._generate_entry_exit_signals(
                rsi_analysis, sma_analysis, adx_analysis, close[-1]
            )
            
            return {
                'rsi_analysis': rsi_analysis,
                'sma_analysis': sma_analysis, 
                'adx_analysis': adx_analysis,
                'support_resistance': support_resistance,
                'candlestick_patterns': candlestick_patterns,
                'trend_confirmation': trend_confirmation,
                'entry_exit_signals': entry_exit_signals,
                'overall_signal': AdvancedTechnicalService._determine_overall_signal(trend_confirmation),
                'confidence_level': AdvancedTechnicalService._calculate_signal_confidence(
                    rsi_analysis, sma_analysis, adx_analysis
                )
            }
            
        except Exception as e:
            return AdvancedTechnicalService._get_default_technical_analysis()
    
    @staticmethod
    def _analyze_rsi(close: np.ndarray) -> Dict:
        """Análise detalhada do RSI com níveis específicos"""
        try:
            rsi = TechnicalIndicators.rsi(pd.Series(close), AdvancedTechnicalService.RSI_PERIOD).values
            current_rsi = rsi[-1]
            previous_rsi = rsi[-2]
            
            # Determinar zona do RSI
            if current_rsi >= AdvancedTechnicalService.RSI_OVERBOUGHT:
                rsi_zone = 'SOBRECOMPRADO'
                rsi_signal = 'VENDA'
            elif current_rsi <= AdvancedTechnicalService.RSI_OVERSOLD:
                rsi_zone = 'SOBREVENDIDO' 
                rsi_signal = 'COMPRA'
            else:
                rsi_zone = 'NEUTRO'
                if current_rsi > AdvancedTechnicalService.RSI_NEUTRAL:
                    rsi_signal = 'LEVE_ALTA'
                else:
                    rsi_signal = 'LEVE_BAIXA'
            
            # Divergências RSI
            rsi_divergence = AdvancedTechnicalService._detect_rsi_divergence(rsi, close)
            
            # Momentum RSI
            rsi_momentum = 'CRESCENTE' if current_rsi > previous_rsi else 'DECRESCENTE'
            
            # Força do sinal RSI
            if rsi_zone in ['SOBRECOMPRADO', 'SOBREVENDIDO']:
                rsi_strength = 'FORTE'
            elif abs(current_rsi - 50) > 15:
                rsi_strength = 'MODERADA'
            else:
                rsi_strength = 'FRACA'
            
            return {
                'current_rsi': round(current_rsi, 2),
                'previous_rsi': round(previous_rsi, 2),
                'rsi_zone': rsi_zone,
                'rsi_signal': rsi_signal,
                'rsi_momentum': rsi_momentum,
                'rsi_strength': rsi_strength,
                'rsi_divergence': rsi_divergence,
                'oversold_level': AdvancedTechnicalService.RSI_OVERSOLD,
                'overbought_level': AdvancedTechnicalService.RSI_OVERBOUGHT
            }
            
        except Exception:
            return {
                'current_rsi': 50,
                'rsi_zone': 'NEUTRO',
                'rsi_signal': 'NEUTRO',
                'rsi_strength': 'INDEFINIDA'
            }
    
    @staticmethod 
    def _analyze_sma_crossover(close: np.ndarray) -> Dict:
        """Análise de crossover das médias móveis SMA 50/200"""
        try:
            sma_50 = TechnicalIndicators.sma(pd.Series(close), AdvancedTechnicalService.SMA_SHORT).values
            sma_200 = TechnicalIndicators.sma(pd.Series(close), AdvancedTechnicalService.SMA_LONG).values
            
            current_sma_50 = sma_50[-1]
            current_sma_200 = sma_200[-1]
            current_price = close[-1]
            
            # Previous values for crossover detection
            prev_sma_50 = sma_50[-2]
            prev_sma_200 = sma_200[-2]
            
            # Detectar crossovers
            golden_cross = (current_sma_50 > current_sma_200 and 
                           prev_sma_50 <= prev_sma_200)
            death_cross = (current_sma_50 < current_sma_200 and
                          prev_sma_50 >= prev_sma_200)
            
            # Determinar tendência baseada nas SMAs
            if current_sma_50 > current_sma_200:
                if golden_cross:
                    sma_trend = 'ALTA_FORTE'  # Golden cross recente
                    sma_signal = 'COMPRA_FORTE'
                else:
                    sma_trend = 'ALTA'
                    sma_signal = 'COMPRA'
            elif current_sma_50 < current_sma_200:
                if death_cross:
                    sma_trend = 'BAIXA_FORTE'  # Death cross recente
                    sma_signal = 'VENDA_FORTE'
                else:
                    sma_trend = 'BAIXA'
                    sma_signal = 'VENDA'
            else:
                sma_trend = 'LATERAL'
                sma_signal = 'NEUTRO'
            
            # Posição do preço em relação às médias
            if current_price > current_sma_50 > current_sma_200:
                price_position = 'ACIMA_AMBAS'
            elif current_price < current_sma_50 < current_sma_200:
                price_position = 'ABAIXO_AMBAS'
            elif current_sma_200 < current_price < current_sma_50:
                price_position = 'ENTRE_MEDIAS_BAIXA'
            elif current_sma_50 < current_price < current_sma_200:
                price_position = 'ENTRE_MEDIAS_ALTA'
            else:
                price_position = 'INDEFINIDA'
            
            # Distância percentual das médias
            sma_distance_pct = abs(current_sma_50 - current_sma_200) / current_sma_200 * 100
            
            return {
                'current_sma_50': round(current_sma_50, 5),
                'current_sma_200': round(current_sma_200, 5),
                'sma_trend': sma_trend,
                'sma_signal': sma_signal,
                'golden_cross': golden_cross,
                'death_cross': death_cross,
                'price_position': price_position,
                'sma_distance_pct': round(sma_distance_pct, 2),
                'trend_strength': 'FORTE' if sma_distance_pct > 1 else 'MODERADA' if sma_distance_pct > 0.5 else 'FRACA'
            }
            
        except Exception:
            return {
                'sma_trend': 'INDEFINIDA',
                'sma_signal': 'NEUTRO',
                'golden_cross': False,
                'death_cross': False
            }
    
    @staticmethod
    def _analyze_adx_strength(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Análise da força de tendência usando ADX"""
        try:
            # Usar implementação própria do ADX
            adx_data = AdvancedTechnicalService._calculate_adx(high, low, close, AdvancedTechnicalService.ADX_PERIOD)
            adx = adx_data['adx']
            plus_di = adx_data['plus_di'] 
            minus_di = adx_data['minus_di']
            
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # Determinar força da tendência
            if current_adx >= AdvancedTechnicalService.ADX_VERY_STRONG:
                adx_strength = 'MUITO_FORTE'
            elif current_adx >= AdvancedTechnicalService.ADX_STRONG_TREND:
                adx_strength = 'FORTE'
            else:
                adx_strength = 'FRACA'
            
            # Direção da tendência baseada em DI
            if current_plus_di > current_minus_di:
                adx_direction = 'ALTA'
                adx_signal = 'COMPRA' if adx_strength != 'FRACA' else 'NEUTRO'
            elif current_minus_di > current_plus_di:
                adx_direction = 'BAIXA'
                adx_signal = 'VENDA' if adx_strength != 'FRACA' else 'NEUTRO'
            else:
                adx_direction = 'LATERAL'
                adx_signal = 'NEUTRO'
            
            # Momentum do ADX
            prev_adx = adx[-2]
            adx_momentum = 'CRESCENTE' if current_adx > prev_adx else 'DECRESCENTE'
            
            # Qualidade da tendência
            di_separation = abs(current_plus_di - current_minus_di)
            if di_separation > 20:
                trend_quality = 'EXCELENTE'
            elif di_separation > 10:
                trend_quality = 'BOA'
            else:
                trend_quality = 'RUIM'
            
            return {
                'current_adx': round(current_adx, 2),
                'current_plus_di': round(current_plus_di, 2),
                'current_minus_di': round(current_minus_di, 2),
                'adx_strength': adx_strength,
                'adx_direction': adx_direction,
                'adx_signal': adx_signal,
                'adx_momentum': adx_momentum,
                'trend_quality': trend_quality,
                'di_separation': round(di_separation, 2)
            }
            
        except Exception:
            return {
                'current_adx': 20,
                'adx_strength': 'INDEFINIDA',
                'adx_direction': 'LATERAL',
                'adx_signal': 'NEUTRO'
            }
    
    @staticmethod
    def _analyze_support_resistance(df: pd.DataFrame) -> Dict:
        """Análise de níveis de suporte e resistência"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calcular pivots para suporte/resistência
            recent_data = df.tail(50)  # Últimas 50 velas
            
            # Resistência: máximas recentes
            resistance_levels = []
            for i in range(2, len(recent_data)-2):
                if (recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and
                    recent_data.iloc[i]['high'] > recent_data.iloc[i-2]['high'] and
                    recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high'] and
                    recent_data.iloc[i]['high'] > recent_data.iloc[i+2]['high']):
                    resistance_levels.append(recent_data.iloc[i]['high'])
            
            # Suporte: mínimas recentes  
            support_levels = []
            for i in range(2, len(recent_data)-2):
                if (recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and
                    recent_data.iloc[i]['low'] < recent_data.iloc[i-2]['low'] and
                    recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low'] and
                    recent_data.iloc[i]['low'] < recent_data.iloc[i+2]['low']):
                    support_levels.append(recent_data.iloc[i]['low'])
            
            # Selecionar níveis mais relevantes
            current_price = close[-1]
            
            # Resistência mais próxima acima
            resistance_above = [r for r in resistance_levels if r > current_price]
            nearest_resistance = min(resistance_above) if resistance_above else None
            
            # Suporte mais próximo abaixo
            support_below = [s for s in support_levels if s < current_price]
            nearest_support = max(support_below) if support_below else None
            
            # Distâncias percentuais
            resistance_distance = None
            support_distance = None
            
            if nearest_resistance:
                resistance_distance = (nearest_resistance - current_price) / current_price * 100
                
            if nearest_support:
                support_distance = (current_price - nearest_support) / current_price * 100
            
            return {
                'nearest_resistance': round(nearest_resistance, 5) if nearest_resistance else None,
                'nearest_support': round(nearest_support, 5) if nearest_support else None,
                'resistance_distance_pct': round(resistance_distance, 2) if resistance_distance else None,
                'support_distance_pct': round(support_distance, 2) if support_distance else None,
                'total_resistance_levels': len(resistance_levels),
                'total_support_levels': len(support_levels),
                'price_position': AdvancedTechnicalService._determine_price_position(
                    current_price, nearest_support, nearest_resistance
                )
            }
            
        except Exception:
            return {
                'nearest_resistance': None,
                'nearest_support': None,
                'price_position': 'INDEFINIDA'
            }
    
    @staticmethod
    def _analyze_candlestick_patterns(df: pd.DataFrame) -> Dict:
        """Análise de padrões de candlestick"""
        try:
            open_prices = df['open'].values
            high = df['high'].values  
            low = df['low'].values
            close = df['close'].values
            
            # Implementação própria de padrões de candlestick
            patterns = AdvancedTechnicalService._detect_candlestick_patterns(open_prices, high, low, close)
            
            # Usar padrões detectados
            recent_patterns = patterns
            
            # Determinar sinal geral dos padrões
            bullish_patterns = ['HAMMER', 'ENGULFING_BULLISH']
            bearish_patterns = ['SHOOTING_STAR', 'HANGING_MAN']
            neutral_patterns = ['DOJI']
            
            bullish_count = sum(1 for p in recent_patterns if p in bullish_patterns)
            bearish_count = sum(1 for p in recent_patterns if p in bearish_patterns)
            
            if bullish_count > bearish_count:
                pattern_signal = 'ALTA'
            elif bearish_count > bullish_count:
                pattern_signal = 'BAIXA'
            else:
                pattern_signal = 'NEUTRO'
            
            return {
                'recent_patterns': recent_patterns,
                'pattern_signal': pattern_signal,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'pattern_strength': 'FORTE' if max(bullish_count, bearish_count) >= 2 else 'FRACA'
            }
            
        except Exception:
            return {
                'recent_patterns': [],
                'pattern_signal': 'NEUTRO',
                'pattern_strength': 'INDEFINIDA'
            }
    
    @staticmethod
    def _calculate_trend_consensus(rsi_analysis: Dict, sma_analysis: Dict, adx_analysis: Dict) -> Dict:
        """Calcula consenso de tendência entre indicadores"""
        
        # Converter sinais em pontos
        signal_points = 0
        total_weight = 0
        
        # RSI (peso 30%)
        rsi_signal = rsi_analysis.get('rsi_signal', 'NEUTRO')
        if rsi_signal == 'COMPRA':
            signal_points += 30
        elif rsi_signal == 'VENDA':
            signal_points -= 30
        elif rsi_signal == 'LEVE_ALTA':
            signal_points += 15
        elif rsi_signal == 'LEVE_BAIXA':
            signal_points -= 15
        total_weight += 30
        
        # SMA (peso 40%)
        sma_signal = sma_analysis.get('sma_signal', 'NEUTRO')
        if 'COMPRA' in sma_signal:
            points = 40 if 'FORTE' in sma_signal else 30
            signal_points += points
        elif 'VENDA' in sma_signal:
            points = 40 if 'FORTE' in sma_signal else 30
            signal_points -= points
        total_weight += 40
        
        # ADX (peso 30%)
        adx_signal = adx_analysis.get('adx_signal', 'NEUTRO')
        adx_strength = adx_analysis.get('adx_strength', 'FRACA')
        
        if adx_signal == 'COMPRA':
            if adx_strength == 'MUITO_FORTE':
                signal_points += 30
            elif adx_strength == 'FORTE':
                signal_points += 20
            else:
                signal_points += 10
        elif adx_signal == 'VENDA':
            if adx_strength == 'MUITO_FORTE':
                signal_points -= 30
            elif adx_strength == 'FORTE':
                signal_points -= 20
            else:
                signal_points -= 10
        total_weight += 30
        
        # Calcular score final (-100 a +100)
        consensus_score = signal_points / total_weight * 100 if total_weight > 0 else 0
        
        # Determinar consenso
        if consensus_score >= 60:
            consensus = 'COMPRA_FORTE'
            direction = 'ALTA'
        elif consensus_score >= 30:
            consensus = 'COMPRA'
            direction = 'ALTA'
        elif consensus_score <= -60:
            consensus = 'VENDA_FORTE'
            direction = 'BAIXA'
        elif consensus_score <= -30:
            consensus = 'VENDA'
            direction = 'BAIXA'
        else:
            consensus = 'NEUTRO'
            direction = 'LATERAL'
        
        # Nível de acordo entre indicadores
        agreement_indicators = []
        if rsi_signal in ['COMPRA', 'LEVE_ALTA'] and 'COMPRA' in sma_signal and adx_signal == 'COMPRA':
            agreement_level = 'ALTO'
        elif rsi_signal in ['VENDA', 'LEVE_BAIXA'] and 'VENDA' in sma_signal and adx_signal == 'VENDA':
            agreement_level = 'ALTO'
        elif abs(consensus_score) >= 30:
            agreement_level = 'MODERADO'
        else:
            agreement_level = 'BAIXO'
        
        return {
            'consensus': consensus,
            'direction': direction,
            'consensus_score': round(consensus_score, 1),
            'agreement_level': agreement_level,
            'supporting_indicators': AdvancedTechnicalService._count_supporting_indicators(
                rsi_analysis, sma_analysis, adx_analysis, direction
            )
        }
    
    @staticmethod
    def _generate_entry_exit_signals(rsi_analysis: Dict, sma_analysis: Dict, 
                                   adx_analysis: Dict, current_price: float) -> Dict:
        """Gera sinais específicos de entrada e saída"""
        
        signals = {
            'entry_signals': [],
            'exit_signals': [],
            'stop_loss_suggestion': None,
            'take_profit_suggestion': None
        }
        
        # Sinais de entrada baseados em confluência
        rsi_zone = rsi_analysis.get('rsi_zone', 'NEUTRO')
        sma_signal = sma_analysis.get('sma_signal', 'NEUTRO')
        adx_strength = adx_analysis.get('adx_strength', 'FRACA')
        
        # Entry signals - COMPRA
        if (rsi_zone == 'SOBREVENDIDO' and 
            'COMPRA' in sma_signal and 
            adx_strength in ['FORTE', 'MUITO_FORTE']):
            signals['entry_signals'].append({
                'type': 'COMPRA',
                'strength': 'MUITO_FORTE',
                'reason': 'RSI sobrevendido + SMA alta + ADX forte'
            })
        
        # Entry signals - VENDA  
        if (rsi_zone == 'SOBRECOMPRADO' and
            'VENDA' in sma_signal and
            adx_strength in ['FORTE', 'MUITO_FORTE']):
            signals['entry_signals'].append({
                'type': 'VENDA', 
                'strength': 'MUITO_FORTE',
                'reason': 'RSI sobrecomprado + SMA baixa + ADX forte'
            })
        
        # Exit signals baseados em RSI extremo
        if rsi_zone == 'SOBRECOMPRADO':
            signals['exit_signals'].append({
                'type': 'SAIR_COMPRAS',
                'reason': 'RSI em zona de sobrecompra'
            })
        
        if rsi_zone == 'SOBREVENDIDO':
            signals['exit_signals'].append({
                'type': 'SAIR_VENDAS',
                'reason': 'RSI em zona de sobrevenda'
            })
        
        # Sugestões de stop loss e take profit
        sma_50 = sma_analysis.get('current_sma_50')
        sma_200 = sma_analysis.get('current_sma_200')
        
        if sma_50 and sma_200:
            # Para posições de compra
            if 'COMPRA' in sma_signal:
                signals['stop_loss_suggestion'] = {
                    'level': min(sma_50, sma_200) * 0.995,  # 0.5% abaixo da SMA mais baixa
                    'type': 'DYNAMIC_SMA'
                }
                signals['take_profit_suggestion'] = {
                    'level': current_price * 1.02,  # 2% acima
                    'type': 'PERCENTAGE'
                }
            
            # Para posições de venda
            elif 'VENDA' in sma_signal:
                signals['stop_loss_suggestion'] = {
                    'level': max(sma_50, sma_200) * 1.005,  # 0.5% acima da SMA mais alta
                    'type': 'DYNAMIC_SMA'
                }
                signals['take_profit_suggestion'] = {
                    'level': current_price * 0.98,  # 2% abaixo
                    'type': 'PERCENTAGE'
                }
        
        return signals
    
    @staticmethod
    def _determine_overall_signal(trend_confirmation: Dict) -> Dict:
        """Determina sinal geral baseado na confirmação de tendência"""
        
        consensus = trend_confirmation.get('consensus', 'NEUTRO')
        agreement_level = trend_confirmation.get('agreement_level', 'BAIXO')
        consensus_score = trend_confirmation.get('consensus_score', 0)
        
        # Determinar ação recomendada
        if consensus in ['COMPRA_FORTE', 'COMPRA']:
            action = 'COMPRAR'
            confidence = 'ALTA' if agreement_level == 'ALTO' else 'MEDIA'
        elif consensus in ['VENDA_FORTE', 'VENDA']:
            action = 'VENDER' 
            confidence = 'ALTA' if agreement_level == 'ALTO' else 'MEDIA'
        else:
            action = 'AGUARDAR'
            confidence = 'BAIXA'
        
        # Probabilidade baseada no score
        probability = min(100, max(0, 50 + abs(consensus_score)))
        
        return {
            'action': action,
            'confidence': confidence,
            'probability': round(probability, 1),
            'signal_strength': 'FORTE' if abs(consensus_score) >= 60 else 'MODERADA' if abs(consensus_score) >= 30 else 'FRACA'
        }
    
    @staticmethod
    def _calculate_signal_confidence(rsi_analysis: Dict, sma_analysis: Dict, adx_analysis: Dict) -> Dict:
        """Calcula nível de confiança do sinal"""
        
        confidence_factors = []
        
        # RSI confidence
        rsi_strength = rsi_analysis.get('rsi_strength', 'INDEFINIDA')
        if rsi_strength == 'FORTE':
            confidence_factors.append(0.3)
        elif rsi_strength == 'MODERADA':
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # SMA confidence  
        trend_strength = sma_analysis.get('trend_strength', 'INDEFINIDA')
        if trend_strength == 'FORTE':
            confidence_factors.append(0.3)
        elif trend_strength == 'MODERADA':
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # ADX confidence
        adx_strength = adx_analysis.get('adx_strength', 'INDEFINIDA')
        if adx_strength == 'MUITO_FORTE':
            confidence_factors.append(0.4)
        elif adx_strength == 'FORTE':
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)
        
        # Score final de confiança
        total_confidence = sum(confidence_factors)
        
        if total_confidence >= 0.8:
            confidence_level = 'MUITO_ALTA'
        elif total_confidence >= 0.6:
            confidence_level = 'ALTA'
        elif total_confidence >= 0.4:
            confidence_level = 'MEDIA'
        else:
            confidence_level = 'BAIXA'
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': round(total_confidence, 2),
            'contributing_factors': confidence_factors
        }
    
    # Métodos auxiliares
    @staticmethod
    def _detect_rsi_divergence(rsi: np.ndarray, close: np.ndarray) -> str:
        """Detecta divergências entre RSI e preço"""
        try:
            # Simplificado: comparar últimas 10 velas
            recent_rsi = rsi[-10:]
            recent_close = close[-10:]
            
            rsi_trend = 'ALTA' if recent_rsi[-1] > recent_rsi[0] else 'BAIXA'
            price_trend = 'ALTA' if recent_close[-1] > recent_close[0] else 'BAIXA'
            
            if rsi_trend != price_trend:
                return 'DIVERGENCIA_DETECTADA'
            else:
                return 'SEM_DIVERGENCIA'
        except:
            return 'INDEFINIDA'
    
    @staticmethod
    def _determine_price_position(current_price: float, support: Optional[float], resistance: Optional[float]) -> str:
        """Determina posição do preço em relação a suporte/resistência"""
        
        if resistance and support:
            range_size = resistance - support
            position_in_range = (current_price - support) / range_size
            
            if position_in_range >= 0.8:
                return 'PROXIMO_RESISTENCIA'
            elif position_in_range <= 0.2:
                return 'PROXIMO_SUPORTE'
            else:
                return 'MEIO_RANGE'
        elif resistance:
            return 'ABAIXO_RESISTENCIA'
        elif support:
            return 'ACIMA_SUPORTE'
        else:
            return 'SEM_NIVEIS_DEFINIDOS'
    
    @staticmethod
    def _count_supporting_indicators(rsi_analysis: Dict, sma_analysis: Dict, 
                                   adx_analysis: Dict, direction: str) -> int:
        """Conta quantos indicadores suportam a direção"""
        
        count = 0
        
        if direction == 'ALTA':
            if rsi_analysis.get('rsi_signal') in ['COMPRA', 'LEVE_ALTA']:
                count += 1
            if 'COMPRA' in sma_analysis.get('sma_signal', ''):
                count += 1
            if adx_analysis.get('adx_signal') == 'COMPRA':
                count += 1
        elif direction == 'BAIXA':
            if rsi_analysis.get('rsi_signal') in ['VENDA', 'LEVE_BAIXA']:
                count += 1
            if 'VENDA' in sma_analysis.get('sma_signal', ''):
                count += 1
            if adx_analysis.get('adx_signal') == 'VENDA':
                count += 1
        
        return count
    
    @staticmethod
    def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Dict:
        """Implementação própria do ADX"""
        try:
            # True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Directional Movement
            dm_plus = high - np.roll(high, 1)
            dm_minus = np.roll(low, 1) - low
            
            dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
            dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
            
            # Smooth TR and DM
            atr = pd.Series(tr).rolling(window=period).mean().values
            plus_dm = pd.Series(dm_plus).rolling(window=period).mean().values
            minus_dm = pd.Series(dm_minus).rolling(window=period).mean().values
            
            # Calculate DI
            plus_di = 100 * (plus_dm / atr)
            minus_di = 100 * (minus_dm / atr)
            
            # Calculate DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX
            adx = pd.Series(dx).rolling(window=period).mean().values
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except Exception:
            return {
                'adx': np.full(len(high), 20),
                'plus_di': np.full(len(high), 25),
                'minus_di': np.full(len(high), 25)
            }
    
    @staticmethod
    def _detect_candlestick_patterns(open_prices: np.ndarray, high: np.ndarray, 
                                   low: np.ndarray, close: np.ndarray) -> List[str]:
        """Implementação própria de detecção de padrões de candlestick"""
        patterns = []
        
        # Analisar apenas as últimas 3 velas
        for i in range(max(1, len(close) - 3), len(close)):
            if i < 1:
                continue
                
            o_curr = open_prices[i]
            h_curr = high[i] 
            l_curr = low[i]
            c_curr = close[i]
            
            o_prev = open_prices[i-1]
            h_prev = high[i-1]
            l_prev = low[i-1]
            c_prev = close[i-1]
            
            # Body and shadows
            body_curr = abs(c_curr - o_curr)
            upper_shadow_curr = h_curr - max(o_curr, c_curr)
            lower_shadow_curr = min(o_curr, c_curr) - l_curr
            
            range_curr = h_curr - l_curr
            
            # Hammer pattern
            if (lower_shadow_curr > 2 * body_curr and 
                upper_shadow_curr < 0.1 * range_curr and
                body_curr > 0.1 * range_curr):
                patterns.append('HAMMER')
            
            # Doji pattern
            if body_curr < 0.1 * range_curr:
                patterns.append('DOJI')
            
            # Shooting Star
            if (upper_shadow_curr > 2 * body_curr and
                lower_shadow_curr < 0.1 * range_curr and
                body_curr > 0.1 * range_curr):
                patterns.append('SHOOTING_STAR')
            
            # Engulfing Bullish
            if (c_prev < o_prev and  # Previous bearish
                c_curr > o_curr and  # Current bullish
                o_curr < c_prev and  # Current open below previous close
                c_curr > o_prev):    # Current close above previous open
                patterns.append('ENGULFING_BULLISH')
        
        return patterns
    
    @staticmethod
    def _get_default_technical_analysis() -> Dict:
        """Análise padrão em caso de erro"""
        return {
            'rsi_analysis': {
                'current_rsi': 50,
                'rsi_zone': 'NEUTRO',
                'rsi_signal': 'NEUTRO',
                'rsi_strength': 'INDEFINIDA'
            },
            'sma_analysis': {
                'sma_trend': 'INDEFINIDA',
                'sma_signal': 'NEUTRO',
                'golden_cross': False,
                'death_cross': False
            },
            'adx_analysis': {
                'current_adx': 20,
                'adx_strength': 'INDEFINIDA',
                'adx_direction': 'LATERAL',
                'adx_signal': 'NEUTRO'
            },
            'trend_confirmation': {
                'consensus': 'NEUTRO',
                'direction': 'LATERAL',
                'agreement_level': 'BAIXO'
            },
            'overall_signal': {
                'action': 'AGUARDAR',
                'confidence': 'BAIXA',
                'probability': 50
            }
        }