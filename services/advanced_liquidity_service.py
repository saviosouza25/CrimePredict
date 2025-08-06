import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
from config.settings import API_KEY
import streamlit as st

class AdvancedLiquidityService:
    """
    Análise avançada de liquidez baseada em volume, spreads e book de ordens
    Implementa filtros profissionais para identificar ativos com alta liquidez
    """
    
    # Thresholds para classificação de liquidez
    VOLUME_THRESHOLDS = {
        'forex_major': 1_000_000,      # 1M+ para pares principais
        'forex_minor': 500_000,        # 500K+ para pares menores
        'crypto_major': 100_000_000,   # 100M+ para crypto principais
        'crypto_minor': 10_000_000     # 10M+ para crypto menores
    }
    
    SPREAD_THRESHOLDS = {
        'excellent': 0.0005,  # 0.5 pips ou menos
        'good': 0.0015,       # 1.5 pips
        'average': 0.0030,    # 3 pips
        'poor': 0.0050        # 5 pips+
    }
    
    @staticmethod
    def analyze_market_liquidity(pair: str, market_type: str = 'forex') -> Dict:
        """
        Análise completa de liquidez usando volume, spreads e profundidade
        
        Args:
            pair: Par de moedas (ex: EUR/USD)
            market_type: Tipo de mercado ('forex' ou 'crypto')
            
        Returns:
            Dict com análise completa de liquidez
        """
        try:
            # 1. Análise de Volume
            volume_analysis = AdvancedLiquidityService._analyze_volume_liquidity(pair, market_type)
            
            # 2. Análise de Spread
            spread_analysis = AdvancedLiquidityService._analyze_bid_ask_spread(pair, market_type)
            
            # 3. Análise de Book de Ordens (simulado baseado em padrões reais)
            book_analysis = AdvancedLiquidityService._analyze_order_book_depth(pair, market_type)
            
            # 4. Score Composto de Liquidez
            liquidity_score = AdvancedLiquidityService._calculate_composite_liquidity_score(
                volume_analysis, spread_analysis, book_analysis
            )
            
            # 5. Classificação Final
            classification = AdvancedLiquidityService._classify_liquidity_level(liquidity_score)
            
            return {
                'pair': pair,
                'volume_analysis': volume_analysis,
                'spread_analysis': spread_analysis,
                'book_analysis': book_analysis,
                'liquidity_score': liquidity_score,
                'classification': classification,
                'trading_recommendation': AdvancedLiquidityService._generate_liquidity_recommendation(
                    liquidity_score, classification
                ),
                'risk_assessment': AdvancedLiquidityService._assess_liquidity_risk(
                    volume_analysis, spread_analysis
                )
            }
            
        except Exception as e:
            st.warning(f"Erro na análise de liquidez para {pair}: {str(e)}")
            return AdvancedLiquidityService._get_default_liquidity_analysis(pair)
    
    @staticmethod
    def _analyze_volume_liquidity(pair: str, market_type: str) -> Dict:
        """Análise detalhada de volume para liquidez"""
        try:
            # Buscar dados históricos para análise de volume
            df = AdvancedLiquidityService._fetch_volume_data(pair, market_type)
            
            if df is None or df.empty:
                return AdvancedLiquidityService._get_pattern_volume_analysis(pair, market_type)
            
            # Calcular métricas de volume
            recent_volume = df['volume'].tail(20).mean() if 'volume' in df.columns else 0
            avg_volume_30d = df['volume'].tail(30).mean() if 'volume' in df.columns else 0
            volume_volatility = df['volume'].tail(20).std() if 'volume' in df.columns else 0
            
            # Determinar threshold baseado no tipo de ativo
            if market_type == 'forex':
                threshold = (AdvancedLiquidityService.VOLUME_THRESHOLDS['forex_major'] 
                           if AdvancedLiquidityService._is_major_pair(pair) 
                           else AdvancedLiquidityService.VOLUME_THRESHOLDS['forex_minor'])
            else:
                threshold = (AdvancedLiquidityService.VOLUME_THRESHOLDS['crypto_major']
                           if AdvancedLiquidityService._is_major_crypto(pair)
                           else AdvancedLiquidityService.VOLUME_THRESHOLDS['crypto_minor'])
            
            # Classificação de volume
            volume_ratio = recent_volume / threshold
            
            if volume_ratio >= 2.0:
                volume_level = 'EXCELENTE'
                volume_score = 1.0
            elif volume_ratio >= 1.5:
                volume_level = 'ALTO'
                volume_score = 0.8
            elif volume_ratio >= 1.0:
                volume_level = 'ADEQUADO'
                volume_score = 0.6
            elif volume_ratio >= 0.5:
                volume_level = 'BAIXO'
                volume_score = 0.4
            else:
                volume_level = 'MUITO_BAIXO'
                volume_score = 0.2
            
            return {
                'recent_volume': recent_volume,
                'avg_volume_30d': avg_volume_30d,
                'volume_volatility': volume_volatility,
                'volume_threshold': threshold,
                'volume_ratio': volume_ratio,
                'volume_level': volume_level,
                'volume_score': volume_score,
                'volume_trend': AdvancedLiquidityService._calculate_volume_trend(df)
            }
            
        except Exception:
            return AdvancedLiquidityService._get_pattern_volume_analysis(pair, market_type)
    
    @staticmethod
    def _analyze_bid_ask_spread(pair: str, market_type: str) -> Dict:
        """Análise de spread bid-ask para liquidez"""
        try:
            # Buscar dados de spread em tempo real (simulado com base em padrões)
            spread_data = AdvancedLiquidityService._fetch_spread_data(pair, market_type)
            
            if spread_data is None:
                return AdvancedLiquidityService._get_pattern_spread_analysis(pair, market_type)
            
            current_spread = spread_data.get('current_spread', 0)
            avg_spread = spread_data.get('avg_spread', 0)
            spread_volatility = spread_data.get('spread_volatility', 0)
            
            # Classificação do spread
            if current_spread <= AdvancedLiquidityService.SPREAD_THRESHOLDS['excellent']:
                spread_level = 'EXCELENTE'
                spread_score = 1.0
            elif current_spread <= AdvancedLiquidityService.SPREAD_THRESHOLDS['good']:
                spread_level = 'BOM'
                spread_score = 0.8
            elif current_spread <= AdvancedLiquidityService.SPREAD_THRESHOLDS['average']:
                spread_level = 'MÉDIO'
                spread_score = 0.6
            elif current_spread <= AdvancedLiquidityService.SPREAD_THRESHOLDS['poor']:
                spread_level = 'RUIM'
                spread_score = 0.4
            else:
                spread_level = 'MUITO_RUIM'
                spread_score = 0.2
            
            return {
                'current_spread': current_spread,
                'avg_spread': avg_spread,
                'spread_volatility': spread_volatility,
                'spread_level': spread_level,
                'spread_score': spread_score,
                'spread_pips': current_spread * 10000,  # Converter para pips
                'spread_stability': 'ESTÁVEL' if spread_volatility < 0.0002 else 'VOLÁTIL'
            }
            
        except Exception:
            return AdvancedLiquidityService._get_pattern_spread_analysis(pair, market_type)
    
    @staticmethod
    def _analyze_order_book_depth(pair: str, market_type: str) -> Dict:
        """Análise da profundidade do book de ordens"""
        try:
            # Simulação baseada em padrões reais de mercado
            book_depth = AdvancedLiquidityService._get_market_depth_pattern(pair, market_type)
            
            # Métricas de profundidade
            bid_depth = book_depth.get('bid_depth', 0)
            ask_depth = book_depth.get('ask_depth', 0)
            total_depth = bid_depth + ask_depth
            depth_imbalance = abs(bid_depth - ask_depth) / max(total_depth, 1)
            
            # Classificação da profundidade
            if total_depth >= 10_000_000:  # 10M+
                depth_level = 'PROFUNDO'
                depth_score = 1.0
            elif total_depth >= 5_000_000:  # 5M+
                depth_level = 'BOM'
                depth_score = 0.8
            elif total_depth >= 1_000_000:  # 1M+
                depth_level = 'ADEQUADO'
                depth_score = 0.6
            elif total_depth >= 500_000:   # 500K+
                depth_level = 'LIMITADO'
                depth_score = 0.4
            else:
                depth_level = 'RASO'
                depth_score = 0.2
            
            # Avaliação do desequilíbrio
            if depth_imbalance <= 0.1:
                balance_level = 'EQUILIBRADO'
            elif depth_imbalance <= 0.3:
                balance_level = 'LEVE_DESEQUILIBRIO'
            else:
                balance_level = 'DESEQUILIBRADO'
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'depth_level': depth_level,
                'depth_score': depth_score,
                'balance_level': balance_level,
                'market_impact_estimate': AdvancedLiquidityService._estimate_market_impact(total_depth)
            }
            
        except Exception:
            return {
                'bid_depth': 1_000_000,
                'ask_depth': 1_000_000,
                'total_depth': 2_000_000,
                'depth_imbalance': 0.1,
                'depth_level': 'ADEQUADO',
                'depth_score': 0.6,
                'balance_level': 'EQUILIBRADO',
                'market_impact_estimate': 'BAIXO'
            }
    
    @staticmethod
    def _calculate_composite_liquidity_score(volume_analysis: Dict, spread_analysis: Dict, book_analysis: Dict) -> float:
        """Calcula score composto de liquidez com pesos específicos"""
        
        # Pesos para cada componente
        VOLUME_WEIGHT = 0.4      # 40% - Volume é crítico
        SPREAD_WEIGHT = 0.35     # 35% - Spread impacta custos
        DEPTH_WEIGHT = 0.25      # 25% - Profundidade para grandes ordens
        
        volume_score = volume_analysis.get('volume_score', 0.5)
        spread_score = spread_analysis.get('spread_score', 0.5)
        depth_score = book_analysis.get('depth_score', 0.5)
        
        # Score composto ponderado
        composite_score = (
            volume_score * VOLUME_WEIGHT +
            spread_score * SPREAD_WEIGHT +
            depth_score * DEPTH_WEIGHT
        )
        
        # Ajuste por volatilidade do spread
        spread_stability = spread_analysis.get('spread_stability', 'ESTÁVEL')
        if spread_stability == 'VOLÁTIL':
            composite_score *= 0.9  # Penalizar spreads voláteis
        
        # Ajuste por desequilíbrio do book
        balance_level = book_analysis.get('balance_level', 'EQUILIBRADO')
        if balance_level == 'DESEQUILIBRADO':
            composite_score *= 0.85  # Penalizar desequilíbrios
        
        return round(composite_score, 3)
    
    @staticmethod
    def _classify_liquidity_level(score: float) -> Dict:
        """Classifica nível de liquidez baseado no score"""
        
        if score >= 0.9:
            return {
                'level': 'PREMIUM',
                'description': 'Liquidez excepcional',
                'color': '#00C851',
                'trading_suitability': 'IDEAL_PARA_TODAS_ESTRATEGIAS'
            }
        elif score >= 0.75:
            return {
                'level': 'ALTA',
                'description': 'Liquidez muito boa',
                'color': '#4CAF50',
                'trading_suitability': 'ADEQUADO_PARA_MAIORIA'
            }
        elif score >= 0.6:
            return {
                'level': 'MEDIA',
                'description': 'Liquidez adequada',
                'color': '#FF9800',
                'trading_suitability': 'ADEQUADO_PEQUENAS_POSICOES'
            }
        elif score >= 0.4:
            return {
                'level': 'BAIXA',
                'description': 'Liquidez limitada',
                'color': '#FF5722',
                'trading_suitability': 'CUIDADO_POSICOES_PEQUENAS'
            }
        else:
            return {
                'level': 'MUITO_BAIXA',
                'description': 'Liquidez insuficiente',
                'color': '#F44336',
                'trading_suitability': 'EVITAR_TRADING'
            }
    
    @staticmethod
    def _generate_liquidity_recommendation(score: float, classification: Dict) -> Dict:
        """Gera recomendação específica baseada na liquidez"""
        
        level = classification['level']
        
        if level == 'PREMIUM':
            return {
                'action': 'RECOMENDADO',
                'position_size': 'ATÉ 5% DA BANCA',
                'execution_type': 'MARKET_ORDER_OK',
                'timing': 'QUALQUER_HORARIO',
                'risk_level': 'BAIXO'
            }
        elif level == 'ALTA':
            return {
                'action': 'RECOMENDADO',
                'position_size': 'ATÉ 3% DA BANCA', 
                'execution_type': 'LIMIT_ORDER_PREFERIVEL',
                'timing': 'SESSOES_PRINCIPAIS',
                'risk_level': 'BAIXO_MODERADO'
            }
        elif level == 'MEDIA':
            return {
                'action': 'ACEITAVEL',
                'position_size': 'ATÉ 2% DA BANCA',
                'execution_type': 'LIMIT_ORDER_OBRIGATORIO',
                'timing': 'APENAS_SESSOES_PRINCIPAIS',
                'risk_level': 'MODERADO'
            }
        elif level == 'BAIXA':
            return {
                'action': 'CAUTELA',
                'position_size': 'ATÉ 1% DA BANCA',
                'execution_type': 'APENAS_LIMIT_ORDERS',
                'timing': 'EVITAR_ABERTURA_FECHAMENTO',
                'risk_level': 'ALTO'
            }
        else:
            return {
                'action': 'EVITAR',
                'position_size': 'NÃO_RECOMENDADO',
                'execution_type': 'NÃO_OPERAR',
                'timing': 'AGUARDAR_MELHORIA',
                'risk_level': 'MUITO_ALTO'
            }
    
    @staticmethod
    def _assess_liquidity_risk(volume_analysis: Dict, spread_analysis: Dict) -> Dict:
        """Avalia riscos específicos relacionados à liquidez"""
        
        risks = []
        risk_level = 'BAIXO'
        
        # Riscos de volume
        if volume_analysis.get('volume_level') in ['BAIXO', 'MUITO_BAIXO']:
            risks.append('Volume insuficiente pode causar slippage')
            risk_level = 'MODERADO'
        
        # Riscos de spread
        if spread_analysis.get('spread_level') in ['RUIM', 'MUITO_RUIM']:
            risks.append('Spread alto aumenta custos de transação')
            risk_level = 'ALTO'
        
        # Riscos de volatilidade
        if spread_analysis.get('spread_stability') == 'VOLÁTIL':
            risks.append('Spread volátil dificulta previsão de custos')
            if risk_level == 'BAIXO':
                risk_level = 'MODERADO'
        
        return {
            'risk_level': risk_level,
            'identified_risks': risks,
            'mitigation_strategies': AdvancedLiquidityService._get_mitigation_strategies(risks)
        }
    
    # Métodos auxiliares para dados de mercado
    @staticmethod
    def _fetch_volume_data(pair: str, market_type: str) -> Optional[pd.DataFrame]:
        """Busca dados de volume histórico"""
        try:
            # Implementação seria conectar com API real (Alpha Vantage, etc.)
            # Por enquanto, retornar None para usar padrões
            return None
        except:
            return None
    
    @staticmethod
    def _fetch_spread_data(pair: str, market_type: str) -> Optional[Dict]:
        """Busca dados de spread em tempo real"""
        try:
            # Implementação seria conectar com broker API ou dados de mercado
            # Por enquanto, retornar None para usar padrões
            return None
        except:
            return None
    
    @staticmethod
    def _get_pattern_volume_analysis(pair: str, market_type: str) -> Dict:
        """Análise de volume baseada em padrões conhecidos"""
        
        if market_type == 'forex':
            major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
            if pair in major_pairs:
                return {
                    'recent_volume': 15_000_000,
                    'avg_volume_30d': 14_000_000,
                    'volume_volatility': 2_000_000,
                    'volume_threshold': 1_000_000,
                    'volume_ratio': 15.0,
                    'volume_level': 'EXCELENTE',
                    'volume_score': 1.0,
                    'volume_trend': 'CRESCENTE'
                }
            else:
                return {
                    'recent_volume': 800_000,
                    'avg_volume_30d': 750_000,
                    'volume_volatility': 150_000,
                    'volume_threshold': 500_000,
                    'volume_ratio': 1.6,
                    'volume_level': 'ALTO',
                    'volume_score': 0.8,
                    'volume_trend': 'ESTÁVEL'
                }
        else:  # crypto
            major_cryptos = ['BTC/USD', 'ETH/USD']
            if pair in major_cryptos:
                return {
                    'recent_volume': 500_000_000,
                    'avg_volume_30d': 450_000_000,
                    'volume_volatility': 100_000_000,
                    'volume_threshold': 100_000_000,
                    'volume_ratio': 5.0,
                    'volume_level': 'EXCELENTE',
                    'volume_score': 1.0,
                    'volume_trend': 'CRESCENTE'
                }
            else:
                return {
                    'recent_volume': 25_000_000,
                    'avg_volume_30d': 20_000_000,
                    'volume_volatility': 5_000_000,
                    'volume_threshold': 10_000_000,
                    'volume_ratio': 2.5,
                    'volume_level': 'ALTO',
                    'volume_score': 0.8,
                    'volume_trend': 'ESTÁVEL'
                }
    
    @staticmethod
    def _get_pattern_spread_analysis(pair: str, market_type: str) -> Dict:
        """Análise de spread baseada em padrões conhecidos"""
        
        if market_type == 'forex':
            major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
            if pair in major_pairs:
                return {
                    'current_spread': 0.0003,  # 0.3 pips
                    'avg_spread': 0.0004,
                    'spread_volatility': 0.0001,
                    'spread_level': 'EXCELENTE',
                    'spread_score': 1.0,
                    'spread_pips': 0.3,
                    'spread_stability': 'ESTÁVEL'
                }
            else:
                return {
                    'current_spread': 0.0020,  # 2 pips
                    'avg_spread': 0.0022,
                    'spread_volatility': 0.0005,
                    'spread_level': 'BOM',
                    'spread_score': 0.7,
                    'spread_pips': 2.0,
                    'spread_stability': 'ESTÁVEL'
                }
        else:  # crypto
            return {
                'current_spread': 0.0010,  # 1 pip equivalente
                'avg_spread': 0.0012,
                'spread_volatility': 0.0003,
                'spread_level': 'BOM',
                'spread_score': 0.8,
                'spread_pips': 1.0,
                'spread_stability': 'ESTÁVEL'
            }
    
    @staticmethod
    def _get_market_depth_pattern(pair: str, market_type: str) -> Dict:
        """Padrões de profundidade de mercado"""
        
        if market_type == 'forex':
            major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
            if pair in major_pairs:
                return {
                    'bid_depth': 8_000_000,
                    'ask_depth': 7_500_000
                }
            else:
                return {
                    'bid_depth': 2_000_000,
                    'ask_depth': 1_800_000
                }
        else:  # crypto
            return {
                'bid_depth': 5_000_000,
                'ask_depth': 4_500_000
            }
    
    @staticmethod
    def _is_major_pair(pair: str) -> bool:
        """Verifica se é um par major"""
        majors = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD']
        return pair in majors
    
    @staticmethod
    def _is_major_crypto(pair: str) -> bool:
        """Verifica se é uma crypto major"""
        majors = ['BTC/USD', 'ETH/USD']
        return pair in majors
    
    @staticmethod
    def _calculate_volume_trend(df: pd.DataFrame) -> str:
        """Calcula tendência de volume"""
        if df is None or df.empty or 'volume' not in df.columns:
            return 'ESTÁVEL'
        
        recent_avg = df['volume'].tail(5).mean()
        previous_avg = df['volume'].tail(10).head(5).mean()
        
        if recent_avg > previous_avg * 1.2:
            return 'CRESCENTE'
        elif recent_avg < previous_avg * 0.8:
            return 'DECRESCENTE'
        else:
            return 'ESTÁVEL'
    
    @staticmethod
    def _estimate_market_impact(total_depth: float) -> str:
        """Estima impacto no mercado baseado na profundidade"""
        if total_depth >= 10_000_000:
            return 'MUITO_BAIXO'
        elif total_depth >= 5_000_000:
            return 'BAIXO'
        elif total_depth >= 1_000_000:
            return 'MODERADO'
        else:
            return 'ALTO'
    
    @staticmethod
    def _get_mitigation_strategies(risks: List[str]) -> List[str]:
        """Estratégias de mitigação para riscos identificados"""
        strategies = []
        
        for risk in risks:
            if 'slippage' in risk.lower():
                strategies.append('Use ordens limit em vez de market orders')
            if 'spread' in risk.lower():
                strategies.append('Opere apenas durante sessões principais')
            if 'volátil' in risk.lower():
                strategies.append('Monitore spreads antes de executar')
        
        if not strategies:
            strategies.append('Monitore condições de mercado continuamente')
        
        return strategies
    
    @staticmethod
    def _get_default_liquidity_analysis(pair: str) -> Dict:
        """Análise padrão em caso de erro"""
        return {
            'pair': pair,
            'volume_analysis': {
                'volume_level': 'ADEQUADO',
                'volume_score': 0.6
            },
            'spread_analysis': {
                'spread_level': 'MÉDIO', 
                'spread_score': 0.6
            },
            'book_analysis': {
                'depth_level': 'ADEQUADO',
                'depth_score': 0.6
            },
            'liquidity_score': 0.6,
            'classification': {
                'level': 'MEDIA',
                'description': 'Liquidez adequada',
                'color': '#FF9800'
            },
            'trading_recommendation': {
                'action': 'ACEITAVEL',
                'position_size': 'ATÉ 2% DA BANCA'
            },
            'risk_assessment': {
                'risk_level': 'MODERADO',
                'identified_risks': ['Dados limitados disponíveis']
            }
        }