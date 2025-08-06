"""
Advanced Trend Analysis Engine
Combines Alpha Vantage indicators with operational profile optimization
"""
import streamlit as st
from typing import Dict, List, Optional
from services.alpha_vantage_indicators import AlphaVantageIndicators

class TrendAnalysisEngine:
    """
    Advanced trend analysis engine optimized for each trading profile
    Uses authentic Alpha Vantage data with profile-specific indicator combinations
    """
    
    @staticmethod
    def analyze_trend_by_profile(pair: str, profile: str, interval: str) -> Dict:
        """
        Main trend analysis function optimized for trading profile
        Returns comprehensive trend analysis with precision scoring
        """
        
        # Validate profile
        valid_profiles = ['scalping', 'intraday', 'swing', 'position']
        if profile not in valid_profiles:
            profile = 'swing'  # Default to swing trading
        
        # Get Alpha Vantage indicators for this profile
        try:
            analysis = AlphaVantageIndicators.get_trend_analysis(pair, profile, interval)
            
            # Enhance with profile-specific logic
            enhanced_analysis = TrendAnalysisEngine._enhance_profile_analysis(
                analysis, pair, profile, interval
            )
            
            return enhanced_analysis
            
        except Exception as e:
            st.error(f"❌ Erro Alpha Vantage: {e}")
            return TrendAnalysisEngine._create_error_response(pair, profile, str(e))
    
    @staticmethod
    def _enhance_profile_analysis(analysis: Dict, pair: str, profile: str, interval: str) -> Dict:
        """Enhance analysis with profile-specific logic and recommendations"""
        
        enhanced = analysis.copy()
        enhanced.update({
            'profile_recommendations': TrendAnalysisEngine._get_profile_recommendations(analysis, profile),
            'risk_management': TrendAnalysisEngine._calculate_risk_management(analysis, profile),
            'timing_signals': TrendAnalysisEngine._generate_timing_signals(analysis, profile),
            'market_context': TrendAnalysisEngine._analyze_market_context(analysis, pair, interval),
            'execution_plan': TrendAnalysisEngine._create_execution_plan(analysis, profile)
        })
        
        return enhanced
    
    @staticmethod
    def _get_profile_recommendations(analysis: Dict, profile: str) -> Dict:
        """Generate specific recommendations based on trading profile"""
        
        recommendations = {
            'scalping': {
                'optimal_sessions': ['London Open', 'NY Open', 'London-NY Overlap'],
                'avoid_sessions': ['Asian Low Liquidity'],
                'max_trade_duration': '5-15 minutes',
                'ideal_spread': '< 2 pips',
                'volume_requirement': 'High',
                'news_impact': 'Avoid major news releases',
                'indicators_focus': 'EMA crossovers + Stochastic extremes'
            },
            'intraday': {
                'optimal_sessions': ['London Session', 'NY Session'],
                'avoid_sessions': ['Asian Consolidation'],
                'max_trade_duration': '2-8 hours',
                'ideal_spread': '< 3 pips',
                'volume_requirement': 'Medium-High',
                'news_impact': 'Can trade through minor news',
                'indicators_focus': 'MACD + ADX + Bollinger Bands'
            },
            'swing': {
                'optimal_sessions': ['Any - Multi-day trades'],
                'avoid_sessions': 'Friday late (weekend risk)',
                'max_trade_duration': '2-10 days',
                'ideal_spread': '< 5 pips',
                'volume_requirement': 'Medium',
                'news_impact': 'Incorporate fundamental analysis',
                'indicators_focus': 'SMA systems + ADX + Parabolic SAR'
            },
            'position': {
                'optimal_sessions': ['Any - Long-term perspective'],
                'avoid_sessions': 'Month-end (rebalancing)',
                'max_trade_duration': '2 weeks - 6 months',
                'ideal_spread': '< 10 pips',
                'volume_requirement': 'Low-Medium',
                'news_impact': 'Central bank policies critical',
                'indicators_focus': 'Long-term SMAs + Monthly MACD'
            }
        }
        
        return recommendations.get(profile, recommendations['swing'])
    
    @staticmethod
    def _calculate_risk_management(analysis: Dict, profile: str) -> Dict:
        """Calculate risk management parameters based on profile and trend strength"""
        
        trend_signals = analysis.get('trend_signals', {})
        confidence = trend_signals.get('confidence', 0.5)
        
        risk_params = {
            'scalping': {
                'position_size': min(5.0, confidence * 10),  # Max 5% risk per trade
                'stop_loss_pips': 8,
                'take_profit_pips': 12,
                'risk_reward_ratio': 1.5,
                'max_concurrent_trades': 3
            },
            'intraday': {
                'position_size': min(3.0, confidence * 6),  # Max 3% risk per trade
                'stop_loss_pips': 25,
                'take_profit_pips': 50,
                'risk_reward_ratio': 2.0,
                'max_concurrent_trades': 2
            },
            'swing': {
                'position_size': min(2.0, confidence * 4),  # Max 2% risk per trade
                'stop_loss_pips': 80,
                'take_profit_pips': 200,
                'risk_reward_ratio': 2.5,
                'max_concurrent_trades': 5
            },
            'position': {
                'position_size': min(1.0, confidence * 2),  # Max 1% risk per trade
                'stop_loss_pips': 200,
                'take_profit_pips': 600,
                'risk_reward_ratio': 3.0,
                'max_concurrent_trades': 3
            }
        }
        
        base_params = risk_params.get(profile, risk_params['swing'])
        
        # Adjust based on confidence level
        if confidence > 0.8:
            base_params['position_size'] *= 1.2
        elif confidence < 0.6:
            base_params['position_size'] *= 0.7
            
        return base_params
    
    @staticmethod
    def _generate_timing_signals(analysis: Dict, profile: str) -> Dict:
        """Generate timing signals based on profile requirements"""
        
        indicators = analysis.get('indicators', {})
        trend_signals = analysis.get('trend_signals', {})
        
        timing = {
            'entry_condition': 'WAIT',
            'exit_condition': 'HOLD',
            'signal_strength': 'WEAK',
            'timing_score': 0.0,
            'next_review': '1 hour'
        }
        
        # Profile-specific timing logic
        if profile == 'scalping':
            # Fast signals for scalping
            if 'stoch' in indicators and 'ema' in indicators:
                stoch = indicators['stoch']
                if stoch.get('trend') == 'BULLISH' and stoch.get('strength', 0) > 0.7:
                    timing['entry_condition'] = 'ENTER_LONG'
                    timing['signal_strength'] = 'STRONG'
                    timing['timing_score'] = 0.85
                elif stoch.get('trend') == 'BEARISH' and stoch.get('strength', 0) > 0.7:
                    timing['entry_condition'] = 'ENTER_SHORT'
                    timing['signal_strength'] = 'STRONG' 
                    timing['timing_score'] = 0.85
            timing['next_review'] = '5 minutes'
            
        elif profile == 'intraday':
            # Medium-term signals
            if 'macd' in indicators and 'adx' in indicators:
                macd = indicators['macd']
                adx = indicators['adx']
                if (macd.get('trend') == 'BULLISH' and adx.get('trend_strong', False)):
                    timing['entry_condition'] = 'ENTER_LONG'
                    timing['signal_strength'] = 'STRONG'
                    timing['timing_score'] = 0.80
            timing['next_review'] = '30 minutes'
            
        elif profile == 'swing':
            # Multi-day signals
            unified_trend = trend_signals.get('unified_trend', 'NEUTRAL')
            confidence = trend_signals.get('confidence', 0.0)
            
            if unified_trend == 'BULLISH' and confidence > 0.75:
                timing['entry_condition'] = 'ENTER_LONG'
                timing['signal_strength'] = 'STRONG'
                timing['timing_score'] = confidence
            elif unified_trend == 'BEARISH' and confidence > 0.75:
                timing['entry_condition'] = 'ENTER_SHORT'
                timing['signal_strength'] = 'STRONG'
                timing['timing_score'] = confidence
            timing['next_review'] = '4 hours'
            
        elif profile == 'position':
            # Long-term signals
            if 'sma' in indicators and 'aroon' in indicators:
                sma = indicators['sma']
                if sma.get('trend') == 'BULLISH' and sma.get('strength', 0) > 0.8:
                    timing['entry_condition'] = 'ENTER_LONG'
                    timing['signal_strength'] = 'STRONG'
                    timing['timing_score'] = 0.75
            timing['next_review'] = '1 day'
        
        return timing
    
    @staticmethod
    def _analyze_market_context(analysis: Dict, pair: str, interval: str) -> Dict:
        """Analyze broader market context for the trading pair"""
        
        context = {
            'market_session': TrendAnalysisEngine._get_current_session(),
            'volatility_environment': 'NORMAL',
            'correlation_risk': 'MEDIUM',
            'news_risk': 'LOW',
            'liquidity_level': 'HIGH'
        }
        
        # Pair-specific context
        if 'JPY' in pair:
            context['correlation_risk'] = 'HIGH'  # Yen pairs often correlate
            context['news_risk'] = 'MEDIUM'  # BoJ interventions
        elif 'GBP' in pair:
            context['volatility_environment'] = 'HIGH'  # GBP naturally volatile
            context['news_risk'] = 'MEDIUM'  # Brexit/BoE sensitive
        elif 'EUR' in pair:
            context['news_risk'] = 'MEDIUM'  # ECB sensitive
            
        return context
    
    @staticmethod
    def _create_execution_plan(analysis: Dict, profile: str) -> Dict:
        """Create detailed execution plan based on analysis"""
        
        trend_signals = analysis.get('trend_signals', {})
        confidence = trend_signals.get('confidence', 0.0)
        
        plan = {
            'recommended_action': 'WAIT',
            'entry_strategy': 'Market Order',
            'stop_strategy': 'Fixed Stop',
            'profit_strategy': 'Fixed Target',
            'monitoring_frequency': '1 hour',
            'review_triggers': []
        }
        
        if confidence > 0.8:
            plan['recommended_action'] = 'EXECUTE'
            plan['entry_strategy'] = 'Market Order'
        elif confidence > 0.6:
            plan['recommended_action'] = 'PREPARE'
            plan['entry_strategy'] = 'Limit Order'
        else:
            plan['recommended_action'] = 'WAIT'
            plan['entry_strategy'] = 'No Entry'
            
        # Profile-specific execution details
        if profile == 'scalping':
            plan['monitoring_frequency'] = '1 minute'
            plan['stop_strategy'] = 'Tight Stop (8 pips)'
            plan['review_triggers'] = ['Price action', 'Volume spike']
        elif profile == 'swing':
            plan['monitoring_frequency'] = '4 hours'
            plan['stop_strategy'] = 'Swing Stop (80 pips)'
            plan['review_triggers'] = ['Daily close', 'Major news']
            
        return plan
    
    @staticmethod
    def _get_current_session() -> str:
        """Determine current forex trading session"""
        from datetime import datetime, timezone
        
        utc_now = datetime.now(timezone.utc)
        hour = utc_now.hour
        
        if 22 <= hour or hour < 7:
            return 'ASIAN'
        elif 7 <= hour < 16:
            return 'LONDON'  
        elif 16 <= hour < 22:
            return 'NEW_YORK'
        else:
            return 'OVERLAP'
    
    @staticmethod
    def _create_error_response(pair: str, profile: str, error: str) -> Dict:
        """Create error response when analysis fails"""
        return {
            'pair': pair,
            'profile': profile,
            'error': True,
            'error_message': error,
            'trend_signals': {
                'unified_trend': 'UNKNOWN',
                'confidence': 0.0,
                'strength': 0.0
            },
            'profile_recommendations': {
                'recommendation': 'SISTEMA INDISPONÍVEL - Aguarde correção da API Alpha Vantage'
            },
            'execution_plan': {
                'recommended_action': 'WAIT',
                'reason': 'Dados não disponíveis'
            }
        }