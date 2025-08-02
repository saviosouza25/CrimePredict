import sys
import os

# Ensure we can find all installed packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional

class ForexVisualizer:
    """Enhanced visualization utilities for forex analysis"""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, pair: str, indicators: bool = True) -> go.Figure:
        """Create interactive price chart with technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{pair} Price Chart', 'RSI', 'MACD'),
            row_width=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        if indicators and 'SMA_20' in df.columns:
            # Moving averages
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
        
        if indicators and 'BB_Upper' in df.columns:
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        if indicators and 'RSI' in df.columns:
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        if indicators and 'MACD' in df.columns:
            # MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            if 'MACD_Signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=3, col=1
                )
            
            if 'MACD_Histogram' in df.columns:
                colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['MACD_Histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{pair} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_prediction_chart(df: pd.DataFrame, predictions: List[float], 
                              uncertainties: List[float], pair: str, horizon: str) -> go.Figure:
        """Create prediction visualization with confidence intervals"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date, 
            periods=len(predictions) + 1, 
            freq='h'
        )[1:]  # Skip the first date as it's the same as last_date
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', width=3),
                marker=dict(size=6)
            )
        )
        
        # Confidence intervals
        if uncertainties:
            upper_bound = [p + 2*u for p, u in zip(predictions, uncertainties)]
            lower_bound = [p - 2*u for p, u in zip(predictions, uncertainties)]
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name='Upper Bound (95%)',
                    line=dict(color='rgba(255,0,0,0.3)', width=1),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name='Lower Bound (95%)',
                    line=dict(color='rgba(255,0,0,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    showlegend=True
                )
            )
        
        # Add vertical line at prediction start
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
        )
        
        # Add annotation for prediction start
        fig.add_annotation(
            x=last_date,
            y=0.9,
            yref="paper",
            text="Prediction Start",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray"
        )
        
        fig.update_layout(
            title=f'{pair} Price Prediction - {horizon}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_sentiment_gauge(sentiment_score: float, sentiment_signal: str) -> go.Figure:
        """Create sentiment gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Market Sentiment: {sentiment_signal}"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.1], 'color': 'lightcoral'},
                    {'range': [-0.1, 0.1], 'color': 'lightyellow'},
                    {'range': [0.1, 1], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def create_risk_metrics_chart(predictions: List[float], uncertainties: List[float]) -> go.Figure:
        """Create risk metrics visualization"""
        if not uncertainties:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Prediction Confidence', 'Risk Distribution'),
            vertical_spacing=0.15
        )
        
        # Confidence over time
        confidence = [1 - (u / max(uncertainties)) for u in uncertainties]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(confidence))),
                y=confidence,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Risk distribution
        fig.add_trace(
            go.Histogram(
                x=uncertainties,
                name='Risk Distribution',
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Risk Analysis',
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_performance_metrics_table(metrics: Dict) -> pd.DataFrame:
        """Create performance metrics table"""
        metrics_df = pd.DataFrame([
            {'Metric': 'Training Loss', 'Value': f"{metrics.get('final_train_loss', 0):.6f}"},
            {'Metric': 'Validation Loss', 'Value': f"{metrics.get('final_val_loss', 0):.6f}"},
            {'Metric': 'Mean Absolute Error', 'Value': f"{metrics.get('mae', 0):.6f}"},
            {'Metric': 'Root Mean Square Error', 'Value': f"{metrics.get('rmse', 0):.6f}"},
            {'Metric': 'Epochs Trained', 'Value': str(metrics.get('epochs_trained', 0))}
        ])
        return metrics_df
