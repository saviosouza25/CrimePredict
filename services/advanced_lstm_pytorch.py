import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class PyTorchLSTM(nn.Module):
    """LSTM neural network using PyTorch for forex prediction"""
    
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism simulation with linear layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Simple attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Dense layers
        output = self.fc1(attended_output)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

class AdvancedLSTMService:
    """Advanced LSTM service with >80% accuracy requirement using PyTorch"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.model_cache = {}
        self.accuracy_threshold = 0.80
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for LSTM training"""
        
        if df.empty or len(df) < 50:
            return df
            
        try:
            features_df = df.copy()
            
            # Basic OHLC features
            features_df['price_change'] = features_df['Close'].pct_change()
            features_df['volatility'] = features_df['price_change'].rolling(window=10).std()
            features_df['volume_ma'] = features_df.get('Volume', 0).rolling(window=10).mean() if 'Volume' in features_df.columns else 0
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(features_df['Close'], 14)
            features_df['sma_20'] = features_df['Close'].rolling(window=20).mean()
            features_df['sma_50'] = features_df['Close'].rolling(window=50).mean()
            
            # Price ratios
            features_df['hl_ratio'] = (features_df['High'] - features_df['Low']) / features_df['Close']
            features_df['oc_ratio'] = (features_df['Open'] - features_df['Close']) / features_df['Close']
            
            # Trend features
            features_df['trend_strength'] = (features_df['Close'] - features_df['sma_20']) / features_df['sma_20']
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            st.error(f"Erro na preparação de features: {e}")
            return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])  # Predict close price
            
        return np.array(X), np.array(y)
    
    def train_model(self, df: pd.DataFrame, pair: str) -> Dict:
        """Train LSTM model with hyperparameter optimization"""
        
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            if len(features_df) < 100:
                return {'success': False, 'error': 'Dados insuficientes para treinamento'}
            
            # Select features for training
            feature_columns = ['Close', 'Volume', 'rsi', 'volatility', 'trend_strength']
            available_features = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_features) < 3:
                return {'success': False, 'error': 'Features insuficientes disponíveis'}
            
            # Prepare training data
            training_data = features_df[available_features].values
            
            # Scale data
            scaled_data = self.scaler.fit_transform(training_data)
            
            # Create sequences
            sequence_length = min(60, len(scaled_data) // 4)
            X, y = self.create_sequences(scaled_data, sequence_length)
            
            if len(X) < 20:
                return {'success': False, 'error': 'Sequências insuficientes para treinamento'}
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
            # Initialize model
            input_size = len(available_features)
            self.model = PyTorchLSTM(input_size=input_size, hidden_size=50, num_layers=2)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            epochs = 50
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(X_test)
                    test_loss = criterion(test_outputs.squeeze(), y_test)
                
                # Early stopping
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Calculate accuracy
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test).squeeze().numpy()
                actual = y_test.numpy()
                
                # Calculate accuracy as percentage of predictions within 5% of actual
                accuracy = np.mean(np.abs(predictions - actual) / np.abs(actual) < 0.05)
                
                # Calculate other metrics
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
            
            # Cache model if accuracy meets threshold
            if accuracy >= self.accuracy_threshold:
                self.model_cache[pair] = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': available_features,
                    'accuracy': accuracy,
                    'timestamp': datetime.now()
                }
            
            return {
                'success': True,
                'accuracy': float(accuracy),
                'mse': float(mse),
                'mae': float(mae),
                'meets_threshold': accuracy >= self.accuracy_threshold,
                'features_used': available_features,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Erro no treinamento: {str(e)}'}
    
    def predict(self, df: pd.DataFrame, pair: str, days_ahead: int = 1) -> Dict:
        """Make predictions using trained model"""
        
        try:
            # Check if we have a cached model
            if pair not in self.model_cache:
                training_result = self.train_model(df, pair)
                if not training_result['success'] or not training_result.get('meets_threshold', False):
                    return {
                        'success': False,
                        'error': f'Modelo não atende critério de {self.accuracy_threshold*100}% acurácia'
                    }
            
            cached_model = self.model_cache[pair]
            model = cached_model['model']
            scaler = cached_model['scaler']
            features = cached_model['features']
            
            # Prepare recent data
            features_df = self.prepare_features(df)
            recent_data = features_df[features].tail(60).values
            
            if len(recent_data) < 60:
                return {'success': False, 'error': 'Dados insuficientes para predição'}
            
            # Scale data
            scaled_data = scaler.transform(recent_data)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
                prediction = model(input_tensor).item()
            
            # Inverse transform prediction
            dummy_array = np.zeros((1, len(features)))
            dummy_array[0, 0] = prediction
            predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
            
            current_price = df['Close'].iloc[-1]
            price_change = predicted_price - current_price
            percentage_change = (price_change / current_price) * 100
            
            # Determine direction and confidence
            if abs(percentage_change) > 0.5:
                direction = 'COMPRA' if percentage_change > 0 else 'VENDA'
                confidence = min(0.95, cached_model['accuracy'] + abs(percentage_change) * 0.01)
            else:
                direction = 'LATERAL'
                confidence = 0.5
            
            return {
                'success': True,
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'price_change': float(price_change),
                'percentage_change': float(percentage_change),
                'direction': direction,
                'confidence': float(confidence),
                'model_accuracy': float(cached_model['accuracy']),
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Erro na predição: {str(e)}'}
    
    def get_model_performance(self, pair: str) -> Dict:
        """Get performance metrics for cached model"""
        
        if pair not in self.model_cache:
            return {'available': False}
        
        cached_model = self.model_cache[pair]
        
        return {
            'available': True,
            'accuracy': cached_model['accuracy'],
            'meets_threshold': cached_model['accuracy'] >= self.accuracy_threshold,
            'last_trained': cached_model['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'features_count': len(cached_model['features']),
            'threshold': self.accuracy_threshold
        }

# Função auxiliar para integração
def analyze_with_advanced_lstm(df: pd.DataFrame, pair: str) -> Dict:
    """Função auxiliar para análise LSTM avançada"""
    
    service = AdvancedLSTMService()
    
    # Train and predict
    prediction_result = service.predict(df, pair)
    performance = service.get_model_performance(pair)
    
    return {
        'prediction': prediction_result,
        'performance': performance,
        'service_type': 'Advanced LSTM (PyTorch)'
    }