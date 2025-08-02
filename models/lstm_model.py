import sys
import os

# Ensure we can find all installed packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, List, Dict
import streamlit as st

class ForexDataset(Dataset):
    """Enhanced dataset class for forex data"""
    
    def __init__(self, data: np.ndarray, lookback: int, target_col: int = 0):
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(data).astype(np.float32)
        self.lookback = lookback
        self.target_col = target_col
    
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx:idx+self.lookback], dtype=torch.float32)
        y = torch.tensor(self.data[idx+self.lookback, self.target_col], dtype=torch.float32)
        return X, y

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM model with attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        out = attn_out[:, -1, :]
        
        # Feed through fully connected layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class ForexPredictor:
    """Enhanced forex prediction system"""
    
    def __init__(self, lookback: int = 60, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self.training_history = []
    
    def prepare_features(self, df: pd.DataFrame, sentiment: float) -> np.ndarray:
        """Prepare feature matrix for training"""
        features = []
        
        # Price features
        features.extend([
            df['close'].values,
            df['high'].values,
            df['low'].values,
            df['open'].values
        ])
        
        # Technical indicators
        if 'SMA_10' in df.columns:
            features.append(df['SMA_10'].values)
        if 'SMA_20' in df.columns:
            features.append(df['SMA_20'].values)
        if 'RSI' in df.columns:
            features.append(df['RSI'].values)
        if 'MACD' in df.columns:
            features.append(df['MACD'].values)
        if 'BB_Position' in df.columns:
            features.append(df['BB_Position'].values)
        if 'ATR' in df.columns:
            features.append(df['ATR'].values)
        
        # Add sentiment as a feature
        sentiment_array = np.full(len(df), sentiment, dtype=np.float32)
        features.append(sentiment_array)
        
        # Stack features
        feature_matrix = np.column_stack(features)
        return feature_matrix.astype(np.float32)
    
    def train_model(self, df: pd.DataFrame, sentiment: float, epochs: int = 10, 
                   batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """Train the LSTM model with enhanced features"""
        
        # Prepare features
        feature_matrix = self.prepare_features(df, sentiment)
        
        # Create dataset
        dataset = ForexDataset(feature_matrix, self.lookback)
        self.scaler = dataset.scaler
        
        if len(dataset) < batch_size:
            raise ValueError(f"Not enough data for training. Need at least {self.lookback + batch_size} samples")
        
        # Split data
        train_size = int(0.8 * len(dataset))
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(dataset)))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = feature_matrix.shape[1]
        self.model = EnhancedLSTMModel(input_size, self.hidden_size, self.num_layers, self.dropout)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        self.training_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss = 0.0
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_x).squeeze()
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    predictions = self.model(batch_x).squeeze()
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Track history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
        
        # Calculate metrics
        if val_predictions:
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
        else:
            val_predictions = np.array([])
            val_targets = np.array([])
        
        metrics = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'mae': mean_absolute_error(val_targets, val_predictions) if len(val_predictions) > 0 else 0.0,
            'rmse': np.sqrt(mean_squared_error(val_targets, val_predictions)) if len(val_predictions) > 0 else 0.0,
            'epochs_trained': len(self.training_history)
        }
        
        return metrics
    
    def predict_future(self, df: pd.DataFrame, sentiment: float, steps: int, mc_samples: int = 20) -> Tuple[List[float], List[float]]:
        """Make future predictions with uncertainty estimation"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before making predictions")
        
        feature_matrix = self.prepare_features(df, sentiment)
        scaled_data = self.scaler.transform(feature_matrix)
        
        # Get the last lookback period
        current_input = torch.tensor(
            scaled_data[-self.lookback:], 
            dtype=torch.float32
        ).unsqueeze(0)
        
        predictions = []
        uncertainties = []
        
        self.model.eval()
        
        for step in range(steps):
            step_predictions = []
            
            # Monte Carlo sampling for uncertainty
            for _ in range(mc_samples):
                with torch.no_grad():
                    # Enable dropout during inference for MC sampling
                    self.model.train()  # This enables dropout
                    pred = self.model(current_input)
                    step_predictions.append(pred.item())
            
            # Calculate statistics
            mean_pred = np.mean(step_predictions)
            std_pred = np.std(step_predictions)
            
            predictions.append(mean_pred)
            uncertainties.append(std_pred)
            
            # Update input for next prediction
            # Create new feature vector (simplified approach)
            new_features = np.zeros((1, feature_matrix.shape[1]), dtype=np.float32)
            new_features[0, 0] = mean_pred  # Close price
            new_features[0, -1] = sentiment  # Sentiment
            
            # Scale the new features
            new_features_scaled = self.scaler.transform(new_features)
            
            # Update current_input
            current_input = torch.cat([
                current_input[:, 1:, :], 
                torch.tensor(new_features_scaled, dtype=torch.float32).unsqueeze(0)
            ], dim=1)
        
        return predictions, uncertainties
    
    def get_model_confidence(self, df: pd.DataFrame, sentiment: float) -> float:
        """Calculate model confidence based on recent performance"""
        if self.model is None:
            return 0.0
        
        try:
            feature_matrix = self.prepare_features(df, sentiment)
            if self.scaler is None:
                return 0.5
            scaled_data = self.scaler.transform(feature_matrix)
            
            # Use last 20 samples for confidence calculation
            test_samples = min(20, len(scaled_data) - self.lookback)
            if test_samples <= 0:
                return 0.5
            
            predictions = []
            actuals = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(test_samples):
                    start_idx = len(scaled_data) - test_samples - self.lookback + i
                    end_idx = start_idx + self.lookback
                    
                    input_seq = torch.tensor(
                        scaled_data[start_idx:end_idx], 
                        dtype=torch.float32
                    ).unsqueeze(0)
                    
                    pred = self.model(input_seq).item()
                    actual = scaled_data[end_idx, 0]  # Close price
                    
                    predictions.append(pred)
                    actuals.append(actual)
            
            # Calculate accuracy-based confidence
            actuals_array = np.array(actuals)
            predictions_array = np.array(predictions)
            
            # Avoid division by zero
            non_zero_mask = actuals_array != 0
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((actuals_array[non_zero_mask] - predictions_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
                confidence = max(0.0, min(1.0, 1.0 - mape / 100))
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception:
            return 0.5
