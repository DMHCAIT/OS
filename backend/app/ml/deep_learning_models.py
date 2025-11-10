"""
Deep Learning Lead Scoring Models
Advanced transformer-based models for sophisticated lead scoring and prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel, 
    DistilBertModel, RobertaModel,
    TrainingArguments, Trainer
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class LeadScoringPrediction:
    """Lead scoring prediction result"""
    lead_id: str
    overall_score: float
    conversion_probability: float
    revenue_potential: float
    urgency_score: float
    engagement_score: float
    behavioral_score: float
    demographic_score: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_timestamp: datetime
    model_version: str


class MultiModalLeadDataset(Dataset):
    """Dataset for multi-modal lead data processing"""
    
    def __init__(
        self,
        text_data: List[str],
        numerical_features: np.ndarray,
        categorical_features: np.ndarray,
        behavioral_sequences: np.ndarray,
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int = 512
    ):
        self.text_data = text_data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.behavioral_sequences = behavioral_sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        # Tokenize text data
        text = str(self.text_data[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.FloatTensor(self.numerical_features[idx]),
            'categorical_features': torch.LongTensor(self.categorical_features[idx]),
            'behavioral_sequence': torch.FloatTensor(self.behavioral_sequences[idx]),
            'labels': torch.FloatTensor([self.labels[idx]])
        }


class AttentionPooling(nn.Module):
    """Attention pooling layer for sequence aggregation"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, sequence_output, attention_mask):
        # Calculate attention scores
        attention_scores = self.attention_weights(sequence_output)
        attention_scores = attention_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float('-inf')
        )
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        weighted_output = torch.sum(attention_weights * sequence_output, dim=1)
        return weighted_output


class BehavioralEncoder(nn.Module):
    """Encoder for behavioral sequence data"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = AttentionPooling(hidden_size * 2)
        
    def forward(self, behavioral_sequence, sequence_lengths=None):
        # LSTM encoding
        lstm_output, _ = self.lstm(behavioral_sequence)
        
        # Create attention mask
        batch_size, seq_len, _ = lstm_output.shape
        if sequence_lengths is not None:
            attention_mask = torch.zeros(batch_size, seq_len)
            for i, length in enumerate(sequence_lengths):
                attention_mask[i, :length] = 1
        else:
            attention_mask = torch.ones(batch_size, seq_len)
        
        # Apply attention pooling
        pooled_output = self.attention(lstm_output, attention_mask)
        return pooled_output


class TransformerLeadScoringModel(nn.Module):
    """Advanced transformer-based lead scoring model"""
    
    def __init__(
        self,
        text_model_name: str = 'distilbert-base-uncased',
        numerical_feature_size: int = 50,
        categorical_vocab_sizes: List[int] = None,
        categorical_embedding_dims: List[int] = None,
        behavioral_input_size: int = 20,
        behavioral_sequence_length: int = 100,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.2,
        output_size: int = 6  # overall, conversion, revenue, urgency, engagement, behavioral
    ):
        super().__init__()
        
        # Text encoder (transformer)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_model.config.hidden_size
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Categorical features embeddings
        if categorical_vocab_sizes and categorical_embedding_dims:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, embedding_dim)
                for vocab_size, embedding_dim in zip(categorical_vocab_sizes, categorical_embedding_dims)
            ])
            categorical_total_dim = sum(categorical_embedding_dims)
        else:
            self.categorical_embeddings = None
            categorical_total_dim = 0
        
        # Behavioral sequence encoder
        self.behavioral_encoder = BehavioralEncoder(
            input_size=behavioral_input_size,
            hidden_size=hidden_size // 2,
            dropout=dropout
        )
        
        # Feature fusion layer
        fusion_input_size = (
            text_hidden_size + 
            hidden_size +  # numerical
            categorical_total_dim + 
            hidden_size  # behavioral (bidirectional LSTM output)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention for feature interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer layers for complex feature interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Output heads for different predictions
        self.output_heads = nn.ModuleDict({
            'overall_score': nn.Linear(hidden_size * 2, 1),
            'conversion_probability': nn.Linear(hidden_size * 2, 1),
            'revenue_potential': nn.Linear(hidden_size * 2, 1),
            'urgency_score': nn.Linear(hidden_size * 2, 1),
            'engagement_score': nn.Linear(hidden_size * 2, 1),
            'behavioral_score': nn.Linear(hidden_size * 2, 1)
        })
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(hidden_size * 2, 2)  # mean, std
        
        # Feature importance extraction
        self.feature_importance = nn.Linear(hidden_size * 2, fusion_input_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        categorical_features: Optional[torch.Tensor] = None,
        behavioral_sequence: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Text encoding
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Numerical features encoding
        numerical_encoded = self.numerical_encoder(numerical_features)
        
        # Categorical features encoding
        categorical_encoded = []
        if categorical_features is not None and self.categorical_embeddings:
            for i, embedding_layer in enumerate(self.categorical_embeddings):
                embedded = embedding_layer(categorical_features[:, i])
                categorical_encoded.append(embedded)
            categorical_features_encoded = torch.cat(categorical_encoded, dim=1)
        else:
            categorical_features_encoded = torch.empty(batch_size, 0, device=device)
        
        # Behavioral sequence encoding
        if behavioral_sequence is not None:
            behavioral_encoded = self.behavioral_encoder(behavioral_sequence)
        else:
            behavioral_encoded = torch.zeros(batch_size, self.behavioral_encoder.attention.attention_weights.in_features, device=device)
        
        # Feature fusion
        fused_features = torch.cat([
            text_features,
            numerical_encoded,
            categorical_features_encoded,
            behavioral_encoded
        ], dim=1)
        
        fused_features = self.feature_fusion(fused_features)
        
        # Reshape for transformer (add sequence dimension)
        fused_features = fused_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # Apply transformer layers
        transformer_output = self.transformer_encoder(fused_features)
        final_features = transformer_output.squeeze(1)  # (batch_size, feature_dim)
        
        # Generate predictions
        predictions = {}
        for head_name, head_layer in self.output_heads.items():
            if head_name in ['conversion_probability', 'overall_score', 'urgency_score', 'engagement_score', 'behavioral_score']:
                predictions[head_name] = torch.sigmoid(head_layer(final_features))
            else:  # revenue_potential
                predictions[head_name] = F.relu(head_layer(final_features))
        
        # Confidence estimation
        confidence_params = self.confidence_head(final_features)
        confidence_mean = confidence_params[:, 0:1]
        confidence_std = F.softplus(confidence_params[:, 1:2]) + 1e-6
        
        # Feature importance
        feature_importance = torch.abs(self.feature_importance(final_features))
        feature_importance = F.softmax(feature_importance, dim=1)
        
        results = {
            'predictions': predictions,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'feature_importance': feature_importance,
            'final_features': final_features
        }
        
        if return_attention:
            # Extract attention weights from transformer
            results['attention_weights'] = transformer_output
        
        return results


class DeepLeadScoringEngine:
    """Advanced deep learning engine for lead scoring"""
    
    def __init__(
        self,
        model_name: str = 'advanced_lead_scorer_v1',
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[TransformerLeadScoringModel] = None
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = {}
        self.is_trained = False
        self.model_version = "1.0.0"
        
        # Model configuration
        self.config = {
            'text_model_name': 'distilbert-base-uncased',
            'max_sequence_length': 512,
            'numerical_feature_size': 50,
            'categorical_vocab_sizes': [100, 50, 20, 10],  # company_size, industry, region, source
            'categorical_embedding_dims': [32, 16, 8, 4],
            'behavioral_input_size': 20,
            'behavioral_sequence_length': 100,
            'hidden_size': 256,
            'num_attention_heads': 8,
            'num_transformer_layers': 4,
            'dropout': 0.2
        }
        
        logger.info(f"DeepLeadScoringEngine initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize the deep learning model and tokenizer"""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['text_model_name']
            )
            
            # Initialize model
            self.model = TransformerLeadScoringModel(
                text_model_name=self.config['text_model_name'],
                numerical_feature_size=self.config['numerical_feature_size'],
                categorical_vocab_sizes=self.config['categorical_vocab_sizes'],
                categorical_embedding_dims=self.config['categorical_embedding_dims'],
                behavioral_input_size=self.config['behavioral_input_size'],
                hidden_size=self.config['hidden_size'],
                num_attention_heads=self.config['num_attention_heads'],
                num_transformer_layers=self.config['num_transformer_layers'],
                dropout=self.config['dropout']
            )
            
            self.model.to(self.device)
            
            # Try to load pre-trained weights
            await self._load_model_if_exists()
            
            logger.info("DeepLeadScoringEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DeepLeadScoringEngine: {e}")
            raise
    
    async def train_model(
        self,
        training_data: pd.DataFrame,
        text_columns: List[str] = ['company_description', 'lead_notes', 'interaction_summary'],
        numerical_columns: List[str] = None,
        categorical_columns: List[str] = ['company_size', 'industry', 'region', 'source'],
        behavioral_columns: List[str] = None,
        target_columns: List[str] = ['conversion_probability', 'revenue_potential'],
        validation_split: float = 0.2,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 2e-5
    ):
        """Train the deep learning model"""
        try:
            logger.info(f"Starting model training with {len(training_data)} samples")
            
            # Prepare features
            text_data, numerical_features, categorical_features, behavioral_sequences, labels = \
                await self._prepare_training_data(
                    training_data,
                    text_columns,
                    numerical_columns,
                    categorical_columns,
                    behavioral_columns,
                    target_columns
                )
            
            # Create dataset
            dataset = MultiModalLeadDataset(
                text_data=text_data,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                behavioral_sequences=behavioral_sequences,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self.config['max_sequence_length']
            )
            
            # Split data
            train_size = int((1 - validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Initialize optimizer and loss function
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs
            )
            
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(num_epochs):
                train_loss = await self._train_epoch(
                    train_loader, optimizer, criterion
                )
                val_loss = await self._validate_epoch(val_loader, criterion)
                
                scheduler.step()
                
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    await self._save_model()
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            
            return training_history
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    async def predict_lead_score(
        self,
        lead_data: Dict[str, Any],
        return_confidence: bool = True,
        return_feature_importance: bool = True
    ) -> LeadScoringPrediction:
        """Generate comprehensive lead scoring prediction"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            self.model.eval()
            
            # Prepare input data
            processed_data = await self._prepare_prediction_data(lead_data)
            
            with torch.no_grad():
                # Make prediction
                results = self.model(
                    input_ids=processed_data['input_ids'].unsqueeze(0),
                    attention_mask=processed_data['attention_mask'].unsqueeze(0),
                    numerical_features=processed_data['numerical_features'].unsqueeze(0),
                    categorical_features=processed_data['categorical_features'].unsqueeze(0),
                    behavioral_sequence=processed_data['behavioral_sequence'].unsqueeze(0),
                    return_attention=True
                )
                
                predictions = results['predictions']
                
                # Extract individual scores
                overall_score = predictions['overall_score'].item()
                conversion_probability = predictions['conversion_probability'].item()
                revenue_potential = predictions['revenue_potential'].item()
                urgency_score = predictions['urgency_score'].item()
                engagement_score = predictions['engagement_score'].item()
                behavioral_score = predictions['behavioral_score'].item()
                
                # Calculate confidence interval
                if return_confidence:
                    confidence_mean = results['confidence_mean'].item()
                    confidence_std = results['confidence_std'].item()
                    confidence_interval = (
                        max(0, confidence_mean - 1.96 * confidence_std),
                        min(1, confidence_mean + 1.96 * confidence_std)
                    )
                else:
                    confidence_interval = (0.0, 1.0)
                
                # Extract feature importance
                if return_feature_importance:
                    feature_importance_tensor = results['feature_importance'].squeeze(0)
                    feature_importance = self._extract_feature_importance(
                        feature_importance_tensor,
                        lead_data
                    )
                else:
                    feature_importance = {}
                
                # Calculate demographic score (from categorical and numerical features)
                demographic_score = (
                    0.4 * overall_score +
                    0.3 * conversion_probability +
                    0.3 * revenue_potential
                )
            
            return LeadScoringPrediction(
                lead_id=lead_data.get('lead_id', 'unknown'),
                overall_score=overall_score,
                conversion_probability=conversion_probability,
                revenue_potential=revenue_potential,
                urgency_score=urgency_score,
                engagement_score=engagement_score,
                behavioral_score=behavioral_score,
                demographic_score=demographic_score,
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                prediction_timestamp=datetime.now(),
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Error making lead score prediction: {e}")
            raise
    
    async def batch_predict(
        self,
        leads_data: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[LeadScoringPrediction]:
        """Generate predictions for multiple leads efficiently"""
        try:
            predictions = []
            
            for i in range(0, len(leads_data), batch_size):
                batch = leads_data[i:i + batch_size]
                batch_predictions = []
                
                for lead_data in batch:
                    prediction = await self.predict_lead_score(
                        lead_data,
                        return_confidence=True,
                        return_feature_importance=False  # Skip for efficiency
                    )
                    batch_predictions.append(prediction)
                
                predictions.extend(batch_predictions)
            
            logger.info(f"Generated predictions for {len(predictions)} leads")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    async def _prepare_training_data(
        self,
        data: pd.DataFrame,
        text_columns: List[str],
        numerical_columns: Optional[List[str]],
        categorical_columns: List[str],
        behavioral_columns: Optional[List[str]],
        target_columns: List[str]
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for the model"""
        # Combine text columns
        text_data = []
        for _, row in data.iterrows():
            combined_text = " ".join([
                str(row.get(col, "")) for col in text_columns
            ])
            text_data.append(combined_text)
        
        # Prepare numerical features
        if numerical_columns:
            numerical_data = data[numerical_columns].fillna(0).values
            numerical_features = self.scaler.fit_transform(numerical_data)
        else:
            numerical_features = np.zeros((len(data), self.config['numerical_feature_size']))
        
        # Prepare categorical features
        categorical_features = np.zeros((len(data), len(categorical_columns)), dtype=int)
        for i, col in enumerate(categorical_columns):
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(data[col].fillna('unknown'))
            categorical_features[:, i] = self.label_encoders[col].transform(
                data[col].fillna('unknown')
            )
        
        # Prepare behavioral sequences
        if behavioral_columns:
            behavioral_sequences = data[behavioral_columns].fillna(0).values
            # Reshape to sequence format
            behavioral_sequences = behavioral_sequences.reshape(
                len(data), -1, self.config['behavioral_input_size']
            )
        else:
            behavioral_sequences = np.zeros((
                len(data),
                self.config['behavioral_sequence_length'],
                self.config['behavioral_input_size']
            ))
        
        # Prepare labels
        labels = data[target_columns].fillna(0).values
        
        return text_data, numerical_features, categorical_features, behavioral_sequences, labels
    
    async def _prepare_prediction_data(self, lead_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare single lead data for prediction"""
        # Combine text data
        text_columns = ['company_description', 'lead_notes', 'interaction_summary']
        combined_text = " ".join([
            str(lead_data.get(col, "")) for col in text_columns
        ])
        
        # Tokenize text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_sequence_length'],
            return_tensors='pt'
        )
        
        # Prepare numerical features
        numerical_features = np.zeros(self.config['numerical_feature_size'])
        # Fill with actual numerical data from lead_data
        
        # Prepare categorical features
        categorical_columns = ['company_size', 'industry', 'region', 'source']
        categorical_features = np.zeros(len(categorical_columns), dtype=int)
        for i, col in enumerate(categorical_columns):
            if col in self.label_encoders:
                value = lead_data.get(col, 'unknown')
                try:
                    categorical_features[i] = self.label_encoders[col].transform([value])[0]
                except ValueError:
                    categorical_features[i] = 0  # Unknown category
        
        # Prepare behavioral sequence
        behavioral_sequence = np.zeros((
            self.config['behavioral_sequence_length'],
            self.config['behavioral_input_size']
        ))
        
        return {
            'input_ids': encoding['input_ids'].flatten().to(self.device),
            'attention_mask': encoding['attention_mask'].flatten().to(self.device),
            'numerical_features': torch.FloatTensor(numerical_features).to(self.device),
            'categorical_features': torch.LongTensor(categorical_features).to(self.device),
            'behavioral_sequence': torch.FloatTensor(behavioral_sequence).to(self.device)
        }
    
    async def _train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            results = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                numerical_features=batch['numerical_features'],
                categorical_features=batch['categorical_features'],
                behavioral_sequence=batch['behavioral_sequence']
            )
            
            # Calculate loss
            predictions = results['predictions']
            targets = batch['labels']
            
            loss = 0
            for pred_key, pred_values in predictions.items():
                target_idx = {
                    'conversion_probability': 0,
                    'revenue_potential': 1
                }.get(pred_key, 0)
                
                if target_idx < targets.size(1):
                    loss += criterion(pred_values, targets[:, target_idx:target_idx+1])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    async def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                results = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    numerical_features=batch['numerical_features'],
                    categorical_features=batch['categorical_features'],
                    behavioral_sequence=batch['behavioral_sequence']
                )
                
                # Calculate loss
                predictions = results['predictions']
                targets = batch['labels']
                
                loss = 0
                for pred_key, pred_values in predictions.items():
                    target_idx = {
                        'conversion_probability': 0,
                        'revenue_potential': 1
                    }.get(pred_key, 0)
                    
                    if target_idx < targets.size(1):
                        loss += criterion(pred_values, targets[:, target_idx:target_idx+1])
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _extract_feature_importance(
        self,
        importance_tensor: torch.Tensor,
        lead_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract and map feature importance to named features"""
        importance_values = importance_tensor.cpu().numpy()
        
        # Map to feature names (this would be more sophisticated in practice)
        feature_names = [
            'text_content', 'company_size', 'industry', 'region', 'source',
            'engagement_history', 'behavioral_patterns', 'demographic_fit'
        ]
        
        # Take top features based on tensor size
        top_features = min(len(feature_names), len(importance_values))
        
        feature_importance = {}
        for i in range(top_features):
            feature_importance[feature_names[i]] = float(importance_values[i])
        
        return feature_importance
    
    async def _save_model(self):
        """Save model and training artifacts"""
        try:
            model_path = f"models/{self.model_name}_model.pt"
            config_path = f"models/{self.model_name}_config.json"
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler_state': self.scaler,
                'label_encoders': self.label_encoders,
                'config': self.config,
                'model_version': self.model_version
            }, model_path)
            
            # Save configuration
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    async def _load_model_if_exists(self):
        """Load pre-trained model if it exists"""
        try:
            model_path = f"models/{self.model_name}_model.pt"
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler_state']
            self.label_encoders = checkpoint['label_encoders']
            self.model_version = checkpoint.get('model_version', '1.0.0')
            
            self.is_trained = True
            logger.info(f"Loaded pre-trained model from {model_path}")
            
        except FileNotFoundError:
            logger.info("No pre-trained model found, starting fresh")
        except Exception as e:
            logger.warning(f"Error loading model: {e}, starting fresh")


# Global instance
deep_lead_scoring_engine = DeepLeadScoringEngine()