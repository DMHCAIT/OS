"""
AI Training and Continuous Learning Pipeline
Advanced machine learning system that learns from your sales data
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import joblib
import os
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training metrics for model performance"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    data_size: int
    feature_count: int
    cross_val_score: float

@dataclass
class LearningInsight:
    """Insights from continuous learning"""
    insight_type: str
    description: str
    confidence: float
    impact_score: float
    recommendation: str
    data_points: int

class ContinuousLearningEngine:
    """
    Advanced machine learning engine that continuously learns from sales data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_extractors = {}
        self.training_history = []
        self.learning_insights = []
        
        # Model storage paths
        self.model_dir = Path("./models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Learning configuration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.retrain_threshold = 1000  # Retrain after 1000 new data points
        self.performance_threshold = 0.1  # Retrain if performance drops by 10%
        
        # Initialize model registry
        self.model_registry = {
            "lead_conversion": {
                "model_type": "classification",
                "features": ["engagement_score", "company_size", "industry", "title", "source"],
                "target": "converted"
            },
            "deal_size_prediction": {
                "model_type": "regression", 
                "features": ["company_size", "industry", "title", "budget_range", "urgency"],
                "target": "deal_amount"
            },
            "churn_prediction": {
                "model_type": "classification",
                "features": ["last_contact", "engagement_trend", "support_tickets", "usage_pattern"],
                "target": "churned"
            },
            "optimal_contact_time": {
                "model_type": "regression",
                "features": ["timezone", "industry", "role", "previous_responses"],
                "target": "response_probability"
            },
            "email_optimization": {
                "model_type": "classification",
                "features": ["subject_line", "content_length", "personalization_score", "send_time"],
                "target": "opened"
            }
        }
        
        logger.info("Continuous Learning Engine initialized")
    
    async def initialize_models(self):
        """Initialize all ML models"""
        try:
            for model_name, config in self.model_registry.items():
                await self._load_or_create_model(model_name, config)
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    async def _load_or_create_model(self, model_name: str, config: Dict[str, Any]):
        """Load existing model or create new one"""
        model_path = self.model_dir / f"{model_name}.joblib"
        
        if model_path.exists():
            # Load existing model
            model_data = joblib.load(model_path)
            self.models[model_name] = model_data["model"]
            self.scalers[model_name] = model_data.get("scaler")
            self.encoders[model_name] = model_data.get("encoder")
            
            logger.info(f"Loaded existing model: {model_name}")
        else:
            # Create new model
            if config["model_type"] == "classification":
                model = self._create_ensemble_classifier()
            else:
                model = self._create_ensemble_regressor()
            
            self.models[model_name] = model
            self.scalers[model_name] = StandardScaler()
            self.encoders[model_name] = {}
            
            logger.info(f"Created new model: {model_name}")
    
    def _create_ensemble_classifier(self):
        """Create ensemble classification model"""
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(random_state=42)
        xgb_clf = xgb.XGBClassifier(random_state=42)
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('xgb', xgb_clf)],
            voting='soft'
        )
        
        return ensemble
    
    def _create_ensemble_regressor(self):
        """Create ensemble regression model"""
        from sklearn.ensemble import VotingRegressor
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gbr = GradientBoostingRegressor(random_state=42)
        xgb_reg = xgb.XGBRegressor(random_state=42)
        
        ensemble = VotingRegressor(
            estimators=[('rf', rf), ('gbr', gbr), ('xgb', xgb_reg)]
        )
        
        return ensemble
    
    async def train_model(self, model_name: str, training_data: List[Dict[str, Any]]) -> TrainingMetrics:
        """Train or retrain a specific model"""
        try:
            logger.info(f"Training model: {model_name}")
            start_time = datetime.now()
            
            # Prepare training data
            df = pd.DataFrame(training_data)
            config = self.model_registry[model_name]
            
            # Feature engineering
            X, y = await self._prepare_features(df, config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            predictions = model.predict(X_test_scaled)
            
            if config["model_type"] == "classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            else:
                accuracy = r2_score(y_test, predictions)
                precision = mean_absolute_error(y_test, predictions)
                recall = mean_squared_error(y_test, predictions)
                f1 = np.sqrt(recall)  # RMSE
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cv_score = cv_scores.mean()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            await self._save_model(model_name)
            
            metrics = TrainingMetrics(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                data_size=len(training_data),
                feature_count=X.shape[1],
                cross_val_score=cv_score
            )
            
            self.training_history.append(metrics)
            
            logger.info(f"Model training completed: {model_name}")
            logger.info(f"Accuracy: {accuracy:.3f}, CV Score: {cv_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            raise
    
    async def _prepare_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        features = config["features"]
        target = config["target"]
        
        # Extract features
        X_data = []
        for feature in features:
            if feature in df.columns:
                if df[feature].dtype == 'object':
                    # Encode categorical features
                    if feature not in self.encoders[config.get("model_name", "default")]:
                        encoder = LabelEncoder()
                        self.encoders[config.get("model_name", "default")][feature] = encoder
                    else:
                        encoder = self.encoders[config.get("model_name", "default")][feature]
                    
                    # Handle unseen categories
                    try:
                        encoded = encoder.fit_transform(df[feature].fillna("unknown"))
                    except ValueError:
                        encoded = encoder.transform(df[feature].fillna("unknown"))
                    
                    X_data.append(encoded)
                else:
                    X_data.append(df[feature].fillna(0).values)
            else:
                # Create dummy feature if missing
                X_data.append(np.zeros(len(df)))
        
        X = np.column_stack(X_data)
        y = df[target].values
        
        return X, y
    
    async def _save_model(self, model_name: str):
        """Save model to disk"""
        model_path = self.model_dir / f"{model_name}.joblib"
        
        model_data = {
            "model": self.models[model_name],
            "scaler": self.scalers[model_name],
            "encoder": self.encoders[model_name],
            "saved_at": datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved: {model_path}")
    
    async def predict(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            config = self.model_registry[model_name]
            
            # Prepare features
            X_data = []
            for feature in config["features"]:
                value = features.get(feature, 0)
                
                if isinstance(value, str) and feature in self.encoders[model_name]:
                    encoder = self.encoders[model_name][feature]
                    try:
                        encoded_value = encoder.transform([value])[0]
                    except ValueError:
                        encoded_value = 0  # Unknown category
                    X_data.append(encoded_value)
                else:
                    X_data.append(float(value) if value is not None else 0.0)
            
            X = np.array(X_data).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scalers[model_name].transform(X)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_scaled)[0]
            
            # Get prediction probability/confidence
            confidence = 0.0
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {str(e)}")
            raise
    
    async def continuous_learning_loop(self):
        """Main continuous learning loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check if retraining is needed
                for model_name in self.model_registry.keys():
                    if await self._should_retrain(model_name):
                        logger.info(f"Retraining model: {model_name}")
                        
                        # Get new training data
                        training_data = await self._get_recent_training_data(model_name)
                        
                        if len(training_data) >= self.retrain_threshold:
                            metrics = await self.train_model(model_name, training_data)
                            
                            # Generate learning insights
                            insights = await self._analyze_learning_patterns(model_name, metrics)
                            self.learning_insights.extend(insights)
                
                # Clean up old insights
                self._cleanup_old_insights()
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {str(e)}")
                await asyncio.sleep(3600)  # Wait an hour before retrying
    
    async def _should_retrain(self, model_name: str) -> bool:
        """Check if model should be retrained"""
        # Check data volume
        new_data_count = await self._count_new_data_points(model_name)
        if new_data_count >= self.retrain_threshold:
            return True
        
        # Check performance degradation
        current_performance = await self._evaluate_current_performance(model_name)
        if current_performance and len(self.training_history) > 0:
            last_performance = self.training_history[-1].accuracy
            if (last_performance - current_performance) > self.performance_threshold:
                return True
        
        return False
    
    async def _count_new_data_points(self, model_name: str) -> int:
        """Count new data points since last training"""
        # This would query your database for new data
        # For now, return a simulated count
        return np.random.randint(0, 2000)
    
    async def _evaluate_current_performance(self, model_name: str) -> float:
        """Evaluate current model performance on recent data"""
        # This would evaluate model on recent data
        # For now, return a simulated performance score
        return np.random.uniform(0.7, 0.95)
    
    async def _get_recent_training_data(self, model_name: str) -> List[Dict[str, Any]]:
        """Get recent data for training"""
        # This would query your database for recent data
        # For now, return simulated data
        return []
    
    async def _analyze_learning_patterns(self, model_name: str, metrics: TrainingMetrics) -> List[LearningInsight]:
        """Analyze learning patterns and generate insights"""
        insights = []
        
        # Performance improvement insight
        if len(self.training_history) > 1:
            previous_accuracy = self.training_history[-2].accuracy
            improvement = metrics.accuracy - previous_accuracy
            
            if improvement > 0.05:
                insights.append(LearningInsight(
                    insight_type="performance_improvement",
                    description=f"Model {model_name} improved by {improvement:.1%}",
                    confidence=0.9,
                    impact_score=improvement * 100,
                    recommendation="Continue current training strategy",
                    data_points=metrics.data_size
                ))
        
        # Feature importance insight
        if hasattr(self.models[model_name], 'feature_importances_'):
            importances = self.models[model_name].feature_importances_
            top_feature_idx = np.argmax(importances)
            config = self.model_registry[model_name]
            top_feature = config["features"][top_feature_idx]
            
            insights.append(LearningInsight(
                insight_type="feature_importance",
                description=f"Feature '{top_feature}' is most predictive for {model_name}",
                confidence=0.8,
                impact_score=importances[top_feature_idx] * 100,
                recommendation=f"Focus on improving '{top_feature}' data quality",
                data_points=metrics.data_size
            ))
        
        return insights
    
    def _cleanup_old_insights(self):
        """Remove old learning insights"""
        cutoff_date = datetime.now() - timedelta(days=30)
        self.learning_insights = [
            insight for insight in self.learning_insights 
            if hasattr(insight, 'timestamp') and insight.timestamp > cutoff_date
        ]
    
    async def get_learning_insights(self) -> List[Dict[str, Any]]:
        """Get current learning insights"""
        return [asdict(insight) for insight in self.learning_insights]
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_name in self.model_registry.keys():
            if model_name in self.models:
                # Get latest training metrics
                model_metrics = [m for m in self.training_history if m.model_name == model_name]
                if model_metrics:
                    latest = model_metrics[-1]
                    performance[model_name] = {
                        "accuracy": latest.accuracy,
                        "last_trained": latest.training_time,
                        "data_size": latest.data_size,
                        "cross_val_score": latest.cross_val_score
                    }
        
        return performance
    
    async def export_model(self, model_name: str, format: str = "onnx") -> str:
        """Export model for deployment"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        export_path = self.model_dir / f"{model_name}_export.{format}"
        
        if format == "onnx":
            # Export to ONNX format for production deployment
            try:
                import skl2onnx
                from skl2onnx.common.data_types import FloatTensorType
                
                model = self.models[model_name]
                config = self.model_registry[model_name]
                
                initial_type = [('float_input', FloatTensorType([None, len(config["features"])]))]
                onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
                
                with open(export_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                logger.info(f"Model exported to ONNX: {export_path}")
                
            except ImportError:
                logger.warning("skl2onnx not installed, exporting as joblib instead")
                joblib.dump(self.models[model_name], export_path.with_suffix('.joblib'))
        
        return str(export_path)

# Global instance
continuous_learning_engine = ContinuousLearningEngine()

async def start_learning_engine():
    """Initialize and start the continuous learning engine"""
    await continuous_learning_engine.initialize_models()
    # Start the continuous learning loop in background
    asyncio.create_task(continuous_learning_engine.continuous_learning_loop())