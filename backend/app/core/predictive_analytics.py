"""
Predictive Analytics for Sales Forecasting
Advanced ML algorithms for sales predictions, pipeline analysis, and revenue forecasting
"""

import asyncio
import json
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import pickle
import joblib
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum

# Machine Learning imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Time series analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Statistical analysis
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

class ForecastHorizon(Enum):
    """Forecast time horizons"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class PredictionType(Enum):
    """Types of predictions"""
    REVENUE_FORECAST = "revenue_forecast"
    DEAL_PROBABILITY = "deal_probability"
    PIPELINE_CONVERSION = "pipeline_conversion"
    SALES_VELOCITY = "sales_velocity"
    CUSTOMER_LIFETIME_VALUE = "customer_lifetime_value"
    CHURN_PREDICTION = "churn_prediction"

class ModelType(Enum):
    """ML model types"""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    NEURAL_NETWORK = "neural_network"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"

@dataclass
class ForecastResult:
    """Forecast result with confidence intervals"""
    prediction_type: PredictionType
    forecast_horizon: ForecastHorizon
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_score: float
    factors_contributing: Dict[str, float]
    model_used: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2_score: float
    mape: float  # Mean Absolute Percentage Error
    cross_val_score: float
    feature_importance: Dict[str, float]
    last_trained: datetime

@dataclass
class SalesPipelineInsight:
    """Sales pipeline analysis insights"""
    insight_type: str
    description: str
    current_value: float
    predicted_value: float
    change_percentage: float
    confidence: float
    recommendations: List[str]
    impact_level: str  # low, medium, high

class PredictiveAnalyticsEngine:
    """
    Advanced Predictive Analytics Engine for Sales Forecasting
    Combines multiple ML algorithms and time series analysis
    """
    
    def __init__(self, model_path: str = "/Users/rubeenakhan/Desktop/OS/ml-models/predictive"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Models storage
        self.models = defaultdict(dict)
        self.scalers = defaultdict(dict)
        self.encoders = defaultdict(dict)
        self.model_performance = {}
        
        # Data processing
        self.feature_columns = defaultdict(list)
        self.target_encoders = {}
        
        # Historical data storage
        self.historical_data = defaultdict(deque)
        self.data_cache = {}
        
        # Model configurations
        self.model_configs = {}
        self.ensemble_weights = {}
        
        # Initialize models
        self._initialize_models()
        self._initialize_feature_engineering()
    
    def _initialize_models(self):
        """Initialize all predictive models"""
        
        # Revenue Forecasting Models
        self.models[PredictionType.REVENUE_FORECAST] = {
            'linear': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42, verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=200, learning_rate=0.1, depth=8, random_seed=42, verbose=False
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu', 
                solver='adam', alpha=0.001, max_iter=500, random_state=42
            )
        }
        
        # Deal Probability Models
        self.models[PredictionType.DEAL_PROBABILITY] = {
            'logistic_ensemble': VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
            ]),
            'xgboost': xgb.XGBRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42
            )
        }
        
        # Pipeline Conversion Models
        self.models[PredictionType.PIPELINE_CONVERSION] = {
            'time_series_rf': RandomForestRegressor(
                n_estimators=150, max_depth=10, random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=300, random_state=42
            )
        }
        
        # Sales Velocity Models
        self.models[PredictionType.SALES_VELOCITY] = {
            'linear_trend': LinearRegression(),
            'exponential_smoothing': None,  # Will be initialized per dataset
            'arima': None,  # Will be fitted per time series
            'ensemble': VotingRegressor([
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
        }
        
        # Customer Lifetime Value Models
        self.models[PredictionType.CUSTOMER_LIFETIME_VALUE] = {
            'regression_ensemble': VotingRegressor([
                ('ridge', Ridge(alpha=1.0)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
            ]),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=400, random_state=42
            )
        }
        
        # Initialize scalers
        for prediction_type in PredictionType:
            self.scalers[prediction_type] = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': StandardScaler()  # Placeholder for RobustScaler
            }
        
        logger.info("Predictive models initialized successfully")
    
    def _initialize_feature_engineering(self):
        """Initialize feature engineering configurations"""
        
        # Revenue forecasting features
        self.feature_columns[PredictionType.REVENUE_FORECAST] = [
            'historical_revenue', 'pipeline_value', 'closed_deals_count',
            'new_leads_count', 'conversion_rate', 'average_deal_size',
            'sales_cycle_length', 'market_conditions', 'seasonality_factor',
            'team_performance_score', 'product_mix_score', 'competitive_pressure'
        ]
        
        # Deal probability features
        self.feature_columns[PredictionType.DEAL_PROBABILITY] = [
            'lead_score', 'engagement_score', 'company_size', 'deal_value',
            'sales_cycle_position', 'competitor_presence', 'budget_confirmed',
            'decision_maker_involved', 'previous_interactions', 'industry_type'
        ]
        
        # Pipeline conversion features
        self.feature_columns[PredictionType.PIPELINE_CONVERSION] = [
            'pipeline_stage', 'time_in_stage', 'deal_value', 'lead_source',
            'sales_rep_performance', 'customer_engagement', 'market_segment',
            'deal_complexity', 'seasonal_trends', 'economic_indicators'
        ]
        
        logger.info("Feature engineering configurations initialized")
    
    async def train_models(self, training_data: Dict[PredictionType, pd.DataFrame]) -> Dict[str, ModelPerformance]:
        """Train all predictive models with provided data"""
        logger.info("Starting predictive models training...")
        
        performance_results = {}
        
        for prediction_type, data in training_data.items():
            if data.empty:
                logger.warning(f"No training data for {prediction_type.value}")
                continue
            
            logger.info(f"Training models for {prediction_type.value}")
            
            # Prepare features and target
            X, y = await self._prepare_training_data(data, prediction_type)
            
            if X is None or y is None:
                logger.error(f"Failed to prepare training data for {prediction_type.value}")
                continue
            
            # Train models for this prediction type
            type_performance = await self._train_prediction_type_models(
                prediction_type, X, y
            )
            performance_results.update(type_performance)
        
        # Save trained models
        await self._save_models()
        
        logger.info("Predictive models training completed")
        return performance_results
    
    async def _prepare_training_data(self, data: pd.DataFrame, 
                                   prediction_type: PredictionType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with feature engineering"""
        try:
            # Feature engineering based on prediction type
            if prediction_type == PredictionType.REVENUE_FORECAST:
                X, y = await self._prepare_revenue_features(data)
            elif prediction_type == PredictionType.DEAL_PROBABILITY:
                X, y = await self._prepare_deal_probability_features(data)
            elif prediction_type == PredictionType.PIPELINE_CONVERSION:
                X, y = await self._prepare_pipeline_features(data)
            elif prediction_type == PredictionType.SALES_VELOCITY:
                X, y = await self._prepare_velocity_features(data)
            elif prediction_type == PredictionType.CUSTOMER_LIFETIME_VALUE:
                X, y = await self._prepare_clv_features(data)
            else:
                logger.error(f"Unknown prediction type: {prediction_type}")
                return None, None
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    async def _prepare_revenue_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for revenue forecasting"""
        features = []
        
        # Historical revenue trends
        if 'revenue' in data.columns:
            # Moving averages
            data['revenue_ma_3'] = data['revenue'].rolling(window=3).mean()
            data['revenue_ma_6'] = data['revenue'].rolling(window=6).mean()
            data['revenue_ma_12'] = data['revenue'].rolling(window=12).mean()
            
            # Revenue growth rates
            data['revenue_growth'] = data['revenue'].pct_change()
            data['revenue_growth_ma'] = data['revenue_growth'].rolling(window=3).mean()
            
            features.extend(['revenue_ma_3', 'revenue_ma_6', 'revenue_ma_12', 'revenue_growth', 'revenue_growth_ma'])
        
        # Pipeline metrics
        pipeline_features = ['pipeline_value', 'deals_in_pipeline', 'average_deal_size']
        for feature in pipeline_features:
            if feature in data.columns:
                features.append(feature)
        
        # Sales team metrics
        team_features = ['sales_rep_count', 'team_performance_score', 'new_hires']
        for feature in team_features:
            if feature in data.columns:
                features.append(feature)
        
        # Market and seasonal factors
        if 'date' in data.columns:
            data['month'] = pd.to_datetime(data['date']).dt.month
            data['quarter'] = pd.to_datetime(data['date']).dt.quarter
            data['is_quarter_end'] = data['month'].isin([3, 6, 9, 12]).astype(int)
            features.extend(['month', 'quarter', 'is_quarter_end'])
        
        # Economic indicators
        economic_features = ['market_conditions', 'competitor_activity', 'industry_growth']
        for feature in economic_features:
            if feature in data.columns:
                features.append(feature)
        
        # Prepare final feature matrix
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0).values
        
        # Target variable
        if 'future_revenue' in data.columns:
            y = data['future_revenue'].fillna(0).values
        elif 'revenue' in data.columns:
            # Shift revenue by 1 period for prediction
            y = data['revenue'].shift(-1).fillna(data['revenue'].mean()).values
        else:
            raise ValueError("No target variable found for revenue forecasting")
        
        return X, y
    
    async def _prepare_deal_probability_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for deal probability prediction"""
        features = []
        
        # Lead scoring features
        scoring_features = ['lead_score', 'engagement_score', 'qualification_score']
        for feature in scoring_features:
            if feature in data.columns:
                features.append(feature)
        
        # Company characteristics
        company_features = ['company_size', 'industry', 'annual_revenue', 'employee_count']
        for feature in company_features:
            if feature in data.columns:
                if data[feature].dtype == 'object':
                    # Encode categorical variables
                    encoder = LabelEncoder()
                    data[f'{feature}_encoded'] = encoder.fit_transform(data[feature].fillna('unknown'))
                    self.encoders[PredictionType.DEAL_PROBABILITY][feature] = encoder
                    features.append(f'{feature}_encoded')
                else:
                    features.append(feature)
        
        # Deal characteristics
        deal_features = ['deal_value', 'deal_stage', 'time_in_stage', 'sales_cycle_length']
        for feature in deal_features:
            if feature in data.columns:
                if data[feature].dtype == 'object':
                    encoder = LabelEncoder()
                    data[f'{feature}_encoded'] = encoder.fit_transform(data[feature].fillna('unknown'))
                    self.encoders[PredictionType.DEAL_PROBABILITY][feature] = encoder
                    features.append(f'{feature}_encoded')
                else:
                    features.append(feature)
        
        # Interaction features
        interaction_features = ['email_responses', 'meeting_count', 'demo_requested', 'proposal_sent']
        for feature in interaction_features:
            if feature in data.columns:
                features.append(feature)
        
        # Create interaction terms
        if 'deal_value' in data.columns and 'engagement_score' in data.columns:
            data['value_engagement_interaction'] = data['deal_value'] * data['engagement_score']
            features.append('value_engagement_interaction')
        
        # Prepare feature matrix
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0).values
        
        # Target variable (deal won/lost as probability)
        if 'deal_outcome' in data.columns:
            # Convert outcome to probability (1 for won, 0 for lost)
            y = (data['deal_outcome'] == 'won').astype(float).values
        elif 'close_probability' in data.columns:
            y = data['close_probability'].fillna(0.5).values
        else:
            raise ValueError("No target variable found for deal probability")
        
        return X, y
    
    async def _prepare_pipeline_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for pipeline conversion prediction"""
        features = []
        
        # Pipeline stage features
        if 'pipeline_stage' in data.columns:
            encoder = LabelEncoder()
            data['stage_encoded'] = encoder.fit_transform(data['pipeline_stage'].fillna('unknown'))
            self.encoders[PredictionType.PIPELINE_CONVERSION]['pipeline_stage'] = encoder
            features.append('stage_encoded')
        
        # Time-based features
        time_features = ['days_in_stage', 'days_since_created', 'total_sales_cycle_days']
        for feature in time_features:
            if feature in data.columns:
                features.append(feature)
        
        # Value and size features
        value_features = ['deal_value', 'pipeline_value', 'weighted_pipeline_value']
        for feature in value_features:
            if feature in data.columns:
                features.append(feature)
        
        # Activity features
        activity_features = ['touchpoints_count', 'last_activity_days', 'activity_score']
        for feature in activity_features:
            if feature in data.columns:
                features.append(feature)
        
        # Historical conversion rates
        if 'lead_source' in data.columns:
            # Calculate conversion rate by lead source
            conversion_by_source = data.groupby('lead_source')['converted'].mean()
            data['source_conversion_rate'] = data['lead_source'].map(conversion_by_source)
            features.append('source_conversion_rate')
        
        # Prepare feature matrix
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0).values
        
        # Target variable
        if 'conversion_outcome' in data.columns:
            y = data['conversion_outcome'].astype(float).values
        elif 'converted' in data.columns:
            y = data['converted'].astype(float).values
        else:
            raise ValueError("No target variable found for pipeline conversion")
        
        return X, y
    
    async def _prepare_velocity_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for sales velocity prediction"""
        features = []
        
        # Time series features
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['day_of_week'] = data['date'].dt.dayofweek
            data['week_of_year'] = data['date'].dt.isocalendar().week
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            features.extend(['day_of_week', 'week_of_year', 'month', 'quarter'])
        
        # Sales metrics
        sales_features = ['deals_closed', 'revenue_generated', 'average_deal_size', 'sales_cycle_length']
        for feature in sales_features:
            if feature in data.columns:
                features.append(feature)
                
                # Add moving averages
                data[f'{feature}_ma_7'] = data[feature].rolling(window=7).mean()
                data[f'{feature}_ma_30'] = data[feature].rolling(window=30).mean()
                features.extend([f'{feature}_ma_7', f'{feature}_ma_30'])
        
        # Team and activity features
        team_features = ['active_sales_reps', 'new_leads', 'qualified_leads', 'demos_given']
        for feature in team_features:
            if feature in data.columns:
                features.append(feature)
        
        # Lag features
        lag_features = ['revenue_generated', 'deals_closed']
        for feature in lag_features:
            if feature in data.columns:
                data[f'{feature}_lag_1'] = data[feature].shift(1)
                data[f'{feature}_lag_7'] = data[feature].shift(7)
                features.extend([f'{feature}_lag_1', f'{feature}_lag_7'])
        
        # Prepare feature matrix
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0).values
        
        # Target variable (sales velocity = deals closed / time period)
        if 'sales_velocity' in data.columns:
            y = data['sales_velocity'].fillna(0).values
        elif 'deals_closed' in data.columns and 'days_in_period' in data.columns:
            y = (data['deals_closed'] / data['days_in_period']).fillna(0).values
        else:
            raise ValueError("No target variable found for sales velocity")
        
        return X, y
    
    async def _prepare_clv_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for Customer Lifetime Value prediction"""
        features = []
        
        # Customer characteristics
        customer_features = ['acquisition_cost', 'first_purchase_value', 'purchase_frequency', 'tenure_days']
        for feature in customer_features:
            if feature in data.columns:
                features.append(feature)
        
        # Behavioral features
        behavior_features = ['avg_order_value', 'total_purchases', 'support_tickets', 'engagement_score']
        for feature in behavior_features:
            if feature in data.columns:
                features.append(feature)
        
        # Company features
        company_features = ['company_size', 'industry', 'plan_type', 'add_ons_count']
        for feature in company_features:
            if feature in data.columns:
                if data[feature].dtype == 'object':
                    encoder = LabelEncoder()
                    data[f'{feature}_encoded'] = encoder.fit_transform(data[feature].fillna('unknown'))
                    self.encoders[PredictionType.CUSTOMER_LIFETIME_VALUE][feature] = encoder
                    features.append(f'{feature}_encoded')
                else:
                    features.append(feature)
        
        # Derived features
        if 'total_revenue' in data.columns and 'tenure_days' in data.columns:
            data['revenue_per_day'] = data['total_revenue'] / np.maximum(data['tenure_days'], 1)
            features.append('revenue_per_day')
        
        if 'total_purchases' in data.columns and 'tenure_days' in data.columns:
            data['purchase_frequency'] = data['total_purchases'] / np.maximum(data['tenure_days'], 1) * 365
            features.append('purchase_frequency')
        
        # Prepare feature matrix
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0).values
        
        # Target variable
        if 'customer_lifetime_value' in data.columns:
            y = data['customer_lifetime_value'].fillna(0).values
        elif 'total_revenue' in data.columns:
            y = data['total_revenue'].fillna(0).values
        else:
            raise ValueError("No target variable found for CLV prediction")
        
        return X, y
    
    async def _train_prediction_type_models(self, prediction_type: PredictionType, 
                                          X: np.ndarray, y: np.ndarray) -> Dict[str, ModelPerformance]:
        """Train models for specific prediction type"""
        performance_results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = self.scalers[prediction_type]['standard']
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train each model
        for model_name, model in self.models[prediction_type].items():
            if model is None:
                continue
                
            try:
                logger.info(f"Training {model_name} for {prediction_type.value}")
                
                # Use scaled data for neural networks, original for tree-based
                if 'neural' in model_name.lower():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate performance metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # MAPE calculation (avoiding division by zero)
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
                )
                cv_score = -cv_scores.mean()
                
                # Feature importance (if available)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_values = model.feature_importances_
                    feature_importance = {
                        f'feature_{i}': float(imp) for i, imp in enumerate(importance_values)
                    }
                elif hasattr(model, 'coef_'):
                    importance_values = np.abs(model.coef_)
                    feature_importance = {
                        f'feature_{i}': float(imp) for i, imp in enumerate(importance_values)
                    }
                
                # Store performance
                model_key = f"{prediction_type.value}_{model_name}"
                performance = ModelPerformance(
                    model_name=model_key,
                    mae=mae,
                    rmse=rmse,
                    r2_score=r2,
                    mape=mape,
                    cross_val_score=cv_score,
                    feature_importance=feature_importance,
                    last_trained=datetime.now()
                )
                
                performance_results[model_key] = performance
                self.model_performance[model_key] = performance
                
                logger.info(f"{model_key} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {prediction_type.value}: {e}")
                continue
        
        return performance_results
    
    async def make_prediction(self, prediction_type: PredictionType, 
                            input_data: Dict[str, Any],
                            forecast_horizon: ForecastHorizon = ForecastHorizon.MONTHLY) -> ForecastResult:
        """Make prediction using trained models"""
        try:
            # Prepare input features
            features = await self._prepare_prediction_features(prediction_type, input_data)
            
            if features is None:
                raise ValueError("Failed to prepare features for prediction")
            
            # Get predictions from all models
            predictions = []
            model_votes = {}
            
            for model_name, model in self.models[prediction_type].items():
                if model is None:
                    continue
                    
                try:
                    # Use appropriate scaler
                    if 'neural' in model_name.lower():
                        scaler = self.scalers[prediction_type]['standard']
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        pred = model.predict(features_scaled)[0]
                    else:
                        pred = model.predict(features.reshape(1, -1))[0]
                    
                    predictions.append(pred)
                    model_votes[model_name] = float(pred)
                    
                except Exception as e:
                    logger.error(f"Error making prediction with {model_name}: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Ensemble prediction
            final_prediction = np.mean(predictions)
            prediction_std = np.std(predictions)
            
            # Calculate confidence interval
            confidence_interval = (
                final_prediction - 1.96 * prediction_std,
                final_prediction + 1.96 * prediction_std
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(predictions, prediction_std)
            
            # Get contributing factors
            contributing_factors = await self._analyze_contributing_factors(
                prediction_type, features, final_prediction
            )
            
            # Determine best model
            best_model = self._get_best_model(prediction_type)
            
            return ForecastResult(
                prediction_type=prediction_type,
                forecast_horizon=forecast_horizon,
                predicted_value=float(final_prediction),
                confidence_interval_lower=float(confidence_interval[0]),
                confidence_interval_upper=float(confidence_interval[1]),
                confidence_score=confidence_score,
                factors_contributing=contributing_factors,
                model_used=best_model,
                timestamp=datetime.now(),
                metadata={'model_votes': model_votes, 'ensemble_std': float(prediction_std)}
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._get_fallback_forecast(prediction_type, forecast_horizon)
    
    async def _prepare_prediction_features(self, prediction_type: PredictionType, 
                                         input_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        try:
            feature_columns = self.feature_columns[prediction_type]
            features = []
            
            for feature_name in feature_columns:
                if feature_name in input_data:
                    features.append(input_data[feature_name])
                elif f"{feature_name}_encoded" in input_data:
                    features.append(input_data[f"{feature_name}_encoded"])
                else:
                    # Use default value or mean
                    features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return None
    
    def _calculate_confidence_score(self, predictions: List[float], std: float) -> float:
        """Calculate confidence score based on model agreement"""
        if len(predictions) <= 1:
            return 0.5
        
        # Lower standard deviation = higher confidence
        # Normalize to 0-1 range
        max_std = np.mean(np.abs(predictions)) * 0.5  # 50% of mean as max std
        confidence = max(0.0, 1.0 - (std / max_std))
        
        return min(confidence, 1.0)
    
    async def _analyze_contributing_factors(self, prediction_type: PredictionType,
                                          features: np.ndarray, 
                                          prediction: float) -> Dict[str, float]:
        """Analyze factors contributing to prediction"""
        try:
            # Get feature importance from best performing model
            best_model_name = self._get_best_model(prediction_type)
            
            if best_model_name in self.model_performance:
                performance = self.model_performance[best_model_name]
                feature_importance = performance.feature_importance
                
                # Calculate contributions
                contributions = {}
                total_importance = sum(feature_importance.values())
                
                if total_importance > 0:
                    for i, (feature_name, importance) in enumerate(feature_importance.items()):
                        if i < len(features):
                            # Contribution = feature_value * importance * prediction_influence
                            contribution = (features[i] * importance / total_importance) * prediction * 0.1
                            contributions[feature_name] = float(contribution)
                
                return contributions
            
            return {}
            
        except Exception as e:
            logger.error(f"Error analyzing contributing factors: {e}")
            return {}
    
    def _get_best_model(self, prediction_type: PredictionType) -> str:
        """Get the best performing model for prediction type"""
        try:
            # Find model with highest R² score
            best_model = None
            best_score = -np.inf
            
            for model_name, performance in self.model_performance.items():
                if prediction_type.value in model_name and performance.r2_score > best_score:
                    best_score = performance.r2_score
                    best_model = model_name
            
            return best_model or f"{prediction_type.value}_ensemble"
            
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return f"{prediction_type.value}_default"
    
    async def generate_sales_insights(self, historical_data: Dict[str, Any]) -> List[SalesPipelineInsight]:
        """Generate sales pipeline insights using predictive analytics"""
        insights = []
        
        try:
            # Revenue forecast insight
            if 'revenue' in historical_data:
                current_revenue = historical_data['revenue']
                
                # Predict next period revenue
                revenue_prediction = await self.make_prediction(
                    PredictionType.REVENUE_FORECAST,
                    historical_data,
                    ForecastHorizon.MONTHLY
                )
                
                change_pct = ((revenue_prediction.predicted_value - current_revenue) / current_revenue) * 100
                
                insights.append(SalesPipelineInsight(
                    insight_type="revenue_forecast",
                    description=f"Revenue is predicted to {'increase' if change_pct > 0 else 'decrease'} by {abs(change_pct):.1f}%",
                    current_value=current_revenue,
                    predicted_value=revenue_prediction.predicted_value,
                    change_percentage=change_pct,
                    confidence=revenue_prediction.confidence_score,
                    recommendations=self._get_revenue_recommendations(change_pct),
                    impact_level="high" if abs(change_pct) > 10 else "medium"
                ))
            
            # Pipeline conversion insight
            if 'pipeline_value' in historical_data:
                pipeline_prediction = await self.make_prediction(
                    PredictionType.PIPELINE_CONVERSION,
                    historical_data
                )
                
                insights.append(SalesPipelineInsight(
                    insight_type="pipeline_efficiency",
                    description=f"Pipeline conversion rate predicted at {pipeline_prediction.predicted_value:.1%}",
                    current_value=historical_data.get('current_conversion_rate', 0),
                    predicted_value=pipeline_prediction.predicted_value,
                    change_percentage=0.0,  # Would need historical comparison
                    confidence=pipeline_prediction.confidence_score,
                    recommendations=self._get_pipeline_recommendations(pipeline_prediction.predicted_value),
                    impact_level="medium"
                ))
            
            # Sales velocity insight
            velocity_prediction = await self.make_prediction(
                PredictionType.SALES_VELOCITY,
                historical_data
            )
            
            insights.append(SalesPipelineInsight(
                insight_type="sales_velocity",
                description=f"Sales velocity predicted at {velocity_prediction.predicted_value:.2f} deals per day",
                current_value=historical_data.get('current_velocity', 0),
                predicted_value=velocity_prediction.predicted_value,
                change_percentage=0.0,
                confidence=velocity_prediction.confidence_score,
                recommendations=self._get_velocity_recommendations(velocity_prediction.predicted_value),
                impact_level="medium"
            ))
            
        except Exception as e:
            logger.error(f"Error generating sales insights: {e}")
        
        return insights
    
    def _get_revenue_recommendations(self, change_pct: float) -> List[str]:
        """Get recommendations based on revenue forecast"""
        recommendations = []
        
        if change_pct > 15:
            recommendations.extend([
                "Strong growth predicted - consider scaling sales team",
                "Ensure adequate inventory/capacity for increased demand",
                "Review pricing strategy to maximize revenue opportunity"
            ])
        elif change_pct > 5:
            recommendations.extend([
                "Positive growth trend - maintain current sales strategies",
                "Consider expanding successful marketing channels",
                "Monitor conversion rates to sustain growth"
            ])
        elif change_pct < -10:
            recommendations.extend([
                "Revenue decline predicted - investigate root causes",
                "Increase sales activities and lead generation",
                "Review and optimize pricing and product offerings",
                "Consider promotional campaigns to boost sales"
            ])
        else:
            recommendations.extend([
                "Stable revenue trend - focus on optimization",
                "Look for opportunities to improve efficiency",
                "Maintain current successful strategies"
            ])
        
        return recommendations
    
    def _get_pipeline_recommendations(self, conversion_rate: float) -> List[str]:
        """Get recommendations based on pipeline conversion prediction"""
        recommendations = []
        
        if conversion_rate > 0.8:
            recommendations.extend([
                "Excellent conversion rate - share best practices across team",
                "Focus on higher-value opportunities",
                "Consider increasing pipeline volume"
            ])
        elif conversion_rate > 0.6:
            recommendations.extend([
                "Good conversion rate - identify top-performing strategies",
                "Optimize sales process for consistent results",
                "Train team on successful conversion techniques"
            ])
        elif conversion_rate > 0.4:
            recommendations.extend([
                "Average conversion rate - analyze conversion bottlenecks",
                "Improve lead qualification process",
                "Enhance sales training and support"
            ])
        else:
            recommendations.extend([
                "Low conversion rate - urgent review needed",
                "Revise lead qualification criteria",
                "Intensive sales training required",
                "Review and optimize entire sales process"
            ])
        
        return recommendations
    
    def _get_velocity_recommendations(self, velocity: float) -> List[str]:
        """Get recommendations based on sales velocity prediction"""
        recommendations = []
        
        if velocity > 2.0:
            recommendations.extend([
                "High sales velocity - excellent performance",
                "Scale successful processes to maintain momentum",
                "Consider expanding to new markets or segments"
            ])
        elif velocity > 1.0:
            recommendations.extend([
                "Good sales velocity - optimize for consistency",
                "Identify and replicate high-velocity patterns",
                "Focus on shortening sales cycle length"
            ])
        elif velocity > 0.5:
            recommendations.extend([
                "Moderate sales velocity - room for improvement",
                "Analyze sales process bottlenecks",
                "Increase sales activity and follow-up frequency"
            ])
        else:
            recommendations.extend([
                "Low sales velocity - immediate action required",
                "Review entire sales process and methodology",
                "Increase lead generation and qualification efforts",
                "Provide intensive sales training and coaching"
            ])
        
        return recommendations
    
    async def _save_models(self):
        """Save trained models and components"""
        try:
            # Save models
            for prediction_type, models in self.models.items():
                type_path = self.model_path / prediction_type.value
                type_path.mkdir(exist_ok=True)
                
                for model_name, model in models.items():
                    if model is not None:
                        model_file = type_path / f"{model_name}_model.pkl"
                        joblib.dump(model, model_file)
            
            # Save scalers and encoders
            for prediction_type, scalers in self.scalers.items():
                type_path = self.model_path / prediction_type.value
                type_path.mkdir(exist_ok=True)
                
                for scaler_name, scaler in scalers.items():
                    scaler_file = type_path / f"{scaler_name}_scaler.pkl"
                    joblib.dump(scaler, scaler_file)
            
            # Save performance metrics
            performance_file = self.model_path / "model_performance.pkl"
            joblib.dump(self.model_performance, performance_file)
            
            logger.info("Predictive models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load trained models and components"""
        try:
            # Load models
            for prediction_type in PredictionType:
                type_path = self.model_path / prediction_type.value
                if type_path.exists():
                    for model_file in type_path.glob("*_model.pkl"):
                        model_name = model_file.stem.replace("_model", "")
                        self.models[prediction_type][model_name] = joblib.load(model_file)
            
            # Load scalers
            for prediction_type in PredictionType:
                type_path = self.model_path / prediction_type.value
                if type_path.exists():
                    for scaler_file in type_path.glob("*_scaler.pkl"):
                        scaler_name = scaler_file.stem.replace("_scaler", "")
                        self.scalers[prediction_type][scaler_name] = joblib.load(scaler_file)
            
            # Load performance metrics
            performance_file = self.model_path / "model_performance.pkl"
            if performance_file.exists():
                self.model_performance = joblib.load(performance_file)
            
            logger.info("Predictive models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _get_fallback_forecast(self, prediction_type: PredictionType, 
                             forecast_horizon: ForecastHorizon) -> ForecastResult:
        """Fallback forecast when models fail"""
        return ForecastResult(
            prediction_type=prediction_type,
            forecast_horizon=forecast_horizon,
            predicted_value=0.0,
            confidence_interval_lower=0.0,
            confidence_interval_upper=0.0,
            confidence_score=0.1,
            factors_contributing={},
            model_used="fallback",
            timestamp=datetime.now(),
            metadata={'error': 'Model prediction failed - using fallback'}
        )

# Global instance
predictive_analytics = PredictiveAnalyticsEngine()