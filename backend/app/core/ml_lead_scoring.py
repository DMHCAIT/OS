"""
Machine Learning Lead Scoring System
Advanced algorithms for predicting lead conversion probability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio
import joblib
import os
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

@dataclass
class MLModelMetrics:
    """ML model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_mean: float
    cross_val_std: float

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    rank: int
    category: str

@dataclass
class LeadScoringPrediction:
    """Complete lead scoring prediction"""
    lead_id: str
    overall_score: int  # 0-100
    conversion_probability: float  # 0-1
    confidence_interval: Tuple[float, float]
    feature_contributions: Dict[str, float]
    model_ensemble_votes: Dict[str, float]
    predicted_deal_value: float
    predicted_conversion_timeline: int  # days
    risk_assessment: Dict[str, float]
    recommended_actions: List[str]
    segment: str
    priority_tier: str

class MLLeadScoringSystem:
    """
    Advanced Machine Learning Lead Scoring System
    Uses ensemble methods and deep learning for accurate predictions
    """
    
    def __init__(self, model_path: str = "/Users/rubeenakhan/Desktop/OS/ml-models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_encoder = LabelEncoder()
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.pca = PCA(n_components=10)
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        
        # Model metadata
        self.model_metrics = {}
        self.feature_importance = {}
        self.training_history = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML model ensemble"""
        
        # Random Forest - Good for feature importance
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - Strong performance
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # XGBoost - High performance
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # LightGBM - Fast and accurate
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            verbose=-1
        )
        
        # CatBoost - Handles categorical features well
        self.models['catboost'] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=8,
            random_seed=42,
            verbose=False
        )
        
        # Neural Network - Complex patterns
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Ensemble model
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('gb', self.models['gradient_boost']),
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm'])
            ],
            voting='soft'
        )
        
        # Scalers
        self.scalers['numerical'] = StandardScaler()
        self.scalers['behavioral'] = StandardScaler()
        
        logger.info("ML models initialized successfully")
    
    async def extract_features(self, lead_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract comprehensive features for ML models
        """
        features = []
        
        # Demographic features
        demographic_features = self._extract_demographic_features(lead_data)
        features.extend(demographic_features)
        
        # Behavioral features
        behavioral_features = self._extract_behavioral_features(lead_data)
        features.extend(behavioral_features)
        
        # Engagement features
        engagement_features = self._extract_engagement_features(lead_data)
        features.extend(engagement_features)
        
        # Company features
        company_features = self._extract_company_features(lead_data)
        features.extend(company_features)
        
        # Interaction features
        interaction_features = self._extract_interaction_features(lead_data)
        features.extend(interaction_features)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(lead_data)
        features.extend(temporal_features)
        
        # Text features (if available)
        text_features = await self._extract_text_features(lead_data)
        features.extend(text_features)
        
        return np.array(features)
    
    def _extract_demographic_features(self, lead_data: Dict) -> List[float]:
        """Extract demographic-based features"""
        features = []
        
        # Job title seniority
        job_title = lead_data.get('job_title', '').lower()
        seniority_keywords = {
            'executive': ['ceo', 'cto', 'cfo', 'president', 'founder'],
            'director': ['director', 'vp', 'vice president'],
            'manager': ['manager', 'head', 'lead'],
            'individual': ['specialist', 'analyst', 'coordinator']
        }
        
        seniority_score = 0
        for level, keywords in seniority_keywords.items():
            if any(keyword in job_title for keyword in keywords):
                seniority_score = list(seniority_keywords.keys()).index(level) + 1
                break
        
        features.append(seniority_score / 4.0)  # Normalize 0-1
        
        # Industry scoring
        industry = lead_data.get('industry', '').lower()
        high_value_industries = ['technology', 'finance', 'healthcare', 'manufacturing', 'enterprise']
        industry_score = 1.0 if any(ind in industry for ind in high_value_industries) else 0.5
        features.append(industry_score)
        
        # Company size impact
        company_size = lead_data.get('company_size', 0)
        if company_size > 1000:
            size_score = 1.0
        elif company_size > 100:
            size_score = 0.8
        elif company_size > 50:
            size_score = 0.6
        else:
            size_score = 0.4
        features.append(size_score)
        
        # Location factor
        location = lead_data.get('location', '').lower()
        major_markets = ['new york', 'san francisco', 'london', 'singapore', 'toronto']
        location_score = 1.0 if any(market in location for market in major_markets) else 0.7
        features.append(location_score)
        
        return features
    
    def _extract_behavioral_features(self, lead_data: Dict) -> List[float]:
        """Extract behavioral pattern features"""
        features = []
        
        # Email engagement
        email_opens = lead_data.get('email_opens', 0)
        email_clicks = lead_data.get('email_clicks', 0)
        emails_sent = max(lead_data.get('emails_sent', 1), 1)
        
        open_rate = email_opens / emails_sent
        click_rate = email_clicks / emails_sent
        
        features.extend([open_rate, click_rate])
        
        # Website behavior
        page_views = lead_data.get('page_views', 0)
        session_duration = lead_data.get('avg_session_duration', 0)  # seconds
        bounce_rate = lead_data.get('bounce_rate', 1.0)
        
        engagement_score = (page_views * 0.3 + 
                          min(session_duration / 300, 1.0) * 0.4 + 
                          (1 - bounce_rate) * 0.3)
        
        features.append(engagement_score)
        
        # Download activity
        downloads = lead_data.get('content_downloads', 0)
        whitepaper_downloads = lead_data.get('whitepaper_downloads', 0)
        demo_requests = lead_data.get('demo_requests', 0)
        
        features.extend([
            min(downloads / 5, 1.0),  # Normalize downloads
            min(whitepaper_downloads / 3, 1.0),
            min(demo_requests / 2, 1.0)
        ])
        
        # Social media engagement
        social_shares = lead_data.get('social_shares', 0)
        linkedin_connections = lead_data.get('linkedin_connections', 0)
        
        features.extend([
            min(social_shares / 10, 1.0),
            min(linkedin_connections / 500, 1.0)
        ])
        
        return features
    
    def _extract_engagement_features(self, lead_data: Dict) -> List[float]:
        """Extract engagement-specific features"""
        features = []
        
        # Communication frequency
        total_interactions = lead_data.get('total_interactions', 0)
        days_since_first_contact = max(lead_data.get('days_since_first_contact', 1), 1)
        interaction_frequency = total_interactions / days_since_first_contact
        
        features.append(min(interaction_frequency, 1.0))
        
        # Response rate
        responses = lead_data.get('responses', 0)
        outreach_attempts = max(lead_data.get('outreach_attempts', 1), 1)
        response_rate = responses / outreach_attempts
        
        features.append(response_rate)
        
        # Meeting acceptance rate
        meetings_scheduled = lead_data.get('meetings_scheduled', 0)
        meeting_requests = max(lead_data.get('meeting_requests', 1), 1)
        meeting_acceptance_rate = meetings_scheduled / meeting_requests
        
        features.append(meeting_acceptance_rate)
        
        # Recency of engagement
        days_since_last_contact = lead_data.get('days_since_last_contact', 0)
        recency_score = max(0, 1 - (days_since_last_contact / 30))  # Decay over 30 days
        
        features.append(recency_score)
        
        return features
    
    def _extract_company_features(self, lead_data: Dict) -> List[float]:
        """Extract company-specific features"""
        features = []
        
        # Revenue indicators
        annual_revenue = lead_data.get('annual_revenue', 0)
        if annual_revenue > 100000000:  # $100M+
            revenue_score = 1.0
        elif annual_revenue > 10000000:  # $10M+
            revenue_score = 0.8
        elif annual_revenue > 1000000:  # $1M+
            revenue_score = 0.6
        else:
            revenue_score = 0.4
        
        features.append(revenue_score)
        
        # Funding status
        funding_stage = lead_data.get('funding_stage', '').lower()
        funding_scores = {
            'series_c': 1.0, 'series_b': 0.8, 'series_a': 0.6,
            'seed': 0.4, 'bootstrap': 0.3, 'unknown': 0.2
        }
        funding_score = funding_scores.get(funding_stage, 0.2)
        features.append(funding_score)
        
        # Technology stack compatibility
        tech_stack = lead_data.get('technology_stack', [])
        compatible_techs = ['aws', 'azure', 'kubernetes', 'react', 'python', 'java']
        compatibility_score = len(set(tech_stack) & set(compatible_techs)) / len(compatible_techs)
        features.append(compatibility_score)
        
        # Growth indicators
        employee_growth = lead_data.get('employee_growth_rate', 0)  # percentage
        features.append(min(employee_growth / 50, 1.0))  # Cap at 50% growth
        
        return features
    
    def _extract_interaction_features(self, lead_data: Dict) -> List[float]:
        """Extract interaction quality features"""
        features = []
        
        # Call quality metrics
        avg_call_duration = lead_data.get('avg_call_duration', 0)  # minutes
        call_completion_rate = lead_data.get('call_completion_rate', 0)
        positive_call_sentiment = lead_data.get('positive_call_sentiment', 0)
        
        features.extend([
            min(avg_call_duration / 30, 1.0),  # Normalize to 30 min max
            call_completion_rate,
            positive_call_sentiment
        ])
        
        # Email interaction quality
        email_reply_time = lead_data.get('avg_email_reply_time', 24)  # hours
        reply_speed_score = max(0, 1 - (email_reply_time / 48))  # 48 hour max
        
        features.append(reply_speed_score)
        
        # Question quality and specificity
        technical_questions = lead_data.get('technical_questions_asked', 0)
        pricing_inquiries = lead_data.get('pricing_inquiries', 0)
        implementation_questions = lead_data.get('implementation_questions', 0)
        
        features.extend([
            min(technical_questions / 5, 1.0),
            min(pricing_inquiries / 3, 1.0),
            min(implementation_questions / 3, 1.0)
        ])
        
        return features
    
    def _extract_temporal_features(self, lead_data: Dict) -> List[float]:
        """Extract time-based features"""
        features = []
        
        # Lead age
        created_date = lead_data.get('created_date')
        if created_date:
            if isinstance(created_date, str):
                created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            lead_age_days = (datetime.now() - created_date).days
            age_score = max(0, 1 - (lead_age_days / 90))  # Decay over 90 days
        else:
            age_score = 0.5
        
        features.append(age_score)
        
        # Sales cycle position
        days_in_pipeline = lead_data.get('days_in_pipeline', 0)
        typical_cycle_length = 45  # days
        cycle_position = min(days_in_pipeline / typical_cycle_length, 1.0)
        
        features.append(cycle_position)
        
        # Seasonal factors
        current_month = datetime.now().month
        if current_month in [3, 6, 9, 12]:  # Quarter end months
            seasonal_score = 1.2
        elif current_month in [1, 7]:  # New year, summer
            seasonal_score = 0.8
        else:
            seasonal_score = 1.0
        
        features.append(min(seasonal_score, 1.0))
        
        # Day of week factor
        last_contact_day = lead_data.get('last_contact_day_of_week', 2)  # Tuesday default
        if last_contact_day in [1, 2, 3]:  # Mon-Wed are better
            day_score = 1.0
        elif last_contact_day in [0, 4]:  # Thu-Fri
            day_score = 0.8
        else:  # Weekend
            day_score = 0.5
        
        features.append(day_score)
        
        return features
    
    async def _extract_text_features(self, lead_data: Dict) -> List[float]:
        """Extract features from text data"""
        features = []
        
        # Combine all text fields
        text_fields = [
            lead_data.get('notes', ''),
            lead_data.get('email_content', ''),
            lead_data.get('call_transcripts', ''),
            lead_data.get('chat_messages', '')
        ]
        
        combined_text = ' '.join(filter(None, text_fields))
        
        if combined_text:
            # Sentiment features
            from textblob import TextBlob
            blob = TextBlob(combined_text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
            
            features.extend([
                (sentiment_polarity + 1) / 2,  # Normalize to 0-1
                sentiment_subjectivity
            ])
            
            # Keyword presence
            buying_keywords = [
                'budget', 'price', 'cost', 'investment', 'roi',
                'timeline', 'implementation', 'decision', 'approval',
                'contract', 'agreement', 'purchase', 'buy'
            ]
            
            keyword_count = sum(1 for keyword in buying_keywords 
                              if keyword in combined_text.lower())
            keyword_density = keyword_count / len(buying_keywords)
            
            features.append(keyword_density)
            
            # Text length and complexity
            word_count = len(combined_text.split())
            avg_word_length = np.mean([len(word) for word in combined_text.split()])
            
            features.extend([
                min(word_count / 1000, 1.0),  # Normalize word count
                min(avg_word_length / 10, 1.0)  # Normalize avg word length
            ])
        else:
            # Default values when no text available
            features.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        return features
    
    async def train_models(self, training_data: List[Dict[str, Any]], 
                          target_variable: str = 'converted') -> Dict[str, MLModelMetrics]:
        """
        Train all models in the ensemble
        """
        logger.info("Starting ML model training...")
        
        # Prepare training data
        X = []
        y = []
        
        for lead_data in training_data:
            features = await self.extract_features(lead_data)
            X.append(features)
            y.append(lead_data.get(target_variable, 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # Store feature columns for later use
        self.feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scalers['numerical'].fit_transform(X_train)
        X_test_scaled = self.scalers['numerical'].transform(X_test)
        
        # Train models and collect metrics
        model_metrics = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue  # Train ensemble last
                
            logger.info(f"Training {model_name}...")
            
            try:
                # Use scaled data for neural network, original for tree-based
                if model_name == 'neural_network':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = MLModelMetrics(
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred, average='weighted'),
                    recall=recall_score(y_test, y_pred, average='weighted'),
                    f1_score=f1_score(y_test, y_pred, average='weighted'),
                    roc_auc=roc_auc_score(y_test, y_pred_proba),
                    cross_val_mean=0.0,  # Will calculate below
                    cross_val_std=0.0
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                metrics.cross_val_mean = cv_scores.mean()
                metrics.cross_val_std = cv_scores.std()
                
                model_metrics[model_name] = metrics
                
                logger.info(f"{model_name} - Accuracy: {metrics.accuracy:.3f}, "
                          f"ROC-AUC: {metrics.roc_auc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        try:
            self.models['ensemble'].fit(X_train, y_train)
            y_pred_ensemble = self.models['ensemble'].predict(X_test)
            y_pred_proba_ensemble = self.models['ensemble'].predict_proba(X_test)[:, 1]
            
            ensemble_metrics = MLModelMetrics(
                accuracy=accuracy_score(y_test, y_pred_ensemble),
                precision=precision_score(y_test, y_pred_ensemble, average='weighted'),
                recall=recall_score(y_test, y_pred_ensemble, average='weighted'),
                f1_score=f1_score(y_test, y_pred_ensemble, average='weighted'),
                roc_auc=roc_auc_score(y_test, y_pred_proba_ensemble),
                cross_val_mean=0.0,
                cross_val_std=0.0
            )
            
            model_metrics['ensemble'] = ensemble_metrics
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
        
        # Save models
        await self._save_models()
        
        # Store metrics
        self.model_metrics = model_metrics
        
        # Calculate feature importance
        self._calculate_feature_importance(X, y)
        
        logger.info("Model training completed successfully")
        return model_metrics
    
    async def predict_lead_score(self, lead_data: Dict[str, Any]) -> LeadScoringPrediction:
        """
        Generate comprehensive lead scoring prediction
        """
        try:
            # Extract features
            features = await self.extract_features(lead_data)
            features = features.reshape(1, -1)
            
            # Make predictions with all models
            model_votes = {}
            probabilities = []
            
            for model_name, model in self.models.items():
                if model_name == 'neural_network':
                    # Use scaled features for neural network
                    scaled_features = self.scalers['numerical'].transform(features)
                    prob = model.predict_proba(scaled_features)[0][1]
                else:
                    prob = model.predict_proba(features)[0][1]
                
                model_votes[model_name] = float(prob)
                probabilities.append(prob)
            
            # Ensemble prediction
            conversion_probability = float(np.mean(probabilities))
            
            # Calculate confidence interval
            prob_std = np.std(probabilities)
            confidence_interval = (
                max(0, conversion_probability - 1.96 * prob_std),
                min(1, conversion_probability + 1.96 * prob_std)
            )
            
            # Overall score (0-100)
            overall_score = int(conversion_probability * 100)
            
            # Feature contributions (simplified)
            feature_contributions = self._calculate_feature_contributions(
                features[0], conversion_probability
            )
            
            # Predict deal value
            predicted_deal_value = self._predict_deal_value(features[0], lead_data)
            
            # Predict timeline
            predicted_timeline = self._predict_conversion_timeline(features[0], lead_data)
            
            # Risk assessment
            risk_assessment = self._assess_risks(features[0], lead_data, conversion_probability)
            
            # Recommended actions
            recommended_actions = self._generate_recommendations(
                conversion_probability, risk_assessment, lead_data
            )
            
            # Segment classification
            segment = self._classify_segment(conversion_probability, features[0])
            
            # Priority tier
            priority_tier = self._determine_priority_tier(
                conversion_probability, predicted_deal_value, predicted_timeline
            )
            
            return LeadScoringPrediction(
                lead_id=lead_data.get('id', 'unknown'),
                overall_score=overall_score,
                conversion_probability=conversion_probability,
                confidence_interval=confidence_interval,
                feature_contributions=feature_contributions,
                model_ensemble_votes=model_votes,
                predicted_deal_value=predicted_deal_value,
                predicted_conversion_timeline=predicted_timeline,
                risk_assessment=risk_assessment,
                recommended_actions=recommended_actions,
                segment=segment,
                priority_tier=priority_tier
            )
            
        except Exception as e:
            logger.error(f"Error in lead scoring prediction: {e}")
            return self._get_fallback_prediction(lead_data)
    
    def _calculate_feature_contributions(self, features: np.ndarray, 
                                       prediction: float) -> Dict[str, float]:
        """Calculate feature contributions to the prediction"""
        # Simplified SHAP-like attribution
        feature_names = [
            'seniority', 'industry', 'company_size', 'location',
            'email_engagement', 'web_engagement', 'downloads',
            'response_rate', 'meeting_rate', 'recency',
            'revenue', 'funding', 'tech_compatibility', 'growth'
        ]
        
        contributions = {}
        base_prediction = 0.3  # baseline
        
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features[:len(feature_names)])):
            # Simple linear contribution estimation
            contribution = (feature_value - 0.5) * prediction * 0.1
            contributions[feature_name] = float(contribution)
        
        return contributions
    
    def _predict_deal_value(self, features: np.ndarray, lead_data: Dict) -> float:
        """Predict potential deal value"""
        # Base value calculation
        company_size = lead_data.get('company_size', 50)
        industry_multiplier = {
            'technology': 1.5, 'finance': 1.8, 'healthcare': 1.4,
            'manufacturing': 1.2, 'retail': 0.9
        }
        
        industry = lead_data.get('industry', '').lower()
        multiplier = 1.0
        for key, mult in industry_multiplier.items():
            if key in industry:
                multiplier = mult
                break
        
        # Base deal value calculation
        if company_size > 1000:
            base_value = 50000
        elif company_size > 100:
            base_value = 25000
        else:
            base_value = 10000
        
        # Apply multipliers based on features
        seniority_boost = features[0] * 1.5  # Higher seniority = larger deals
        engagement_boost = np.mean(features[4:7]) * 1.3  # High engagement
        
        predicted_value = base_value * multiplier * (1 + seniority_boost + engagement_boost)
        
        return float(predicted_value)
    
    def _predict_conversion_timeline(self, features: np.ndarray, lead_data: Dict) -> int:
        """Predict days to conversion"""
        base_timeline = 45  # days
        
        # Factors that speed up conversion
        urgency_factor = features[10] if len(features) > 10 else 0.5  # Assume position 10 is urgency
        engagement_factor = np.mean(features[4:7]) if len(features) > 7 else 0.5
        seniority_factor = features[0] if len(features) > 0 else 0.5
        
        # Speed multiplier (higher = faster)
        speed_multiplier = (urgency_factor * 0.4 + engagement_factor * 0.3 + seniority_factor * 0.3)
        
        # Calculate timeline
        timeline_days = int(base_timeline * (1 - speed_multiplier * 0.5))
        
        return max(7, min(timeline_days, 180))  # Between 1 week and 6 months
    
    def _assess_risks(self, features: np.ndarray, lead_data: Dict, 
                     conversion_prob: float) -> Dict[str, float]:
        """Assess various risk factors"""
        risks = {}
        
        # Budget risk
        if lead_data.get('budget_indicated', False):
            risks['budget_risk'] = 0.2
        else:
            risks['budget_risk'] = 0.7
        
        # Timeline risk
        days_in_pipeline = lead_data.get('days_in_pipeline', 0)
        if days_in_pipeline > 90:
            risks['timeline_risk'] = 0.8
        elif days_in_pipeline > 60:
            risks['timeline_risk'] = 0.5
        else:
            risks['timeline_risk'] = 0.2
        
        # Competition risk
        mentions_competitors = lead_data.get('mentions_competitors', False)
        risks['competition_risk'] = 0.7 if mentions_competitors else 0.3
        
        # Authority risk
        seniority = features[0] if len(features) > 0 else 0.5
        risks['authority_risk'] = 1.0 - seniority
        
        # Engagement risk
        engagement = np.mean(features[4:7]) if len(features) > 7 else 0.5
        risks['engagement_risk'] = 1.0 - engagement
        
        return risks
    
    def _generate_recommendations(self, conversion_prob: float, 
                                risks: Dict[str, float], 
                                lead_data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if conversion_prob > 0.8:
            recommendations.append("High conversion probability - prioritize immediate outreach")
            recommendations.append("Prepare proposal and pricing information")
        elif conversion_prob > 0.6:
            recommendations.append("Good conversion potential - schedule demo or deeper discovery")
            recommendations.append("Identify decision-making process and timeline")
        elif conversion_prob > 0.4:
            recommendations.append("Moderate potential - focus on nurturing and education")
            recommendations.append("Share relevant case studies and ROI information")
        else:
            recommendations.append("Lower conversion probability - long-term nurturing approach")
            recommendations.append("Focus on building relationship and trust")
        
        # Risk-specific recommendations
        if risks.get('budget_risk', 0) > 0.6:
            recommendations.append("Address budget concerns with ROI analysis")
        
        if risks.get('authority_risk', 0) > 0.6:
            recommendations.append("Identify and engage decision makers")
        
        if risks.get('competition_risk', 0) > 0.6:
            recommendations.append("Differentiate from competitors - highlight unique value")
        
        if risks.get('timeline_risk', 0) > 0.6:
            recommendations.append("Create urgency and accelerate decision process")
        
        return recommendations
    
    def _classify_segment(self, conversion_prob: float, features: np.ndarray) -> str:
        """Classify lead into segments"""
        if conversion_prob > 0.8:
            return "hot"
        elif conversion_prob > 0.6:
            return "warm"
        elif conversion_prob > 0.4:
            return "lukewarm"
        else:
            return "cold"
    
    def _determine_priority_tier(self, conversion_prob: float, 
                               deal_value: float, timeline: int) -> str:
        """Determine priority tier for sales team"""
        # Calculate priority score
        prob_score = conversion_prob * 40
        value_score = min(deal_value / 50000, 1.0) * 30
        timeline_score = max(0, (180 - timeline) / 180) * 30
        
        priority_score = prob_score + value_score + timeline_score
        
        if priority_score > 70:
            return "P1 - Critical"
        elif priority_score > 50:
            return "P2 - High"
        elif priority_score > 30:
            return "P3 - Medium"
        else:
            return "P4 - Low"
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate and store feature importance"""
        # Use Random Forest for feature importance
        rf_model = self.models.get('random_forest')
        if rf_model and hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            
            feature_importance_list = []
            for i, importance in enumerate(importances):
                feature_name = f"feature_{i}" if i < len(self.feature_columns) else f"feature_{i}"
                feature_importance_list.append(
                    FeatureImportance(
                        feature_name=feature_name,
                        importance_score=float(importance),
                        rank=i + 1,
                        category="behavioral" if i < 10 else "demographic"
                    )
                )
            
            # Sort by importance
            feature_importance_list.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Update ranks
            for rank, fi in enumerate(feature_importance_list):
                fi.rank = rank + 1
            
            self.feature_importance = feature_importance_list
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                model_file = self.model_path / f"{model_name}_model.pkl"
                joblib.dump(model, model_file)
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_file = self.model_path / f"{scaler_name}_scaler.pkl"
                joblib.dump(scaler, scaler_file)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_file = self.model_path / f"{model_name}_model.pkl"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
            
            # Load scalers
            for scaler_name in self.scalers.keys():
                scaler_file = self.model_path / f"{scaler_name}_scaler.pkl"
                if scaler_file.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_file)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _get_fallback_prediction(self, lead_data: Dict) -> LeadScoringPrediction:
        """Fallback prediction when models fail"""
        return LeadScoringPrediction(
            lead_id=lead_data.get('id', 'unknown'),
            overall_score=50,
            conversion_probability=0.3,
            confidence_interval=(0.2, 0.4),
            feature_contributions={},
            model_ensemble_votes={},
            predicted_deal_value=15000.0,
            predicted_conversion_timeline=45,
            risk_assessment={'unknown_risk': 0.5},
            recommended_actions=['Gather more lead data for accurate scoring'],
            segment='lukewarm',
            priority_tier='P3 - Medium'
        )

# Global instance
ml_lead_scoring = MLLeadScoringSystem()