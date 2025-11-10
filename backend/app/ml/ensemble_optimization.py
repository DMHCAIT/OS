"""
Ensemble Model Auto-Optimization System
Automated hyperparameter tuning and model selection for maximum accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import json
import logging
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_type: str
    model_class: Any
    param_grid: Dict[str, Any]
    search_type: str  # 'grid', 'random', 'bayesian'
    cv_folds: int = 5
    scoring_metric: str = 'f1_weighted'
    max_trials: int = 100
    timeout_minutes: int = 30

@dataclass
class OptimizationResult:
    """Results of model optimization"""
    model_type: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    training_time_seconds: float
    model_instance: Any
    feature_importance: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None

@dataclass
class EnsembleResult:
    """Results of ensemble optimization"""
    ensemble_type: str
    member_models: List[OptimizationResult]
    ensemble_score: float
    individual_scores: Dict[str, float]
    voting_weights: Optional[List[float]] = None
    optimization_timestamp: datetime = None
    cross_validation_scores: List[float] = None
    test_metrics: Optional[Dict[str, float]] = None

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None

class EnsembleModelOptimizer:
    """Advanced ensemble model optimization system"""
    
    def __init__(
        self,
        models_dir: str = "models/optimized",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        
        # Initialize model configurations
        self.model_configs = self._initialize_model_configs()
        
        # Performance tracking
        self.optimization_history: List[EnsembleResult] = []
        self.best_ensemble: Optional[EnsembleResult] = None
        
        logger.info("EnsembleModelOptimizer initialized")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations for optimization"""
        configs = {}
        
        # Random Forest
        configs['random_forest'] = ModelConfig(
            model_type='random_forest',
            model_class=RandomForestClassifier,
            param_grid={
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            search_type='bayesian',
            max_trials=150
        )
        
        # Gradient Boosting
        configs['gradient_boosting'] = ModelConfig(
            model_type='gradient_boosting',
            model_class=GradientBoostingClassifier,
            param_grid={
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            },
            search_type='bayesian',
            max_trials=100
        )
        
        # Logistic Regression
        configs['logistic_regression'] = ModelConfig(
            model_type='logistic_regression',
            model_class=LogisticRegression,
            param_grid={
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 3000],
                'class_weight': ['balanced', None]
            },
            search_type='grid',
            max_trials=50
        )
        
        # SVM
        configs['svm'] = ModelConfig(
            model_type='svm',
            model_class=SVC,
            param_grid={
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': [2, 3, 4],  # Only for poly kernel
                'class_weight': ['balanced', None],
                'probability': [True]  # For probability estimates
            },
            search_type='random',
            max_trials=80
        )
        
        # Neural Network
        configs['neural_network'] = ModelConfig(
            model_type='neural_network',
            model_class=MLPClassifier,
            param_grid={
                'hidden_layer_sizes': [
                    (50,), (100,), (150,),
                    (50, 50), (100, 50), (150, 100),
                    (100, 100, 50), (150, 100, 50)
                ],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000, 2000],
                'early_stopping': [True],
                'validation_fraction': [0.1, 0.2]
            },
            search_type='bayesian',
            max_trials=100
        )
        
        return configs
    
    async def optimize_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        ensemble_methods: List[str] = ['voting_hard', 'voting_soft', 'stacking'],
        models_to_include: Optional[List[str]] = None
    ) -> EnsembleResult:
        """Optimize ensemble of models with hyperparameter tuning"""
        try:
            logger.info("Starting ensemble optimization")
            
            # Prepare data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = None
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
            
            # Select models to optimize
            if models_to_include is None:
                models_to_include = list(self.model_configs.keys())
            
            # Optimize individual models in parallel
            optimization_tasks = []
            for model_name in models_to_include:
                if model_name in self.model_configs:
                    task = self._optimize_single_model(
                        model_name,
                        X_train_scaled,
                        y_train,
                        X_test_scaled,
                        y_test
                    )
                    optimization_tasks.append(task)
            
            # Run optimizations concurrently
            optimized_models = await asyncio.gather(*optimization_tasks)
            
            # Filter successful optimizations
            valid_models = [model for model in optimized_models if model is not None]
            
            if len(valid_models) < 2:
                raise ValueError("Need at least 2 models for ensemble")
            
            logger.info(f"Successfully optimized {len(valid_models)} models")
            
            # Create and evaluate ensemble combinations
            best_ensemble = None
            best_score = -np.inf
            
            for ensemble_method in ensemble_methods:
                logger.info(f"Evaluating ensemble method: {ensemble_method}")
                
                ensemble_result = await self._create_ensemble(
                    valid_models,
                    ensemble_method,
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test
                )
                
                if ensemble_result and ensemble_result.ensemble_score > best_score:
                    best_score = ensemble_result.ensemble_score
                    best_ensemble = ensemble_result
            
            if best_ensemble:
                self.best_ensemble = best_ensemble
                self.optimization_history.append(best_ensemble)
                
                # Save ensemble
                await self._save_ensemble(best_ensemble)
                
                logger.info(
                    f"Best ensemble: {best_ensemble.ensemble_type} "
                    f"with score: {best_ensemble.ensemble_score:.4f}"
                )
            
            return best_ensemble
            
        except Exception as e:
            logger.error(f"Error in ensemble optimization: {e}")
            raise
    
    async def _optimize_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> Optional[OptimizationResult]:
        """Optimize a single model using specified search strategy"""
        try:
            config = self.model_configs[model_name]
            start_time = datetime.now()
            
            logger.info(f"Optimizing {model_name} using {config.search_type} search")
            
            if config.search_type == 'bayesian':
                result = await self._bayesian_optimization(
                    config, X_train, y_train, X_test, y_test
                )
            elif config.search_type == 'random':
                result = await self._random_search_optimization(
                    config, X_train, y_train, X_test, y_test
                )
            elif config.search_type == 'grid':
                result = await self._grid_search_optimization(
                    config, X_train, y_train, X_test, y_test
                )
            else:
                raise ValueError(f"Unknown search type: {config.search_type}")
            
            if result:
                result.training_time_seconds = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Completed {model_name} optimization in "
                    f"{result.training_time_seconds:.2f} seconds"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            return None
    
    async def _bayesian_optimization(
        self,
        config: ModelConfig,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> OptimizationResult:
        """Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in config.param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
            
            # Handle special cases for specific models
            if config.model_type == 'logistic_regression':
                if params.get('penalty') == 'elasticnet':
                    if 'l1_ratio' not in params:
                        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
                else:
                    params.pop('l1_ratio', None)
            
            # Create and evaluate model
            model = config.model_class(random_state=self.random_state, **params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, scoring=config.scoring_metric, n_jobs=1
            )
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, 
            n_trials=config.max_trials,
            timeout=config.timeout_minutes * 60
        )
        
        # Get best model
        best_params = study.best_params
        best_model = config.model_class(random_state=self.random_state, **best_params)
        best_model.fit(X_train, y_train)
        
        # Calculate CV scores
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv, scoring=config.scoring_metric, n_jobs=self.n_jobs
        )
        
        # Calculate validation metrics if test set provided
        validation_metrics = None
        if X_test is not None and y_test is not None:
            validation_metrics = self._calculate_metrics(best_model, X_test, y_test)
        
        # Extract feature importance if available
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = {
                f'feature_{i}': importance 
                for i, importance in enumerate(best_model.feature_importances_)
            }
        elif hasattr(best_model, 'coef_'):
            feature_importance = {
                f'feature_{i}': abs(coef) 
                for i, coef in enumerate(best_model.coef_[0])
            }
        
        return OptimizationResult(
            model_type=config.model_type,
            best_params=best_params,
            best_score=study.best_value,
            cv_scores=cv_scores.tolist(),
            training_time_seconds=0.0,  # Will be set by caller
            model_instance=best_model,
            feature_importance=feature_importance,
            validation_metrics=validation_metrics
        )
    
    async def _random_search_optimization(
        self,
        config: ModelConfig,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> OptimizationResult:
        """Random search optimization"""
        
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            estimator=config.model_class(random_state=self.random_state),
            param_distributions=config.param_grid,
            n_iter=config.max_trials,
            cv=cv,
            scoring=config.scoring_metric,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        search.fit(X_train, y_train)
        
        # Calculate validation metrics if test set provided
        validation_metrics = None
        if X_test is not None and y_test is not None:
            validation_metrics = self._calculate_metrics(search.best_estimator_, X_test, y_test)
        
        # Extract feature importance if available
        feature_importance = None
        best_model = search.best_estimator_
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = {
                f'feature_{i}': importance 
                for i, importance in enumerate(best_model.feature_importances_)
            }
        elif hasattr(best_model, 'coef_'):
            feature_importance = {
                f'feature_{i}': abs(coef) 
                for i, coef in enumerate(best_model.coef_[0])
            }
        
        return OptimizationResult(
            model_type=config.model_type,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_scores=search.cv_results_['mean_test_score'].tolist(),
            training_time_seconds=0.0,
            model_instance=search.best_estimator_,
            feature_importance=feature_importance,
            validation_metrics=validation_metrics
        )
    
    async def _grid_search_optimization(
        self,
        config: ModelConfig,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> OptimizationResult:
        """Grid search optimization"""
        
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = GridSearchCV(
            estimator=config.model_class(random_state=self.random_state),
            param_grid=config.param_grid,
            cv=cv,
            scoring=config.scoring_metric,
            n_jobs=self.n_jobs
        )
        
        search.fit(X_train, y_train)
        
        # Calculate validation metrics if test set provided
        validation_metrics = None
        if X_test is not None and y_test is not None:
            validation_metrics = self._calculate_metrics(search.best_estimator_, X_test, y_test)
        
        # Extract feature importance if available
        feature_importance = None
        best_model = search.best_estimator_
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = {
                f'feature_{i}': importance 
                for i, importance in enumerate(best_model.feature_importances_)
            }
        elif hasattr(best_model, 'coef_'):
            feature_importance = {
                f'feature_{i}': abs(coef) 
                for i, coef in enumerate(best_model.coef_[0])
            }
        
        return OptimizationResult(
            model_type=config.model_type,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_scores=search.cv_results_['mean_test_score'].tolist(),
            training_time_seconds=0.0,
            model_instance=search.best_estimator_,
            feature_importance=feature_importance,
            validation_metrics=validation_metrics
        )
    
    async def _create_ensemble(
        self,
        models: List[OptimizationResult],
        ensemble_method: str,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> Optional[EnsembleResult]:
        """Create and evaluate ensemble"""
        try:
            if ensemble_method == 'voting_hard':
                return await self._create_voting_ensemble(
                    models, 'hard', X_train, y_train, X_test, y_test
                )
            elif ensemble_method == 'voting_soft':
                return await self._create_voting_ensemble(
                    models, 'soft', X_train, y_train, X_test, y_test
                )
            elif ensemble_method == 'stacking':
                return await self._create_stacking_ensemble(
                    models, X_train, y_train, X_test, y_test
                )
            else:
                logger.warning(f"Unknown ensemble method: {ensemble_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating {ensemble_method} ensemble: {e}")
            return None
    
    async def _create_voting_ensemble(
        self,
        models: List[OptimizationResult],
        voting_type: str,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> EnsembleResult:
        """Create voting ensemble"""
        
        # Prepare estimators
        estimators = [(model.model_type, model.model_instance) for model in models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            n_jobs=self.n_jobs
        )
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            ensemble, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=self.n_jobs
        )
        ensemble_score = cv_scores.mean()
        
        # Individual model scores
        individual_scores = {model.model_type: model.best_score for model in models}
        
        # Test metrics if available
        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(ensemble, X_test, y_test)
        
        return EnsembleResult(
            ensemble_type=f'voting_{voting_type}',
            member_models=models,
            ensemble_score=ensemble_score,
            individual_scores=individual_scores,
            optimization_timestamp=datetime.now(),
            cross_validation_scores=cv_scores.tolist(),
            test_metrics=test_metrics
        )
    
    async def _create_stacking_ensemble(
        self,
        models: List[OptimizationResult],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[pd.Series] = None
    ) -> EnsembleResult:
        """Create stacking ensemble"""
        from sklearn.ensemble import StackingClassifier
        
        # Prepare estimators
        estimators = [(model.model_type, model.model_instance) for model in models]
        
        # Create stacking classifier with logistic regression as meta-learner
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            n_jobs=self.n_jobs
        )
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            ensemble, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=self.n_jobs
        )
        ensemble_score = cv_scores.mean()
        
        # Individual model scores
        individual_scores = {model.model_type: model.best_score for model in models}
        
        # Test metrics if available
        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(ensemble, X_test, y_test)
        
        return EnsembleResult(
            ensemble_type='stacking',
            member_models=models,
            ensemble_score=ensemble_score,
            individual_scores=individual_scores,
            optimization_timestamp=datetime.now(),
            cross_validation_scores=cv_scores.tolist(),
            test_metrics=test_metrics
        )
    
    def _calculate_metrics(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        y_pred = model.predict(X)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2 and y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        return metrics
    
    async def _save_ensemble(self, ensemble_result: EnsembleResult):
        """Save ensemble to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save ensemble metadata
            metadata_path = self.models_dir / f"ensemble_{timestamp}_metadata.json"
            metadata = asdict(ensemble_result)
            
            # Convert datetime to string
            metadata['optimization_timestamp'] = ensemble_result.optimization_timestamp.isoformat()
            
            # Remove model instances from metadata
            for model_data in metadata['member_models']:
                model_data.pop('model_instance', None)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save individual models
            for i, model in enumerate(ensemble_result.member_models):
                model_path = self.models_dir / f"model_{timestamp}_{i}_{model.model_type}.joblib"
                joblib.dump(model.model_instance, model_path)
            
            # Save scaler
            scaler_path = self.models_dir / f"scaler_{timestamp}.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Saved ensemble to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
    
    async def load_best_ensemble(self, timestamp: Optional[str] = None) -> Optional[EnsembleResult]:
        """Load best ensemble from disk"""
        try:
            if timestamp:
                metadata_file = f"ensemble_{timestamp}_metadata.json"
            else:
                # Find latest ensemble
                metadata_files = list(self.models_dir.glob("ensemble_*_metadata.json"))
                if not metadata_files:
                    logger.warning("No ensemble metadata files found")
                    return None
                metadata_file = max(metadata_files).name
            
            metadata_path = self.models_dir / metadata_file
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract timestamp
            file_timestamp = metadata_file.split('_')[1] + '_' + metadata_file.split('_')[2]
            
            # Load individual models
            for i, model_data in enumerate(metadata['member_models']):
                model_type = model_data['model_type']
                model_path = self.models_dir / f"model_{file_timestamp}_{i}_{model_type}.joblib"
                model_instance = joblib.load(model_path)
                model_data['model_instance'] = model_instance
            
            # Load scaler
            scaler_path = self.models_dir / f"scaler_{file_timestamp}.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Reconstruct ensemble result
            metadata['optimization_timestamp'] = datetime.fromisoformat(metadata['optimization_timestamp'])
            
            ensemble_result = EnsembleResult(**metadata)
            
            logger.info(f"Loaded ensemble from {metadata_path}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return None
    
    async def predict_with_ensemble(
        self,
        X: pd.DataFrame,
        ensemble_result: Optional[EnsembleResult] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using ensemble"""
        try:
            if ensemble_result is None:
                ensemble_result = self.best_ensemble
            
            if ensemble_result is None:
                raise ValueError("No ensemble available for prediction")
            
            # Scale input data
            X_scaled = self.scaler.transform(X)
            
            # Make predictions with all models
            model_predictions = []
            model_probabilities = []
            
            for model_result in ensemble_result.member_models:
                model = model_result.model_instance
                pred = model.predict(X_scaled)
                model_predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                    model_probabilities.append(proba)
            
            # Ensemble prediction (simple voting)
            if ensemble_result.ensemble_type.startswith('voting'):
                if ensemble_result.ensemble_type == 'voting_hard':
                    # Hard voting
                    ensemble_pred = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(),
                        axis=0,
                        arr=np.array(model_predictions)
                    )
                    ensemble_proba = None
                    if model_probabilities:
                        ensemble_proba = np.mean(model_probabilities, axis=0)
                else:
                    # Soft voting
                    ensemble_proba = np.mean(model_probabilities, axis=0)
                    ensemble_pred = np.argmax(ensemble_proba, axis=1)
            else:
                # For stacking, would need to recreate the ensemble
                # For now, use simple averaging
                if model_probabilities:
                    ensemble_proba = np.mean(model_probabilities, axis=0)
                    ensemble_pred = np.argmax(ensemble_proba, axis=1)
                else:
                    ensemble_pred = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(),
                        axis=0,
                        arr=np.array(model_predictions)
                    )
                    ensemble_proba = None
            
            return ensemble_pred, ensemble_proba
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        best_ensemble = max(self.optimization_history, key=lambda x: x.ensemble_score)
        
        summary = {
            "total_optimizations": len(self.optimization_history),
            "best_ensemble": {
                "type": best_ensemble.ensemble_type,
                "score": best_ensemble.ensemble_score,
                "timestamp": best_ensemble.optimization_timestamp.isoformat(),
                "member_models": [model.model_type for model in best_ensemble.member_models],
                "individual_scores": best_ensemble.individual_scores
            },
            "recent_optimizations": [
                {
                    "type": ensemble.ensemble_type,
                    "score": ensemble.ensemble_score,
                    "timestamp": ensemble.optimization_timestamp.isoformat()
                }
                for ensemble in self.optimization_history[-5:]  # Last 5
            ]
        }
        
        return summary


# Global instance
ensemble_optimizer = EnsembleModelOptimizer()