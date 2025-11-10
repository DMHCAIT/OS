"""
ML Training Routes for Predictive Business Intelligence
Advanced endpoints for training and managing machine learning models
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import pandas as pd
import numpy as np
from app.ml.continuous_learning import continuous_learning_engine, TrainingMetrics
from app.core.auth import get_current_user
from app.core.database import database_manager

router = APIRouter()
logger = logging.getLogger(__name__)

class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to train")
    data: List[Dict[str, Any]] = Field(..., description="Training data")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Training configuration")

class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")

class LearningConfigRequest(BaseModel):
    retrain_threshold: int = Field(1000, description="Number of data points to trigger retraining")
    performance_threshold: float = Field(0.1, description="Performance drop threshold")
    schedule: str = Field("weekly", description="Retraining schedule")
    auto_feature_engineering: bool = Field(True, description="Enable automatic feature engineering")

class DataFeedRequest(BaseModel):
    data_type: str = Field(..., description="Type of data being fed")
    data: Dict[str, Any] = Field(..., description="Data to feed to the learning system")
    timestamp: Optional[str] = Field(default=None, description="Data timestamp")

class EnsembleRequest(BaseModel):
    ensemble_name: str = Field(..., description="Name of the ensemble")
    models: List[Dict[str, Any]] = Field(..., description="Models and their weights")
    voting_strategy: str = Field("soft", description="Voting strategy: soft or hard")

class ABTestRequest(BaseModel):
    test_name: str = Field(..., description="Name of the A/B test")
    control_model: str = Field(..., description="Control model name")
    test_model: str = Field(..., description="Test model name")
    traffic_split: float = Field(0.5, description="Traffic split ratio")
    success_metric: str = Field(..., description="Success metric to measure")
    duration_days: int = Field(14, description="Test duration in days")

@router.post("/train", response_model=Dict[str, Any])
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Train or retrain a machine learning model
    """
    try:
        logger.info(f"Training request for model: {request.model_name}")
        
        if not request.data:
            raise HTTPException(status_code=400, detail="Training data is required")
        
        # Validate model name
        if request.model_name not in continuous_learning_engine.model_registry:
            available_models = list(continuous_learning_engine.model_registry.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown model: {request.model_name}. Available: {available_models}"
            )
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            request.model_name,
            request.data,
            current_user["user_id"]
        )
        
        return {
            "status": "training_started",
            "model_name": request.model_name,
            "data_points": len(request.data),
            "message": "Model training started in background",
            "estimated_completion": "5-15 minutes"
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def _train_model_background(model_name: str, training_data: List[Dict[str, Any]], user_id: str):
    """Background task for model training"""
    try:
        metrics = await continuous_learning_engine.train_model(model_name, training_data)
        
        # Save training record to database
        training_record = {
            "model_name": model_name,
            "user_id": user_id,
            "data_points": len(training_data),
            "accuracy": metrics.accuracy,
            "training_time": metrics.training_time,
            "timestamp": datetime.now(),
            "status": "completed"
        }
        
        await database_manager.insert_one("training_history", training_record)
        logger.info(f"Training completed for {model_name}: {metrics.accuracy:.3f} accuracy")
        
    except Exception as e:
        logger.error(f"Background training failed: {str(e)}")
        # Save error record
        error_record = {
            "model_name": model_name,
            "user_id": user_id,
            "error": str(e),
            "timestamp": datetime.now(),
            "status": "failed"
        }
        await database_manager.insert_one("training_history", error_record)

@router.post("/predict", response_model=Dict[str, Any])
async def make_prediction(
    request: PredictionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Make a prediction using a trained model
    """
    try:
        if request.model_name not in continuous_learning_engine.model_registry:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")
        
        prediction = await continuous_learning_engine.predict(request.model_name, request.features)
        
        # Log prediction for analytics
        prediction_record = {
            "model_name": request.model_name,
            "user_id": current_user["user_id"],
            "features": request.features,
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "timestamp": datetime.now()
        }
        
        await database_manager.insert_one("predictions", prediction_record)
        
        return {
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "model_name": request.model_name,
            "timestamp": prediction["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_model_performance(current_user: Dict = Depends(get_current_user)):
    """
    Get performance metrics for all trained models
    """
    try:
        performance = await continuous_learning_engine.get_model_performance()
        
        # Add training history from database
        training_history = await database_manager.find_many(
            "training_history",
            {"user_id": current_user["user_id"], "status": "completed"},
            sort=[("timestamp", -1)],
            limit=50
        )
        
        return {
            "model_performance": performance,
            "training_history": training_history,
            "total_models": len(continuous_learning_engine.models),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

@router.get("/insights", response_model=Dict[str, Any])
async def get_learning_insights(current_user: Dict = Depends(get_current_user)):
    """
    Get AI-generated learning insights
    """
    try:
        insights = await continuous_learning_engine.get_learning_insights()
        
        return {
            "insights": insights,
            "insight_count": len(insights),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Insights retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.post("/configure-learning", response_model=Dict[str, Any])
async def configure_continuous_learning(
    request: LearningConfigRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Configure continuous learning parameters
    """
    try:
        # Update learning engine configuration
        continuous_learning_engine.retrain_threshold = request.retrain_threshold
        continuous_learning_engine.performance_threshold = request.performance_threshold
        
        # Save configuration to database
        config_record = {
            "user_id": current_user["user_id"],
            "retrain_threshold": request.retrain_threshold,
            "performance_threshold": request.performance_threshold,
            "schedule": request.schedule,
            "auto_feature_engineering": request.auto_feature_engineering,
            "timestamp": datetime.now()
        }
        
        await database_manager.upsert_one(
            "learning_config",
            {"user_id": current_user["user_id"]},
            config_record
        )
        
        return {
            "status": "configuration_updated",
            "config": config_record,
            "message": "Continuous learning configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/feed-data", response_model=Dict[str, Any])
async def feed_learning_data(
    request: DataFeedRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Feed new data to the continuous learning system
    """
    try:
        timestamp = request.timestamp if request.timestamp else datetime.now().isoformat()
        
        # Store data for continuous learning
        data_record = {
            "user_id": current_user["user_id"],
            "data_type": request.data_type,
            "data": request.data,
            "timestamp": timestamp,
            "processed": False
        }
        
        await database_manager.insert_one("learning_data", data_record)
        
        # Check if we need to trigger retraining
        data_count = await database_manager.count_documents(
            "learning_data",
            {"user_id": current_user["user_id"], "processed": False}
        )
        
        retrain_needed = data_count >= continuous_learning_engine.retrain_threshold
        
        return {
            "status": "data_received",
            "data_type": request.data_type,
            "new_data_count": data_count,
            "retrain_needed": retrain_needed,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Data feed error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data feed failed: {str(e)}")

@router.post("/ensemble/create", response_model=Dict[str, Any])
async def create_model_ensemble(
    request: EnsembleRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a model ensemble for improved accuracy
    """
    try:
        # Validate that all specified models exist
        for model_config in request.models:
            model_name = model_config["name"]
            if model_name not in continuous_learning_engine.models:
                raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
        
        # Create ensemble configuration
        ensemble_config = {
            "ensemble_name": request.ensemble_name,
            "user_id": current_user["user_id"],
            "models": request.models,
            "voting_strategy": request.voting_strategy,
            "created_at": datetime.now()
        }
        
        await database_manager.insert_one("ensembles", ensemble_config)
        
        return {
            "status": "ensemble_created",
            "ensemble_name": request.ensemble_name,
            "models": request.models,
            "voting_strategy": request.voting_strategy
        }
        
    except Exception as e:
        logger.error(f"Ensemble creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ensemble creation failed: {str(e)}")

@router.post("/ab-test/setup", response_model=Dict[str, Any])
async def setup_ab_test(
    request: ABTestRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Set up A/B testing for model comparison
    """
    try:
        # Validate models exist
        for model_name in [request.control_model, request.test_model]:
            if model_name not in continuous_learning_engine.models:
                raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
        
        # Create A/B test configuration
        ab_test_config = {
            "test_name": request.test_name,
            "user_id": current_user["user_id"],
            "control_model": request.control_model,
            "test_model": request.test_model,
            "traffic_split": request.traffic_split,
            "success_metric": request.success_metric,
            "duration_days": request.duration_days,
            "start_date": datetime.now(),
            "end_date": datetime.now() + pd.Timedelta(days=request.duration_days),
            "status": "active"
        }
        
        await database_manager.insert_one("ab_tests", ab_test_config)
        
        return {
            "status": "ab_test_started",
            "test_name": request.test_name,
            "control_model": request.control_model,
            "test_model": request.test_model,
            "traffic_split": request.traffic_split,
            "end_date": ab_test_config["end_date"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"A/B test setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"A/B test setup failed: {str(e)}")

@router.get("/ab-test/{test_name}/results", response_model=Dict[str, Any])
async def get_ab_test_results(
    test_name: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get A/B test results
    """
    try:
        # Get A/B test configuration
        ab_test = await database_manager.find_one(
            "ab_tests",
            {"test_name": test_name, "user_id": current_user["user_id"]}
        )
        
        if not ab_test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        # Get test results from predictions
        control_predictions = await database_manager.find_many(
            "predictions",
            {
                "model_name": ab_test["control_model"],
                "user_id": current_user["user_id"],
                "timestamp": {"$gte": ab_test["start_date"]}
            }
        )
        
        test_predictions = await database_manager.find_many(
            "predictions",
            {
                "model_name": ab_test["test_model"],
                "user_id": current_user["user_id"],
                "timestamp": {"$gte": ab_test["start_date"]}
            }
        )
        
        # Calculate performance metrics
        control_accuracy = np.mean([p["confidence"] for p in control_predictions]) if control_predictions else 0
        test_accuracy = np.mean([p["confidence"] for p in test_predictions]) if test_predictions else 0
        
        improvement = ((test_accuracy - control_accuracy) / control_accuracy * 100) if control_accuracy > 0 else 0
        
        return {
            "test_name": test_name,
            "status": ab_test["status"],
            "control_model": ab_test["control_model"],
            "test_model": ab_test["test_model"],
            "control_predictions": len(control_predictions),
            "test_predictions": len(test_predictions),
            "control_accuracy": control_accuracy,
            "test_accuracy": test_accuracy,
            "improvement_percentage": improvement,
            "statistical_significance": improvement > 5.0,  # Simple threshold
            "end_date": ab_test["end_date"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"A/B test results error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get A/B test results: {str(e)}")

@router.post("/export/{model_name}", response_model=Dict[str, Any])
async def export_model(
    model_name: str,
    format: str = "onnx",
    current_user: Dict = Depends(get_current_user)
):
    """
    Export a trained model for deployment
    """
    try:
        if model_name not in continuous_learning_engine.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        export_path = await continuous_learning_engine.export_model(model_name, format)
        
        # Log export activity
        export_record = {
            "model_name": model_name,
            "user_id": current_user["user_id"],
            "format": format,
            "export_path": export_path,
            "timestamp": datetime.now()
        }
        
        await database_manager.insert_one("model_exports", export_record)
        
        return {
            "status": "model_exported",
            "model_name": model_name,
            "format": format,
            "export_path": export_path,
            "download_url": f"/api/ml/download/{model_name}.{format}"
        }
        
    except Exception as e:
        logger.error(f"Model export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")

@router.get("/models", response_model=Dict[str, Any])
async def list_available_models(current_user: Dict = Depends(get_current_user)):
    """
    List all available models and their configurations
    """
    try:
        models_info = {}
        
        for model_name, config in continuous_learning_engine.model_registry.items():
            model_info = {
                "model_type": config["model_type"],
                "features": config["features"],
                "target": config["target"],
                "is_trained": model_name in continuous_learning_engine.models,
                "last_trained": None,
                "accuracy": None
            }
            
            # Get latest training info
            latest_training = await database_manager.find_one(
                "training_history",
                {"model_name": model_name, "user_id": current_user["user_id"], "status": "completed"},
                sort=[("timestamp", -1)]
            )
            
            if latest_training:
                model_info["last_trained"] = latest_training["timestamp"].isoformat()
                model_info["accuracy"] = latest_training["accuracy"]
            
            models_info[model_name] = model_info
        
        return {
            "available_models": models_info,
            "total_models": len(models_info),
            "trained_models": len([m for m in models_info.values() if m["is_trained"]])
        }
        
    except Exception as e:
        logger.error(f"Models listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.delete("/models/{model_name}", response_model=Dict[str, Any])
async def delete_model(
    model_name: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a trained model
    """
    try:
        if model_name not in continuous_learning_engine.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Remove model from memory
        del continuous_learning_engine.models[model_name]
        del continuous_learning_engine.scalers[model_name]
        del continuous_learning_engine.encoders[model_name]
        
        # Delete model file
        model_path = continuous_learning_engine.model_dir / f"{model_name}.joblib"
        if model_path.exists():
            model_path.unlink()
        
        # Log deletion
        deletion_record = {
            "model_name": model_name,
            "user_id": current_user["user_id"],
            "timestamp": datetime.now(),
            "action": "deleted"
        }
        
        await database_manager.insert_one("model_actions", deletion_record)
        
        return {
            "status": "model_deleted",
            "model_name": model_name,
            "message": f"Model {model_name} has been successfully deleted"
        }
        
    except Exception as e:
        logger.error(f"Model deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {str(e)}")