# Training and Learning System Setup Guide

## 🧠 How to Train the AI System for Maximum Learning

### Overview
Your Predictive Business Intelligence system uses advanced machine learning that continuously learns from your sales data. Here's how to set it up and optimize it for maximum performance.

## 🚀 Quick Start Training

### 1. Initial Data Setup
```bash
# Navigate to your project directory
cd /Users/rubeenakhan/Desktop/OS

# Create training data directory
mkdir -p backend/data/training
mkdir -p backend/data/models
mkdir -p backend/data/exports
```

### 2. Upload Your Sales Data
Create CSV files with your historical data in `backend/data/training/`:

```csv
# leads.csv
email,company_size,industry,title,source,engagement_score,converted
john@company.com,500,Technology,CTO,Website,85,1
sarah@corp.com,50,Finance,CFO,LinkedIn,72,1
...

# deals.csv  
lead_id,company_size,industry,title,budget_range,urgency,deal_amount
1,500,Technology,CTO,50000,High,45000
2,50,Finance,CFO,10000,Medium,8500
...

# customers.csv
customer_id,last_contact,engagement_trend,support_tickets,usage_pattern,churned
1,2024-01-15,increasing,2,high,0
2,2024-01-10,decreasing,5,low,1
...
```

### 3. Start Training Process
```bash
# Install training dependencies
pip install scikit-learn xgboost lightgbm pandas numpy joblib

# Start the training API
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up -d
```

## 🎯 Training Your Models

### 1. Lead Conversion Model
```python
# Train lead conversion predictions
import requests

training_data = {
    "model_name": "lead_conversion",
    "data": [
        {
            "engagement_score": 85,
            "company_size": 500,
            "industry": "Technology",
            "title": "CTO",
            "source": "Website",
            "converted": 1
        },
        # Add more training examples...
    ]
}

response = requests.post(
    "http://localhost:8000/api/ml/train",
    json=training_data
)
print(f"Training completed: {response.json()}")
```

### 2. Deal Size Prediction
```python
# Train deal size predictions
deal_training_data = {
    "model_name": "deal_size_prediction", 
    "data": [
        {
            "company_size": 500,
            "industry": "Technology",
            "title": "CTO", 
            "budget_range": 50000,
            "urgency": "High",
            "deal_amount": 45000
        },
        # Add more examples...
    ]
}

response = requests.post(
    "http://localhost:8000/api/ml/train",
    json=deal_training_data
)
```

### 3. Automated Bulk Training
```python
# Upload CSV files for bulk training
import pandas as pd

def bulk_train_from_csv():
    # Load your CSV files
    leads_df = pd.read_csv('backend/data/training/leads.csv')
    deals_df = pd.read_csv('backend/data/training/deals.csv')
    customers_df = pd.read_csv('backend/data/training/customers.csv')
    
    # Train each model
    models = [
        ("lead_conversion", leads_df.to_dict('records')),
        ("deal_size_prediction", deals_df.to_dict('records')),
        ("churn_prediction", customers_df.to_dict('records'))
    ]
    
    for model_name, data in models:
        response = requests.post(
            "http://localhost:8000/api/ml/train",
            json={"model_name": model_name, "data": data}
        )
        print(f"{model_name}: {response.json()}")

# Run bulk training
bulk_train_from_csv()
```

## 🔄 Continuous Learning Setup

### 1. Enable Auto-Learning
The system automatically retrains models when:
- 1,000+ new data points are collected
- Model performance drops by 10%+
- Weekly schedule (configurable)

```python
# Configure continuous learning
learning_config = {
    "retrain_threshold": 1000,  # Retrain after 1000 new data points
    "performance_threshold": 0.1,  # Retrain if performance drops 10%
    "schedule": "weekly"  # Options: hourly, daily, weekly
}

requests.post(
    "http://localhost:8000/api/ml/configure-learning",
    json=learning_config
)
```

### 2. Real-Time Data Integration
```python
# Automatically feed new data for learning
def feed_new_interaction(interaction_data):
    """Feed new sales interactions to the learning system"""
    response = requests.post(
        "http://localhost:8000/api/ml/feed-data",
        json={
            "data_type": "interaction",
            "data": interaction_data,
            "timestamp": datetime.now().isoformat()
        }
    )
    return response.json()

# Example: Feed email interaction
email_interaction = {
    "lead_id": "12345",
    "email_opened": True,
    "clicked_link": True,
    "reply_received": False,
    "engagement_score": 75
}

feed_new_interaction(email_interaction)
```

## 📊 Monitoring Learning Progress

### 1. Check Model Performance
```bash
# Get model performance metrics
curl http://localhost:8000/api/ml/performance

# Response:
{
  "lead_conversion": {
    "accuracy": 0.85,
    "last_trained": "2024-01-15T10:30:00",
    "data_size": 5000,
    "cross_val_score": 0.83
  },
  "deal_size_prediction": {
    "accuracy": 0.78,
    "last_trained": "2024-01-15T10:45:00", 
    "data_size": 3200,
    "cross_val_score": 0.76
  }
}
```

### 2. View Learning Insights
```bash
# Get AI-generated insights
curl http://localhost:8000/api/ml/insights

# Response:
{
  "insights": [
    {
      "insight_type": "performance_improvement",
      "description": "Lead conversion model improved by 12%",
      "confidence": 0.9,
      "impact_score": 12.0,
      "recommendation": "Continue current training strategy"
    },
    {
      "insight_type": "feature_importance", 
      "description": "Engagement score is most predictive for lead conversion",
      "confidence": 0.8,
      "recommendation": "Focus on improving engagement score data quality"
    }
  ]
}
```

## 🎯 Advanced Training Strategies

### 1. Feature Engineering
```python
# Add custom features for better predictions
def engineer_features(raw_data):
    """Create advanced features from raw data"""
    
    # Time-based features
    raw_data['hour_of_contact'] = pd.to_datetime(raw_data['timestamp']).dt.hour
    raw_data['day_of_week'] = pd.to_datetime(raw_data['timestamp']).dt.dayofweek
    
    # Engagement features
    raw_data['engagement_trend'] = raw_data['current_engagement'] - raw_data['avg_engagement']
    raw_data['response_speed'] = raw_data['response_time_hours']
    
    # Company features
    raw_data['company_score'] = raw_data['company_size'] * raw_data['industry_factor']
    
    # Interaction features
    raw_data['email_engagement'] = (
        raw_data['emails_opened'] / raw_data['emails_sent']
    ).fillna(0)
    
    return raw_data

# Use engineered features for training
engineered_data = engineer_features(your_raw_data)
```

### 2. A/B Testing Integration
```python
# Set up A/B testing for model improvements
def setup_ab_testing():
    """Setup A/B testing for model performance"""
    
    ab_config = {
        "test_name": "model_comparison_v2",
        "control_model": "lead_conversion_v1",
        "test_model": "lead_conversion_v2", 
        "traffic_split": 0.5,  # 50/50 split
        "success_metric": "conversion_rate",
        "duration_days": 14
    }
    
    response = requests.post(
        "http://localhost:8000/api/ml/ab-test/setup",
        json=ab_config
    )
    
    return response.json()
```

### 3. Model Ensemble Strategy
```python
# Create model ensembles for better accuracy
def create_ensemble():
    """Create ensemble of multiple models"""
    
    ensemble_config = {
        "ensemble_name": "lead_conversion_ensemble",
        "models": [
            {"name": "random_forest", "weight": 0.3},
            {"name": "xgboost", "weight": 0.4}, 
            {"name": "neural_network", "weight": 0.3}
        ],
        "voting_strategy": "soft"  # or "hard"
    }
    
    response = requests.post(
        "http://localhost:8000/api/ml/ensemble/create",
        json=ensemble_config
    )
    
    return response.json()
```

## 🔧 Production Training Pipeline

### 1. Automated Training Schedule
```yaml
# docker-compose.training.yml
version: '3.8'
services:
  training-scheduler:
    image: your-app:latest
    command: python -m app.ml.training_scheduler
    environment:
      - TRAINING_SCHEDULE=0 2 * * *  # Daily at 2 AM
      - DATA_SOURCE=mongodb://mongodb:27017/salesdb
      - MODEL_STORAGE=/app/models
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - mongodb
      - redis
```

### 2. Training Monitoring
```python
# Set up training alerts
training_alerts = {
    "performance_drop": {
        "threshold": 0.1,  # 10% drop
        "notification": "slack",
        "webhook_url": "YOUR_SLACK_WEBHOOK"
    },
    "training_failure": {
        "notification": "email",
        "email": "admin@yourcompany.com"
    },
    "data_drift": {
        "threshold": 0.2,
        "notification": "dashboard"
    }
}

requests.post(
    "http://localhost:8000/api/ml/alerts/configure",
    json=training_alerts
)
```

## 🚀 Scaling Training for Large Datasets

### 1. Distributed Training
```python
# Use distributed training for large datasets
distributed_config = {
    "strategy": "data_parallel",
    "workers": 4,
    "batch_size": 1000,
    "distributed_backend": "ray"  # or "dask"
}

response = requests.post(
    "http://localhost:8000/api/ml/distributed-training",
    json={
        "model_name": "lead_conversion",
        "config": distributed_config,
        "data_path": "/app/data/large_dataset.csv"
    }
)
```

### 2. Incremental Learning
```python
# Enable incremental learning for continuous updates
incremental_config = {
    "model_name": "lead_conversion",
    "learning_rate": 0.001,
    "batch_size": 32,
    "memory_buffer": 1000,  # Keep last 1000 examples
    "update_frequency": "real_time"
}

requests.post(
    "http://localhost:8000/api/ml/incremental-learning",
    json=incremental_config
)
```

## 📈 Training Results Dashboard

Access your training dashboard at:
- **Local**: http://localhost:8000/dashboard/training
- **Production**: https://yourdomain.com/dashboard/training

Dashboard features:
- ✅ Real-time training progress
- ✅ Model performance charts  
- ✅ Feature importance analysis
- ✅ Learning insights timeline
- ✅ A/B testing results
- ✅ Data quality metrics

## 🎯 Best Practices for Maximum Learning

### 1. Data Quality
- **Clean data**: Remove duplicates and inconsistencies
- **Feature engineering**: Create meaningful derived features
- **Balanced datasets**: Ensure balanced representation
- **Regular updates**: Feed fresh data consistently

### 2. Model Management
- **Version control**: Track model versions and performance
- **A/B testing**: Compare model improvements
- **Monitoring**: Set up alerts for performance degradation
- **Backup**: Regular model backups and rollback capability

### 3. Continuous Improvement
- **Regular retraining**: Weekly or monthly model updates
- **Feature selection**: Remove irrelevant features
- **Hyperparameter tuning**: Optimize model parameters
- **Ensemble methods**: Combine multiple models

## 🔄 Training Workflow Summary

1. **Data Collection**: Gather historical sales data
2. **Data Preparation**: Clean and engineer features
3. **Initial Training**: Train all 5 core models
4. **Validation**: Test model accuracy and performance
5. **Deployment**: Deploy trained models to production
6. **Monitoring**: Track performance and data quality
7. **Continuous Learning**: Automatic retraining with new data
8. **Optimization**: Regular model improvements and tuning

Your AI system will continuously learn from every interaction, becoming more accurate and valuable over time! 🚀