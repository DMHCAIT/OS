#!/usr/bin/env python3
"""
Data Validation and Preparation Script
Validates your real sales data for production training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and prepares real sales data for AI training"""
    
    def __init__(self):
        self.data_dir = Path("data/training")
        self.required_files = {
            'leads': 'historical_leads.csv',
            'conversations': 'sales_conversations.csv', 
            'voice': 'voice_call_data.csv',
            'deals': 'historical_deals.csv'
        }
        
    def validate_all_data(self) -> Dict[str, Dict]:
        """Validate all data files for training readiness"""
        logger.info("🔍 Validating all data files for production training...")
        
        validation_results = {}
        
        for data_type, filename in self.required_files.items():
            filepath = self.data_dir / filename
            validation_results[data_type] = self.validate_file(filepath, data_type)
        
        self.print_validation_summary(validation_results)
        return validation_results
    
    def validate_file(self, filepath: Path, data_type: str) -> Dict:
        """Validate individual data file"""
        result = {
            'exists': filepath.exists(),
            'size': 0,
            'quality_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        if not result['exists']:
            result['issues'].append(f"File {filepath.name} does not exist")
            result['recommendations'].append(f"Create {filepath.name} with your historical {data_type} data")
            return result
        
        try:
            df = pd.read_csv(filepath)
            result['size'] = len(df)
            
            if data_type == 'leads':
                result.update(self.validate_leads_data(df))
            elif data_type == 'conversations':
                result.update(self.validate_conversations_data(df))
            elif data_type == 'voice':
                result.update(self.validate_voice_data(df))
            elif data_type == 'deals':
                result.update(self.validate_deals_data(df))
                
        except Exception as e:
            result['issues'].append(f"Error reading file: {e}")
            
        return result
    
    def validate_leads_data(self, df: pd.DataFrame) -> Dict:
        """Validate leads data quality"""
        issues = []
        recommendations = []
        quality_score = 100
        
        # Required columns for lead scoring
        required_cols = [
            'lead_id', 'company_size', 'industry', 'job_title',
            'email_engagement_rate', 'website_visits', 'converted'
        ]
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            quality_score -= 30
        
        # Check data size
        if len(df) < 100:
            issues.append(f"Only {len(df)} leads - recommend minimum 1000 for good accuracy")
            quality_score -= 20
        elif len(df) < 1000:
            issues.append(f"Only {len(df)} leads - recommend 1000+ for optimal accuracy")
            quality_score -= 10
        
        # Check conversion rate
        if 'converted' in df.columns:
            conversion_rate = df['converted'].mean()
            if conversion_rate < 0.05:
                issues.append(f"Low conversion rate: {conversion_rate:.1%}")
                recommendations.append("Include more converted leads for balanced training")
            elif conversion_rate > 0.50:
                issues.append(f"Unusually high conversion rate: {conversion_rate:.1%}")
                recommendations.append("Verify data quality and labeling")
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.1:
            issues.append(f"High missing values: {missing_pct:.1%}")
            quality_score -= 15
        
        # Check data diversity
        if 'industry' in df.columns:
            industry_diversity = len(df['industry'].unique()) / len(df)
            if industry_diversity < 0.05:
                recommendations.append("Include more industry diversity for better generalization")
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def validate_conversations_data(self, df: pd.DataFrame) -> Dict:
        """Validate conversation data quality"""
        issues = []
        recommendations = []
        quality_score = 100
        
        required_cols = ['conversation_id', 'conversation_text', 'outcome']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            quality_score -= 30
        
        if len(df) < 50:
            issues.append(f"Only {len(df)} conversations - recommend minimum 500")
            quality_score -= 25
        
        # Check text quality
        if 'conversation_text' in df.columns:
            avg_length = df['conversation_text'].str.len().mean()
            if avg_length < 50:
                issues.append("Conversation texts are too short")
                recommendations.append("Include fuller conversation transcripts")
        
        # Check outcome distribution
        if 'outcome' in df.columns:
            outcome_dist = df['outcome'].value_counts()
            if len(outcome_dist) < 2:
                issues.append("Need more diverse conversation outcomes")
                quality_score -= 20
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def validate_voice_data(self, df: pd.DataFrame) -> Dict:
        """Validate voice data quality"""
        issues = []
        recommendations = []
        quality_score = 100
        
        required_cols = ['call_id', 'emotion_label', 'call_outcome']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            quality_score -= 30
        
        if len(df) < 25:
            issues.append(f"Only {len(df)} voice samples - recommend minimum 100")
            quality_score -= 30
        
        # Check emotion diversity
        if 'emotion_label' in df.columns:
            emotion_count = len(df['emotion_label'].unique())
            if emotion_count < 3:
                issues.append("Need more diverse emotion labels")
                quality_score -= 20
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def validate_deals_data(self, df: pd.DataFrame) -> Dict:
        """Validate deals data quality"""
        issues = []
        recommendations = []
        quality_score = 100
        
        required_cols = ['deal_id', 'deal_value', 'close_date']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            quality_score -= 30
        
        if len(df) < 50:
            issues.append(f"Only {len(df)} deals - recommend minimum 200")
            quality_score -= 25
        
        # Check deal value distribution
        if 'deal_value' in df.columns:
            if df['deal_value'].std() / df['deal_value'].mean() > 2:
                recommendations.append("High variance in deal values - consider segmentation")
        
        # Check date range
        if 'close_date' in df.columns:
            try:
                dates = pd.to_datetime(df['close_date'])
                date_range = (dates.max() - dates.min()).days
                if date_range < 90:
                    issues.append("Short time period for deals data")
                    recommendations.append("Include at least 6-12 months of historical data")
            except:
                issues.append("Invalid date format in close_date")
                quality_score -= 20
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def print_validation_summary(self, results: Dict):
        """Print comprehensive validation summary"""
        print("\n🔍 DATA VALIDATION REPORT")
        print("=" * 50)
        
        overall_score = np.mean([r['quality_score'] for r in results.values()])
        
        for data_type, result in results.items():
            status = "✅" if result['exists'] else "❌"
            print(f"\n{status} {data_type.upper()} DATA:")
            
            if result['exists']:
                print(f"   📊 Records: {result['size']}")
                print(f"   🎯 Quality Score: {result['quality_score']}/100")
                
                if result['issues']:
                    print(f"   ⚠️  Issues:")
                    for issue in result['issues']:
                        print(f"      • {issue}")
                
                if result['recommendations']:
                    print(f"   💡 Recommendations:")
                    for rec in result['recommendations']:
                        print(f"      • {rec}")
            else:
                print(f"   📝 File missing: {self.required_files[data_type]}")
        
        print(f"\n📊 OVERALL DATA QUALITY: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print("🎉 EXCELLENT - Your data is ready for production training!")
        elif overall_score >= 60:
            print("👍 GOOD - Address minor issues for optimal results")
        elif overall_score >= 40:
            print("⚠️  FAIR - Significant improvements needed")
        else:
            print("❌ POOR - Major data preparation required")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Address data quality issues above")
        print(f"   2. Run: python train_production_models.py")
        print(f"   3. Monitor model performance")
    
    def create_data_templates(self):
        """Create data templates with instructions"""
        self.data_dir.mkdir(exist_ok=True)
        
        # Create leads template with instructions
        leads_template = """# LEADS DATA TEMPLATE
# Fill this CSV with your historical lead data for training
# 
# REQUIRED FIELDS:
# - lead_id: Unique identifier for each lead
# - company_size: "1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"
# - industry: "Technology", "Healthcare", "Finance", "Retail", etc.
# - job_title: Contact's job title
# - email_engagement_rate: 0.0 to 1.0 (email open/click rate)
# - website_visits: Number of website visits
# - converted: 1 if lead converted, 0 if not (TARGET VARIABLE)
#
# OPTIONAL BUT RECOMMENDED:
# - content_downloads: Number of content pieces downloaded
# - meeting_acceptance_rate: 0.0 to 1.0
# - response_time_hours: Average response time
# - annual_revenue: Company revenue
# - deal_value: Final deal value (0 if not converted)
#
# MINIMUM RECOMMENDED: 1000+ leads for good accuracy
# OPTIMAL: 10,000+ leads across 6+ months

lead_id,company_size,industry,job_title,email_engagement_rate,website_visits,content_downloads,meeting_acceptance_rate,response_time_hours,lead_source,annual_revenue,number_of_employees,decision_maker,budget_indicated,timeline,previous_interactions,competitor_mentions,urgency_score,converted,deal_value,conversion_days
# Add your data below this line
"""
        
        with open(self.data_dir / "historical_leads.csv", "w") as f:
            f.write(leads_template)
        
        print(f"📝 Created data templates in {self.data_dir}/")
        print(f"📋 Fill in your historical data and run validation again")

def main():
    """Main validation function"""
    print("🔍 AI Sales System - Data Validation & Preparation")
    print("=" * 55)
    
    validator = DataValidator()
    
    # Check if data directory exists
    if not validator.data_dir.exists():
        print("📁 Creating data directory and templates...")
        validator.create_data_templates()
        return
    
    # Validate existing data
    results = validator.validate_all_data()
    
    # Check if we need to create templates
    if not any(r['exists'] for r in results.values()):
        print("\n📝 No data files found. Creating templates...")
        validator.create_data_templates()

if __name__ == "__main__":
    main()