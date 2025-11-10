"""
Lead Scoring Machine Learning Engine
Advanced algorithms for lead qualification and conversion prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LeadScoringFeatures:
    """Feature definitions for lead scoring"""
    
    # Demographic Features
    has_company: bool = False
    company_size: int = 0  # Number of employees
    industry_relevance: float = 0.0  # 0-1 score
    title_seniority: float = 0.0  # 0-1 score (C-level, VP, Manager, etc.)
    geographic_relevance: float = 0.0  # 0-1 score
    
    # Behavioral Features
    email_opens: int = 0
    email_clicks: int = 0
    website_visits: int = 0
    page_views: int = 0
    time_on_site: float = 0.0  # minutes
    downloads: int = 0
    form_submissions: int = 0
    social_engagement: int = 0
    
    # Engagement Features
    response_rate: float = 0.0  # 0-1
    meeting_acceptance: float = 0.0  # 0-1
    call_duration_avg: float = 0.0  # minutes
    last_engagement_days: int = 999  # days since last engagement
    engagement_frequency: float = 0.0  # interactions per week
    
    # Intent Features
    pricing_page_visits: int = 0
    demo_requests: int = 0
    trial_signups: int = 0
    competitor_comparisons: int = 0
    product_questions: int = 0
    
    # Source Features
    source_quality: float = 0.0  # 0-1 based on historical conversion rates
    referral_source: bool = False
    paid_channel: bool = False
    organic_source: bool = False
    
    # Temporal Features
    lead_age_days: int = 0
    business_hours_activity: float = 0.0  # 0-1
    weekend_activity: float = 0.0  # 0-1
    response_time_hours: float = 24.0  # hours to respond
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models"""
        return np.array([
            self.has_company,
            self.company_size,
            self.industry_relevance,
            self.title_seniority,
            self.geographic_relevance,
            self.email_opens,
            self.email_clicks,
            self.website_visits,
            self.page_views,
            self.time_on_site,
            self.downloads,
            self.form_submissions,
            self.social_engagement,
            self.response_rate,
            self.meeting_acceptance,
            self.call_duration_avg,
            self.last_engagement_days,
            self.engagement_frequency,
            self.pricing_page_visits,
            self.demo_requests,
            self.trial_signups,
            self.competitor_comparisons,
            self.product_questions,
            self.source_quality,
            self.referral_source,
            self.paid_channel,
            self.organic_source,
            self.lead_age_days,
            self.business_hours_activity,
            self.weekend_activity,
            self.response_time_hours
        ], dtype=np.float32)


class LeadScoringEngine:
    """Advanced lead scoring and qualification engine"""
    
    def __init__(self):
        self.feature_weights = self._initialize_feature_weights()
        self.industry_scores = self._initialize_industry_scores()
        self.title_scores = self._initialize_title_scores()
        self.source_scores = self._initialize_source_scores()
        
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature weights based on business knowledge"""
        return {
            # Demographic weights (25%)
            'demographic': {
                'has_company': 0.05,
                'company_size': 0.08,
                'industry_relevance': 0.07,
                'title_seniority': 0.05
            },
            
            # Behavioral weights (35%)
            'behavioral': {
                'website_engagement': 0.15,
                'email_engagement': 0.10,
                'content_engagement': 0.10
            },
            
            # Intent weights (25%)
            'intent': {
                'product_interest': 0.15,
                'purchase_signals': 0.10
            },
            
            # Engagement weights (15%)
            'engagement': {
                'response_quality': 0.08,
                'meeting_behavior': 0.07
            }
        }
    
    def _initialize_industry_scores(self) -> Dict[str, float]:
        """Initialize industry relevance scores"""
        return {
            'technology': 0.9,
            'software': 0.95,
            'saas': 1.0,
            'financial_services': 0.8,
            'healthcare': 0.75,
            'manufacturing': 0.7,
            'retail': 0.6,
            'education': 0.5,
            'non_profit': 0.3,
            'government': 0.4,
            'other': 0.5
        }
    
    def _initialize_title_scores(self) -> Dict[str, float]:
        """Initialize title seniority scores"""
        return {
            # C-Level
            'ceo': 1.0, 'cto': 0.95, 'cfo': 0.9, 'cmo': 0.9, 'coo': 0.9,
            
            # VP Level
            'vp': 0.8, 'vice_president': 0.8, 'svp': 0.85,
            
            # Director Level
            'director': 0.7, 'head': 0.7,
            
            # Manager Level
            'manager': 0.6, 'lead': 0.55, 'senior': 0.5,
            
            # Individual Contributor
            'analyst': 0.4, 'specialist': 0.4, 'coordinator': 0.3,
            
            # Others
            'intern': 0.1, 'assistant': 0.2, 'other': 0.4
        }
    
    def _initialize_source_scores(self) -> Dict[str, float]:
        """Initialize source quality scores based on historical data"""
        return {
            'referral': 0.9,
            'direct': 0.8,
            'organic_search': 0.7,
            'social_media': 0.6,
            'email_campaign': 0.5,
            'paid_search': 0.5,
            'trade_show': 0.7,
            'webinar': 0.6,
            'content_download': 0.5,
            'cold_outreach': 0.3,
            'other': 0.4
        }
    
    def extract_features(self, lead_data: Dict[str, Any]) -> LeadScoringFeatures:
        """Extract features from lead data"""
        features = LeadScoringFeatures()
        
        # Demographic features
        features.has_company = bool(lead_data.get('company'))
        features.company_size = self._estimate_company_size(lead_data.get('company', ''))
        features.industry_relevance = self._calculate_industry_relevance(
            lead_data.get('industry', '')
        )
        features.title_seniority = self._calculate_title_seniority(
            lead_data.get('title', '')
        )
        
        # Behavioral features from activities
        activities = lead_data.get('activities', [])
        features.email_opens = self._count_activity_type(activities, 'email_open')
        features.email_clicks = self._count_activity_type(activities, 'email_click')
        features.website_visits = self._count_activity_type(activities, 'website_visit')
        features.page_views = self._count_activity_type(activities, 'page_view')
        features.downloads = self._count_activity_type(activities, 'download')
        
        # Engagement features
        features.response_rate = self._calculate_response_rate(activities)
        features.last_engagement_days = self._days_since_last_engagement(activities)
        features.engagement_frequency = self._calculate_engagement_frequency(activities)
        
        # Intent features
        features.pricing_page_visits = self._count_activity_type(activities, 'pricing_visit')
        features.demo_requests = self._count_activity_type(activities, 'demo_request')
        features.trial_signups = self._count_activity_type(activities, 'trial_signup')
        
        # Source features
        source = lead_data.get('source', 'other')
        features.source_quality = self.source_scores.get(source, 0.4)
        features.referral_source = source == 'referral'
        
        # Temporal features
        created_at = lead_data.get('created_at', datetime.utcnow())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        features.lead_age_days = (datetime.utcnow() - created_at).days
        
        return features
    
    def calculate_lead_score(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive lead score"""
        features = self.extract_features(lead_data)
        
        # Calculate sub-scores
        demographic_score = self._calculate_demographic_score(features)
        behavioral_score = self._calculate_behavioral_score(features)
        engagement_score = self._calculate_engagement_score(features)
        intent_score = self._calculate_intent_score(features)
        
        # Calculate weighted overall score
        overall_score = (
            demographic_score * 0.25 +
            behavioral_score * 0.35 +
            engagement_score * 0.15 +
            intent_score * 0.25
        )
        
        # Apply temporal decay for old leads
        overall_score = self._apply_temporal_decay(overall_score, features.lead_age_days)
        
        # Calculate conversion probability using logistic function
        conversion_probability = self._calculate_conversion_probability(overall_score)
        
        return {
            'overall_score': min(max(overall_score, 0.0), 1.0),
            'demographic_score': demographic_score,
            'behavioral_score': behavioral_score,
            'engagement_score': engagement_score,
            'intent_score': intent_score,
            'conversion_probability': conversion_probability,
            'factors': self._get_scoring_factors(features, overall_score)
        }
    
    def _calculate_demographic_score(self, features: LeadScoringFeatures) -> float:
        """Calculate demographic score"""
        score = 0.0
        
        # Company presence
        if features.has_company:
            score += 0.2
        
        # Company size (normalized)
        if features.company_size > 0:
            company_size_score = min(features.company_size / 1000, 1.0)
            score += company_size_score * 0.3
        
        # Industry relevance
        score += features.industry_relevance * 0.3
        
        # Title seniority
        score += features.title_seniority * 0.2
        
        return min(score, 1.0)
    
    def _calculate_behavioral_score(self, features: LeadScoringFeatures) -> float:
        """Calculate behavioral engagement score"""
        score = 0.0
        
        # Email engagement
        email_score = min((features.email_opens * 0.1 + features.email_clicks * 0.2) / 5, 1.0)
        score += email_score * 0.3
        
        # Website engagement
        website_score = min((features.website_visits * 0.2 + features.page_views * 0.1) / 10, 1.0)
        score += website_score * 0.4
        
        # Content engagement
        content_score = min(features.downloads * 0.3 / 3, 1.0)
        score += content_score * 0.3
        
        return min(score, 1.0)
    
    def _calculate_engagement_score(self, features: LeadScoringFeatures) -> float:
        """Calculate engagement quality score"""
        score = 0.0
        
        # Response rate
        score += features.response_rate * 0.4
        
        # Meeting acceptance
        score += features.meeting_acceptance * 0.3
        
        # Recency of engagement (inverse of days)
        recency_score = max(0, 1 - features.last_engagement_days / 30)
        score += recency_score * 0.3
        
        return min(score, 1.0)
    
    def _calculate_intent_score(self, features: LeadScoringFeatures) -> float:
        """Calculate purchase intent score"""
        score = 0.0
        
        # High-intent activities
        if features.pricing_page_visits > 0:
            score += 0.3
        if features.demo_requests > 0:
            score += 0.4
        if features.trial_signups > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_conversion_probability(self, overall_score: float) -> float:
        """Calculate conversion probability using logistic regression"""
        # Sigmoid function to map score to probability
        return 1 / (1 + np.exp(-5 * (overall_score - 0.5)))
    
    def _apply_temporal_decay(self, score: float, age_days: int) -> float:
        """Apply temporal decay to account for lead age"""
        if age_days <= 7:
            return score
        elif age_days <= 30:
            return score * 0.9
        elif age_days <= 90:
            return score * 0.8
        else:
            return score * 0.7
    
    def _get_scoring_factors(self, features: LeadScoringFeatures, score: float) -> Dict[str, Any]:
        """Get factors contributing to the score"""
        factors = {
            'positive_factors': [],
            'negative_factors': [],
            'recommendations': []
        }
        
        # Positive factors
        if features.has_company:
            factors['positive_factors'].append('Has company information')
        if features.industry_relevance > 0.7:
            factors['positive_factors'].append('Highly relevant industry')
        if features.title_seniority > 0.6:
            factors['positive_factors'].append('Senior title/decision maker')
        if features.demo_requests > 0:
            factors['positive_factors'].append('Requested product demo')
        if features.pricing_page_visits > 0:
            factors['positive_factors'].append('Visited pricing page')
        
        # Negative factors
        if features.last_engagement_days > 30:
            factors['negative_factors'].append('No recent engagement')
        if features.response_rate < 0.3:
            factors['negative_factors'].append('Low response rate')
        if features.industry_relevance < 0.4:
            factors['negative_factors'].append('Low industry relevance')
        
        # Recommendations
        if score > 0.8:
            factors['recommendations'].append('High priority - Schedule immediate call')
        elif score > 0.6:
            factors['recommendations'].append('Good prospect - Follow up within 24 hours')
        elif score > 0.4:
            factors['recommendations'].append('Nurture with targeted content')
        else:
            factors['recommendations'].append('Low priority - Add to drip campaign')
        
        return factors
    
    def _estimate_company_size(self, company: str) -> int:
        """Estimate company size (placeholder - would use external API)"""
        if not company:
            return 0
        # This would integrate with services like Clearbit, ZoomInfo, etc.
        return 100  # Placeholder
    
    def _calculate_industry_relevance(self, industry: str) -> float:
        """Calculate industry relevance score"""
        if not industry:
            return 0.5
        
        industry_lower = industry.lower()
        for key, score in self.industry_scores.items():
            if key in industry_lower:
                return score
        
        return 0.5
    
    def _calculate_title_seniority(self, title: str) -> float:
        """Calculate title seniority score"""
        if not title:
            return 0.4
        
        title_lower = title.lower()
        for key, score in self.title_scores.items():
            if key in title_lower:
                return score
        
        return 0.4
    
    def _count_activity_type(self, activities: List[Dict], activity_type: str) -> int:
        """Count activities of specific type"""
        return len([a for a in activities if a.get('activity_type') == activity_type])
    
    def _calculate_response_rate(self, activities: List[Dict]) -> float:
        """Calculate response rate from activities"""
        sent_emails = self._count_activity_type(activities, 'email_sent')
        responses = self._count_activity_type(activities, 'email_reply')
        
        if sent_emails == 0:
            return 0.0
        
        return min(responses / sent_emails, 1.0)
    
    def _days_since_last_engagement(self, activities: List[Dict]) -> int:
        """Calculate days since last engagement"""
        if not activities:
            return 999
        
        latest_activity = max(
            activities,
            key=lambda x: x.get('created_at', '1970-01-01'),
            default={'created_at': '1970-01-01'}
        )
        
        try:
            last_date = datetime.fromisoformat(
                latest_activity['created_at'].replace('Z', '+00:00')
            )
            return (datetime.utcnow() - last_date).days
        except:
            return 999
    
    def _calculate_engagement_frequency(self, activities: List[Dict]) -> float:
        """Calculate engagement frequency (interactions per week)"""
        if not activities:
            return 0.0
        
        # Calculate over the last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_activities = [
            a for a in activities
            if datetime.fromisoformat(a.get('created_at', '1970-01-01').replace('Z', '+00:00')) > thirty_days_ago
        ]
        
        return len(recent_activities) / 4.3  # ~4.3 weeks in 30 days


# Global instance
lead_scoring_engine = LeadScoringEngine()