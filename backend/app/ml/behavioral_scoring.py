"""
Real-time Behavioral Scoring Engine
Dynamic lead scoring based on real-time behavioral data and interactions
"""

import asyncio
import redis
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from enum import Enum
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import aioredis

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Types of behavioral interactions"""
    WEBSITE_VISIT = "website_visit"
    PAGE_VIEW = "page_view"
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    DOWNLOAD = "download"
    FORM_SUBMISSION = "form_submission"
    VIDEO_WATCH = "video_watch"
    SOCIAL_ENGAGEMENT = "social_engagement"
    PHONE_CALL = "phone_call"
    MEETING_SCHEDULED = "meeting_scheduled"
    MEETING_ATTENDED = "meeting_attended"
    PROPOSAL_VIEWED = "proposal_viewed"
    PRICING_VIEWED = "pricing_viewed"
    DEMO_REQUESTED = "demo_requested"
    TRIAL_STARTED = "trial_started"
    FEATURE_USAGE = "feature_usage"

@dataclass
class BehavioralEvent:
    """Individual behavioral event"""
    lead_id: str
    event_type: BehaviorType
    timestamp: datetime
    event_data: Dict[str, Any]
    session_id: Optional[str] = None
    source: str = "unknown"
    value_score: float = 0.0
    engagement_score: float = 0.0
    intent_score: float = 0.0

@dataclass
class BehavioralProfile:
    """Comprehensive behavioral profile for a lead"""
    lead_id: str
    total_events: int
    unique_sessions: int
    first_interaction: datetime
    last_interaction: datetime
    behavior_frequency: Dict[str, int]
    engagement_trend: float
    intent_progression: float
    interaction_velocity: float
    content_affinity: Dict[str, float]
    channel_preference: Dict[str, float]
    peak_activity_hours: List[int]
    behavioral_stage: str
    churn_risk_score: float
    conversion_momentum: float

@dataclass
class RealTimeScoringResult:
    """Real-time scoring result"""
    lead_id: str
    current_score: float
    previous_score: float
    score_change: float
    behavioral_score: float
    engagement_score: float
    intent_score: float
    urgency_score: float
    velocity_score: float
    momentum_score: float
    behavioral_profile: BehavioralProfile
    scoring_timestamp: datetime
    trigger_events: List[str]
    recommendations: List[str]

class BehavioralScoringRules:
    """Configurable scoring rules for different behaviors"""
    
    def __init__(self):
        # Base scores for different event types
        self.event_base_scores = {
            BehaviorType.WEBSITE_VISIT: 5.0,
            BehaviorType.PAGE_VIEW: 2.0,
            BehaviorType.EMAIL_OPEN: 3.0,
            BehaviorType.EMAIL_CLICK: 8.0,
            BehaviorType.DOWNLOAD: 15.0,
            BehaviorType.FORM_SUBMISSION: 20.0,
            BehaviorType.VIDEO_WATCH: 12.0,
            BehaviorType.SOCIAL_ENGAGEMENT: 6.0,
            BehaviorType.PHONE_CALL: 30.0,
            BehaviorType.MEETING_SCHEDULED: 40.0,
            BehaviorType.MEETING_ATTENDED: 50.0,
            BehaviorType.PROPOSAL_VIEWED: 35.0,
            BehaviorType.PRICING_VIEWED: 25.0,
            BehaviorType.DEMO_REQUESTED: 45.0,
            BehaviorType.TRIAL_STARTED: 60.0,
            BehaviorType.FEATURE_USAGE: 20.0
        }
        
        # Intent progression scores
        self.intent_scores = {
            BehaviorType.WEBSITE_VISIT: 0.1,
            BehaviorType.PAGE_VIEW: 0.05,
            BehaviorType.EMAIL_OPEN: 0.1,
            BehaviorType.EMAIL_CLICK: 0.2,
            BehaviorType.DOWNLOAD: 0.4,
            BehaviorType.FORM_SUBMISSION: 0.5,
            BehaviorType.VIDEO_WATCH: 0.3,
            BehaviorType.PRICING_VIEWED: 0.6,
            BehaviorType.DEMO_REQUESTED: 0.8,
            BehaviorType.TRIAL_STARTED: 0.9,
            BehaviorType.MEETING_SCHEDULED: 0.85,
            BehaviorType.MEETING_ATTENDED: 0.95
        }
        
        # Engagement multipliers based on time spent or interaction depth
        self.engagement_multipliers = {
            "high": 2.0,
            "medium": 1.5,
            "low": 1.0,
            "very_low": 0.5
        }
        
        # Recency decay factors (how much recent events matter more)
        self.recency_decay_hours = 72  # 3 days
        self.recency_weight = 0.7
        
        # Frequency bonus multipliers
        self.frequency_bonus_threshold = 3
        self.frequency_multiplier = 1.2
        
        # Velocity scoring (events per day)
        self.velocity_thresholds = {
            "high": 5.0,
            "medium": 2.0,
            "low": 0.5
        }

class RealTimeBehavioralScoringEngine:
    """Real-time behavioral scoring engine"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        scoring_window_days: int = 30,
        update_interval_seconds: int = 300  # 5 minutes
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.scoring_window_days = scoring_window_days
        self.update_interval_seconds = update_interval_seconds
        self.scoring_rules = BehavioralScoringRules()
        self.scaler = MinMaxScaler()
        
        # In-memory cache for frequently accessed data
        self.profile_cache: Dict[str, BehavioralProfile] = {}
        self.score_cache: Dict[str, float] = {}
        self.event_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background task tracking
        self.scoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("RealTimeBehavioralScoringEngine initialized")
    
    async def initialize(self):
        """Initialize the behavioral scoring engine"""
        try:
            # Connect to Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            # Start background scoring task
            await self.start_real_time_scoring()
            
            logger.info("RealTimeBehavioralScoringEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RealTimeBehavioralScoringEngine: {e}")
            raise
    
    async def start_real_time_scoring(self):
        """Start real-time scoring background task"""
        if self.scoring_task is None or self.scoring_task.done():
            self.is_running = True
            self.scoring_task = asyncio.create_task(self._scoring_loop())
            logger.info("Real-time scoring started")
    
    async def stop_real_time_scoring(self):
        """Stop real-time scoring background task"""
        self.is_running = False
        if self.scoring_task and not self.scoring_task.done():
            self.scoring_task.cancel()
            try:
                await self.scoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time scoring stopped")
    
    async def track_behavioral_event(
        self,
        lead_id: str,
        event_type: BehaviorType,
        event_data: Dict[str, Any] = None,
        session_id: Optional[str] = None,
        source: str = "unknown"
    ) -> BehavioralEvent:
        """Track a new behavioral event"""
        try:
            # Create behavioral event
            event = BehavioralEvent(
                lead_id=lead_id,
                event_type=event_type,
                timestamp=datetime.now(),
                event_data=event_data or {},
                session_id=session_id,
                source=source
            )
            
            # Calculate event scores
            await self._calculate_event_scores(event)
            
            # Store event in Redis
            await self._store_event(event)
            
            # Add to in-memory buffer
            self.event_buffer[lead_id].append(event)
            
            # Trigger immediate scoring update for high-value events
            if await self._is_high_value_event(event):
                await self.calculate_real_time_score(lead_id)
            
            logger.debug(f"Tracked behavioral event: {event_type.value} for lead {lead_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error tracking behavioral event: {e}")
            raise
    
    async def calculate_real_time_score(
        self,
        lead_id: str,
        force_refresh: bool = False
    ) -> RealTimeScoringResult:
        """Calculate real-time behavioral score for a lead"""
        try:
            # Check cache if not forcing refresh
            if not force_refresh and lead_id in self.score_cache:
                cached_time = datetime.now() - timedelta(minutes=5)
                if hasattr(self, '_last_score_time') and self._last_score_time.get(lead_id, datetime.min) > cached_time:
                    # Return cached result with updated timestamp
                    pass
            
            # Get behavioral profile
            profile = await self.get_behavioral_profile(lead_id, force_refresh=force_refresh)
            
            # Get recent events
            events = await self._get_recent_events(lead_id, days=self.scoring_window_days)
            
            # Calculate component scores
            behavioral_score = await self._calculate_behavioral_score(events)
            engagement_score = await self._calculate_engagement_score(events, profile)
            intent_score = await self._calculate_intent_score(events)
            urgency_score = await self._calculate_urgency_score(events)
            velocity_score = await self._calculate_velocity_score(events)
            momentum_score = await self._calculate_momentum_score(events)
            
            # Calculate overall score with weighted combination
            current_score = self._combine_scores(
                behavioral_score,
                engagement_score,
                intent_score,
                urgency_score,
                velocity_score,
                momentum_score
            )
            
            # Get previous score for comparison
            previous_score = self.score_cache.get(lead_id, 0.0)
            score_change = current_score - previous_score
            
            # Update cache
            self.score_cache[lead_id] = current_score
            self._last_score_time = getattr(self, '_last_score_time', {})
            self._last_score_time[lead_id] = datetime.now()
            
            # Generate trigger events and recommendations
            trigger_events = await self._identify_trigger_events(events)
            recommendations = await self._generate_recommendations(
                profile, current_score, score_change, trigger_events
            )
            
            # Create result
            result = RealTimeScoringResult(
                lead_id=lead_id,
                current_score=current_score,
                previous_score=previous_score,
                score_change=score_change,
                behavioral_score=behavioral_score,
                engagement_score=engagement_score,
                intent_score=intent_score,
                urgency_score=urgency_score,
                velocity_score=velocity_score,
                momentum_score=momentum_score,
                behavioral_profile=profile,
                scoring_timestamp=datetime.now(),
                trigger_events=trigger_events,
                recommendations=recommendations
            )
            
            # Store result in Redis
            await self._store_scoring_result(result)
            
            logger.debug(f"Calculated real-time score for lead {lead_id}: {current_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating real-time score for lead {lead_id}: {e}")
            raise
    
    async def get_behavioral_profile(
        self,
        lead_id: str,
        force_refresh: bool = False
    ) -> BehavioralProfile:
        """Get comprehensive behavioral profile for a lead"""
        try:
            # Check cache
            if not force_refresh and lead_id in self.profile_cache:
                return self.profile_cache[lead_id]
            
            # Get events from Redis
            events = await self._get_recent_events(lead_id, days=self.scoring_window_days)
            
            if not events:
                # Create empty profile for new leads
                profile = BehavioralProfile(
                    lead_id=lead_id,
                    total_events=0,
                    unique_sessions=0,
                    first_interaction=datetime.now(),
                    last_interaction=datetime.now(),
                    behavior_frequency={},
                    engagement_trend=0.0,
                    intent_progression=0.0,
                    interaction_velocity=0.0,
                    content_affinity={},
                    channel_preference={},
                    peak_activity_hours=[],
                    behavioral_stage="new",
                    churn_risk_score=0.0,
                    conversion_momentum=0.0
                )
            else:
                # Calculate profile metrics
                total_events = len(events)
                unique_sessions = len(set(e.session_id for e in events if e.session_id))
                first_interaction = min(e.timestamp for e in events)
                last_interaction = max(e.timestamp for e in events)
                
                # Behavior frequency
                behavior_frequency = defaultdict(int)
                for event in events:
                    behavior_frequency[event.event_type.value] += 1
                
                # Calculate engagement trend
                engagement_trend = await self._calculate_engagement_trend(events)
                
                # Calculate intent progression
                intent_progression = await self._calculate_intent_progression(events)
                
                # Calculate interaction velocity (events per day)
                days_active = max(1, (last_interaction - first_interaction).days)
                interaction_velocity = total_events / days_active
                
                # Content affinity analysis
                content_affinity = await self._analyze_content_affinity(events)
                
                # Channel preference analysis
                channel_preference = await self._analyze_channel_preference(events)
                
                # Peak activity hours
                peak_activity_hours = await self._identify_peak_activity_hours(events)
                
                # Behavioral stage
                behavioral_stage = await self._determine_behavioral_stage(events)
                
                # Churn risk score
                churn_risk_score = await self._calculate_churn_risk(events)
                
                # Conversion momentum
                conversion_momentum = await self._calculate_conversion_momentum(events)
                
                profile = BehavioralProfile(
                    lead_id=lead_id,
                    total_events=total_events,
                    unique_sessions=unique_sessions,
                    first_interaction=first_interaction,
                    last_interaction=last_interaction,
                    behavior_frequency=dict(behavior_frequency),
                    engagement_trend=engagement_trend,
                    intent_progression=intent_progression,
                    interaction_velocity=interaction_velocity,
                    content_affinity=content_affinity,
                    channel_preference=channel_preference,
                    peak_activity_hours=peak_activity_hours,
                    behavioral_stage=behavioral_stage,
                    churn_risk_score=churn_risk_score,
                    conversion_momentum=conversion_momentum
                )
            
            # Cache profile
            self.profile_cache[lead_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting behavioral profile for lead {lead_id}: {e}")
            raise
    
    async def get_leads_by_score_range(
        self,
        min_score: float,
        max_score: float,
        limit: int = 100
    ) -> List[Tuple[str, float]]:
        """Get leads within a specific score range"""
        try:
            # Get all scored leads from Redis
            pattern = "behavioral_score:*"
            keys = await self.redis_client.keys(pattern)
            
            leads_in_range = []
            for key in keys:
                lead_id = key.split(":")[1]
                score_data = await self.redis_client.get(key)
                if score_data:
                    score = float(json.loads(score_data)['current_score'])
                    if min_score <= score <= max_score:
                        leads_in_range.append((lead_id, score))
            
            # Sort by score descending
            leads_in_range.sort(key=lambda x: x[1], reverse=True)
            
            return leads_in_range[:limit]
            
        except Exception as e:
            logger.error(f"Error getting leads by score range: {e}")
            return []
    
    async def get_trending_leads(self, limit: int = 50) -> List[Tuple[str, float, float]]:
        """Get leads with the highest positive score changes"""
        try:
            pattern = "behavioral_score:*"
            keys = await self.redis_client.keys(pattern)
            
            trending_leads = []
            for key in keys:
                lead_id = key.split(":")[1]
                score_data = await self.redis_client.get(key)
                if score_data:
                    data = json.loads(score_data)
                    score_change = data.get('score_change', 0)
                    current_score = data.get('current_score', 0)
                    if score_change > 0:
                        trending_leads.append((lead_id, current_score, score_change))
            
            # Sort by score change descending
            trending_leads.sort(key=lambda x: x[2], reverse=True)
            
            return trending_leads[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending leads: {e}")
            return []
    
    async def get_at_risk_leads(self, limit: int = 50) -> List[Tuple[str, float, float]]:
        """Get leads with high churn risk"""
        try:
            pattern = "behavioral_profile:*"
            keys = await self.redis_client.keys(pattern)
            
            at_risk_leads = []
            for key in keys:
                lead_id = key.split(":")[1]
                profile_data = await self.redis_client.get(key)
                if profile_data:
                    data = json.loads(profile_data)
                    churn_risk = data.get('churn_risk_score', 0)
                    if churn_risk > 0.6:  # High risk threshold
                        current_score = self.score_cache.get(lead_id, 0)
                        at_risk_leads.append((lead_id, current_score, churn_risk))
            
            # Sort by churn risk descending
            at_risk_leads.sort(key=lambda x: x[2], reverse=True)
            
            return at_risk_leads[:limit]
            
        except Exception as e:
            logger.error(f"Error getting at-risk leads: {e}")
            return []
    
    async def _scoring_loop(self):
        """Background task for periodic scoring updates"""
        while self.is_running:
            try:
                # Get all leads that need scoring updates
                await self._update_scores_batch()
                
                # Wait for next update interval
                await asyncio.sleep(self.update_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_scores_batch(self):
        """Update scores for all leads in batch"""
        try:
            # Get all leads with recent activity
            pattern = "lead_events:*"
            keys = await self.redis_client.keys(pattern)
            
            # Process leads in batches
            batch_size = 50
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                tasks = []
                
                for key in batch_keys:
                    lead_id = key.split(":")[1]
                    task = asyncio.create_task(
                        self.calculate_real_time_score(lead_id)
                    )
                    tasks.append(task)
                
                # Wait for batch completion
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            logger.debug(f"Updated scores for {len(keys)} leads")
            
        except Exception as e:
            logger.error(f"Error updating scores in batch: {e}")
    
    async def _calculate_event_scores(self, event: BehavioralEvent):
        """Calculate scores for individual event"""
        # Base score
        base_score = self.scoring_rules.event_base_scores.get(
            event.event_type, 1.0
        )
        
        # Intent score
        intent_score = self.scoring_rules.intent_scores.get(
            event.event_type, 0.1
        )
        
        # Engagement score based on event data
        engagement_score = await self._calculate_event_engagement(event)
        
        # Apply to event
        event.value_score = base_score
        event.intent_score = intent_score
        event.engagement_score = engagement_score
    
    async def _calculate_event_engagement(self, event: BehavioralEvent) -> float:
        """Calculate engagement score for individual event"""
        engagement = 1.0  # Base engagement
        
        # Time-based engagement for certain events
        if event.event_type in [BehaviorType.PAGE_VIEW, BehaviorType.VIDEO_WATCH]:
            duration = event.event_data.get('duration_seconds', 0)
            if duration > 60:  # More than 1 minute
                engagement *= 1.5
            elif duration > 300:  # More than 5 minutes
                engagement *= 2.0
        
        # Interaction depth
        if 'clicks' in event.event_data:
            clicks = event.event_data['clicks']
            engagement *= (1 + clicks * 0.1)
        
        return min(engagement, 3.0)  # Cap at 3x
    
    async def _get_recent_events(
        self,
        lead_id: str,
        days: int
    ) -> List[BehavioralEvent]:
        """Get recent events for a lead from Redis"""
        try:
            # Check in-memory buffer first
            if lead_id in self.event_buffer:
                buffer_events = list(self.event_buffer[lead_id])
                cutoff_time = datetime.now() - timedelta(days=days)
                recent_buffer = [
                    e for e in buffer_events 
                    if e.timestamp >= cutoff_time
                ]
                if recent_buffer:
                    return recent_buffer
            
            # Get from Redis
            key = f"lead_events:{lead_id}"
            events_data = await self.redis_client.lrange(key, 0, -1)
            
            events = []
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for event_json in events_data:
                event_dict = json.loads(event_json)
                event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                event_dict['event_type'] = BehaviorType(event_dict['event_type'])
                
                event = BehavioralEvent(**event_dict)
                if event.timestamp >= cutoff_time:
                    events.append(event)
            
            return sorted(events, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def _store_event(self, event: BehavioralEvent):
        """Store event in Redis"""
        try:
            key = f"lead_events:{event.lead_id}"
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            event_dict['event_type'] = event.event_type.value
            
            # Store event
            await self.redis_client.lpush(key, json.dumps(event_dict))
            
            # Trim to keep only recent events (last 1000)
            await self.redis_client.ltrim(key, 0, 999)
            
            # Set expiration (90 days)
            await self.redis_client.expire(key, 90 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    async def _is_high_value_event(self, event: BehavioralEvent) -> bool:
        """Check if event is high-value and should trigger immediate scoring"""
        high_value_events = {
            BehaviorType.DEMO_REQUESTED,
            BehaviorType.TRIAL_STARTED,
            BehaviorType.MEETING_SCHEDULED,
            BehaviorType.MEETING_ATTENDED,
            BehaviorType.PROPOSAL_VIEWED,
            BehaviorType.PRICING_VIEWED
        }
        
        return event.event_type in high_value_events
    
    async def _calculate_behavioral_score(self, events: List[BehavioralEvent]) -> float:
        """Calculate overall behavioral score"""
        if not events:
            return 0.0
        
        total_score = 0.0
        now = datetime.now()
        
        for event in events:
            # Base score
            score = event.value_score
            
            # Apply recency decay
            hours_ago = (now - event.timestamp).total_seconds() / 3600
            recency_factor = np.exp(-hours_ago / self.scoring_rules.recency_decay_hours)
            score *= (1 - self.scoring_rules.recency_weight) + (self.scoring_rules.recency_weight * recency_factor)
            
            # Apply engagement multiplier
            score *= event.engagement_score
            
            total_score += score
        
        # Normalize by number of events and apply frequency bonus
        avg_score = total_score / len(events)
        if len(events) >= self.scoring_rules.frequency_bonus_threshold:
            avg_score *= self.scoring_rules.frequency_multiplier
        
        return min(avg_score, 100.0)  # Cap at 100
    
    async def _calculate_engagement_score(
        self,
        events: List[BehavioralEvent],
        profile: BehavioralProfile
    ) -> float:
        """Calculate engagement score"""
        if not events:
            return 0.0
        
        # Session-based engagement
        session_engagement = len(set(e.session_id for e in events if e.session_id)) / max(1, profile.total_events)
        
        # Engagement trend from profile
        trend_score = profile.engagement_trend
        
        # Recent engagement (last 7 days)
        recent_events = [
            e for e in events 
            if (datetime.now() - e.timestamp).days <= 7
        ]
        recent_engagement = sum(e.engagement_score for e in recent_events) / max(1, len(recent_events))
        
        # Combine scores
        engagement_score = (
            0.3 * session_engagement * 100 +
            0.4 * trend_score * 100 +
            0.3 * recent_engagement * 100
        )
        
        return min(engagement_score, 100.0)
    
    async def _calculate_intent_score(self, events: List[BehavioralEvent]) -> float:
        """Calculate intent score based on event progression"""
        if not events:
            return 0.0
        
        # Weight recent events more heavily
        weighted_intent = 0.0
        total_weight = 0.0
        now = datetime.now()
        
        for event in events:
            intent = event.intent_score
            hours_ago = (now - event.timestamp).total_seconds() / 3600
            weight = np.exp(-hours_ago / 48)  # 48-hour decay
            
            weighted_intent += intent * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_intent = weighted_intent / total_weight
        else:
            avg_intent = 0.0
        
        return min(avg_intent * 100, 100.0)
    
    async def _calculate_urgency_score(self, events: List[BehavioralEvent]) -> float:
        """Calculate urgency score based on recent activity"""
        if not events:
            return 0.0
        
        now = datetime.now()
        
        # Recent activity (last 24 hours)
        recent_24h = len([e for e in events if (now - e.timestamp).total_seconds() < 24 * 3600])
        
        # Recent activity (last 7 days)
        recent_7d = len([e for e in events if (now - e.timestamp).days < 7])
        
        # High-value recent events
        high_value_recent = len([
            e for e in events 
            if (now - e.timestamp).days < 3 and await self._is_high_value_event(e)
        ])
        
        # Calculate urgency
        urgency = (
            recent_24h * 20 +  # Recent activity worth 20 points each
            recent_7d * 5 +    # Week activity worth 5 points each
            high_value_recent * 40  # High-value events worth 40 points each
        )
        
        return min(urgency, 100.0)
    
    async def _calculate_velocity_score(self, events: List[BehavioralEvent]) -> float:
        """Calculate velocity score (rate of events)"""
        if not events:
            return 0.0
        
        # Calculate events per day over last 7 days
        now = datetime.now()
        recent_events = [
            e for e in events 
            if (now - e.timestamp).days <= 7
        ]
        
        if not recent_events:
            return 0.0
        
        events_per_day = len(recent_events) / 7.0
        
        # Score based on thresholds
        if events_per_day >= self.scoring_rules.velocity_thresholds["high"]:
            velocity_score = 100.0
        elif events_per_day >= self.scoring_rules.velocity_thresholds["medium"]:
            velocity_score = 70.0
        elif events_per_day >= self.scoring_rules.velocity_thresholds["low"]:
            velocity_score = 40.0
        else:
            velocity_score = 20.0
        
        return velocity_score
    
    async def _calculate_momentum_score(self, events: List[BehavioralEvent]) -> float:
        """Calculate momentum score (acceleration of engagement)"""
        if len(events) < 4:
            return 0.0
        
        # Split events into recent and older periods
        now = datetime.now()
        midpoint = now - timedelta(days=7)
        
        recent_events = [e for e in events if e.timestamp > midpoint]
        older_events = [e for e in events if e.timestamp <= midpoint]
        
        if not recent_events or not older_events:
            return 50.0  # Neutral momentum
        
        # Calculate engagement for each period
        recent_engagement = sum(e.engagement_score for e in recent_events) / len(recent_events)
        older_engagement = sum(e.engagement_score for e in older_events) / len(older_events)
        
        # Calculate momentum
        if older_engagement > 0:
            momentum_ratio = recent_engagement / older_engagement
            if momentum_ratio > 1.2:
                momentum_score = 100.0  # Strong positive momentum
            elif momentum_ratio > 1.0:
                momentum_score = 75.0   # Positive momentum
            elif momentum_ratio > 0.8:
                momentum_score = 50.0   # Stable
            else:
                momentum_score = 25.0   # Declining
        else:
            momentum_score = 100.0 if recent_engagement > 0 else 0.0
        
        return momentum_score
    
    def _combine_scores(
        self,
        behavioral: float,
        engagement: float,
        intent: float,
        urgency: float,
        velocity: float,
        momentum: float
    ) -> float:
        """Combine individual scores into overall score"""
        # Weighted combination
        weights = {
            'behavioral': 0.25,
            'engagement': 0.20,
            'intent': 0.20,
            'urgency': 0.15,
            'velocity': 0.10,
            'momentum': 0.10
        }
        
        overall_score = (
            behavioral * weights['behavioral'] +
            engagement * weights['engagement'] +
            intent * weights['intent'] +
            urgency * weights['urgency'] +
            velocity * weights['velocity'] +
            momentum * weights['momentum']
        )
        
        return min(overall_score, 100.0)
    
    async def _identify_trigger_events(self, events: List[BehavioralEvent]) -> List[str]:
        """Identify significant trigger events"""
        triggers = []
        
        now = datetime.now()
        recent_events = [
            e for e in events 
            if (now - e.timestamp).hours <= 24
        ]
        
        for event in recent_events:
            if await self._is_high_value_event(event):
                triggers.append(f"{event.event_type.value}_recent")
        
        # Check for patterns
        if len(recent_events) >= 5:
            triggers.append("high_activity_burst")
        
        return triggers
    
    async def _generate_recommendations(
        self,
        profile: BehavioralProfile,
        current_score: float,
        score_change: float,
        trigger_events: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High score recommendations
        if current_score >= 80:
            recommendations.append("Priority lead - schedule immediate follow-up")
            recommendations.append("Consider expedited sales process")
        
        # Positive momentum recommendations
        if score_change > 10:
            recommendations.append("Strong positive momentum - capitalize immediately")
        
        # Trigger-based recommendations
        if "demo_requested_recent" in trigger_events:
            recommendations.append("Demo requested - ensure quick response")
        
        if "pricing_viewed_recent" in trigger_events:
            recommendations.append("Pricing interest - prepare proposal")
        
        if "high_activity_burst" in trigger_events:
            recommendations.append("High activity detected - engage while hot")
        
        # Behavioral stage recommendations
        if profile.behavioral_stage == "consideration":
            recommendations.append("Provide detailed product information")
        elif profile.behavioral_stage == "decision":
            recommendations.append("Focus on overcoming objections")
        
        return recommendations
    
    async def _store_scoring_result(self, result: RealTimeScoringResult):
        """Store scoring result in Redis"""
        try:
            key = f"behavioral_score:{result.lead_id}"
            result_dict = asdict(result)
            result_dict['scoring_timestamp'] = result.scoring_timestamp.isoformat()
            result_dict['behavioral_profile'] = asdict(result.behavioral_profile)
            
            await self.redis_client.setex(
                key,
                7 * 24 * 3600,  # 7 days expiration
                json.dumps(result_dict)
            )
            
        except Exception as e:
            logger.error(f"Error storing scoring result: {e}")
    
    # Additional helper methods for profile calculations would go here
    async def _calculate_engagement_trend(self, events: List[BehavioralEvent]) -> float:
        """Calculate engagement trend over time"""
        # Implementation would analyze engagement over time periods
        return 0.5  # Placeholder
    
    async def _calculate_intent_progression(self, events: List[BehavioralEvent]) -> float:
        """Calculate intent progression over time"""
        # Implementation would analyze intent development
        return 0.5  # Placeholder
    
    async def _analyze_content_affinity(self, events: List[BehavioralEvent]) -> Dict[str, float]:
        """Analyze content preferences"""
        return {}  # Placeholder
    
    async def _analyze_channel_preference(self, events: List[BehavioralEvent]) -> Dict[str, float]:
        """Analyze channel preferences"""
        return {}  # Placeholder
    
    async def _identify_peak_activity_hours(self, events: List[BehavioralEvent]) -> List[int]:
        """Identify peak activity hours"""
        return []  # Placeholder
    
    async def _determine_behavioral_stage(self, events: List[BehavioralEvent]) -> str:
        """Determine current behavioral stage"""
        return "awareness"  # Placeholder
    
    async def _calculate_churn_risk(self, events: List[BehavioralEvent]) -> float:
        """Calculate churn risk score"""
        return 0.0  # Placeholder
    
    async def _calculate_conversion_momentum(self, events: List[BehavioralEvent]) -> float:
        """Calculate conversion momentum"""
        return 0.0  # Placeholder


# Global instance
real_time_behavioral_engine = RealTimeBehavioralScoringEngine()