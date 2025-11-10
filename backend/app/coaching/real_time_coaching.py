"""
Real-time Sales Coaching System
Live suggestions and guidance during sales calls for immediate improvement
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import re
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class CoachingTrigger(Enum):
    """Types of coaching triggers"""
    SPEECH_PACE = "speech_pace"
    FILLER_WORDS = "filler_words"
    TONE_QUALITY = "tone_quality"
    ENGAGEMENT_LEVEL = "engagement_level"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING_OPPORTUNITY = "closing_opportunity"
    PRODUCT_KNOWLEDGE = "product_knowledge"
    RAPPORT_BUILDING = "rapport_building"
    QUESTION_TECHNIQUE = "question_technique"
    LISTENING_RATIO = "listening_ratio"

class CoachingSeverity(Enum):
    """Severity levels for coaching suggestions"""
    INFO = "info"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    CRITICAL = "critical"

class CoachingCategory(Enum):
    """Categories of coaching feedback"""
    COMMUNICATION = "communication"
    SALES_TECHNIQUE = "sales_technique"
    CUSTOMER_ENGAGEMENT = "customer_engagement"
    PRODUCT_PRESENTATION = "product_presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"

@dataclass
class CoachingSuggestion:
    """Individual coaching suggestion"""
    trigger: CoachingTrigger
    category: CoachingCategory
    severity: CoachingSeverity
    title: str
    message: str
    suggested_action: str
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]
    examples: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "examples": self.examples or []
        }

@dataclass
class CallAnalysisMetrics:
    """Real-time call analysis metrics"""
    speaking_time_ratio: float  # Rep vs customer speaking time
    speech_pace: float  # Words per minute
    filler_word_frequency: float  # Filler words per minute
    tone_energy: float  # Voice energy/enthusiasm level
    question_frequency: float  # Questions per minute
    interruption_count: int  # Number of interruptions
    pause_quality: float  # Quality of pauses (0-1)
    engagement_score: float  # Overall engagement score
    objections_detected: int  # Number of objections from customer
    closing_attempts: int  # Number of closing attempts
    product_mentions: int  # Product feature mentions
    value_propositions: int  # Value propositions presented

@dataclass
class RealTimeCoachingSession:
    """Active real-time coaching session"""
    session_id: str
    sales_rep_id: str
    call_id: str
    start_time: datetime
    current_metrics: CallAnalysisMetrics
    suggestions_history: List[CoachingSuggestion]
    active_suggestions: List[CoachingSuggestion]
    call_stage: str  # opening, discovery, presentation, objection_handling, closing
    customer_sentiment: float  # -1 to 1
    coaching_preferences: Dict[str, Any]

class RealTimeCoachingEngine:
    """Advanced real-time coaching engine for sales calls"""
    
    def __init__(self, config_path: str = "config/coaching_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Active coaching sessions
        self.active_sessions: Dict[str, RealTimeCoachingSession] = {}
        
        # Coaching rules and thresholds
        self.coaching_rules = self._initialize_coaching_rules()
        
        # Speech analysis components
        self.filler_words = {
            "um", "uh", "er", "ah", "like", "you know", "basically", 
            "actually", "literally", "sort of", "kind of"
        }
        
        # Product knowledge database (would be loaded from external source)
        self.product_keywords = {
            "features": ["automation", "integration", "analytics", "dashboard", "reporting"],
            "benefits": ["efficiency", "productivity", "cost-saving", "scalable", "reliable"],
            "competitors": ["salesforce", "hubspot", "pipedrive", "zoho"]
        }
        
        # Objection patterns
        self.objection_patterns = [
            r"(?i)(too expensive|cost too much|budget|price)",
            r"(?i)(think about it|need time|discuss with team)",
            r"(?i)(not interested|no thank you|busy)",
            r"(?i)(already have|using another|current solution)",
            r"(?i)(not sure|uncertain|concerns|worried)"
        ]
        
        # Closing opportunity patterns
        self.closing_patterns = [
            r"(?i)(sounds good|interested|like that|impressive)",
            r"(?i)(how much|pricing|cost|investment)",
            r"(?i)(when can we|next steps|move forward)",
            r"(?i)(decision maker|approved|authorized)"
        ]
        
        logger.info("RealTimeCoachingEngine initialized")
    
    async def start_coaching_session(
        self,
        session_id: str,
        sales_rep_id: str,
        call_id: str,
        coaching_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Start a new real-time coaching session"""
        try:
            # Initialize session
            session = RealTimeCoachingSession(
                session_id=session_id,
                sales_rep_id=sales_rep_id,
                call_id=call_id,
                start_time=datetime.now(),
                current_metrics=CallAnalysisMetrics(
                    speaking_time_ratio=0.0,
                    speech_pace=0.0,
                    filler_word_frequency=0.0,
                    tone_energy=0.0,
                    question_frequency=0.0,
                    interruption_count=0,
                    pause_quality=0.0,
                    engagement_score=0.0,
                    objections_detected=0,
                    closing_attempts=0,
                    product_mentions=0,
                    value_propositions=0
                ),
                suggestions_history=[],
                active_suggestions=[],
                call_stage="opening",
                customer_sentiment=0.0,
                coaching_preferences=coaching_preferences or {}
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Started coaching session {session_id} for rep {sales_rep_id}")
            
            return {
                "session_id": session_id,
                "status": "started",
                "coaching_enabled": True,
                "preferences": session.coaching_preferences
            }
            
        except Exception as e:
            logger.error(f"Error starting coaching session: {e}")
            raise
    
    async def process_speech_chunk(
        self,
        session_id: str,
        speech_text: str,
        speaker: str,  # "rep" or "customer"
        audio_features: Dict[str, Any],
        timestamp: datetime
    ) -> List[CoachingSuggestion]:
        """Process speech chunk and generate real-time coaching"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"No active session found: {session_id}")
                return []
            
            session = self.active_sessions[session_id]
            
            # Update call metrics
            await self._update_call_metrics(session, speech_text, speaker, audio_features, timestamp)
            
            # Analyze speech for coaching opportunities
            suggestions = await self._analyze_speech_for_coaching(
                session, speech_text, speaker, audio_features, timestamp
            )
            
            # Filter and prioritize suggestions
            prioritized_suggestions = self._prioritize_suggestions(suggestions, session)
            
            # Update session with new suggestions
            session.suggestions_history.extend(prioritized_suggestions)
            session.active_suggestions = prioritized_suggestions[-3:]  # Keep last 3 active
            
            logger.debug(f"Generated {len(prioritized_suggestions)} coaching suggestions for session {session_id}")
            
            return prioritized_suggestions
            
        except Exception as e:
            logger.error(f"Error processing speech chunk: {e}")
            return []
    
    async def get_real_time_suggestions(
        self,
        session_id: str,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Get current real-time coaching suggestions"""
        try:
            if session_id not in self.active_sessions:
                return []
            
            session = self.active_sessions[session_id]
            
            suggestions = []
            for suggestion in session.active_suggestions:
                suggestion_dict = suggestion.to_dict()
                
                if include_context:
                    suggestion_dict["session_context"] = {
                        "call_stage": session.call_stage,
                        "customer_sentiment": session.customer_sentiment,
                        "speaking_time_ratio": session.current_metrics.speaking_time_ratio,
                        "engagement_score": session.current_metrics.engagement_score
                    }
                
                suggestions.append(suggestion_dict)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting real-time suggestions: {e}")
            return []
    
    async def update_call_stage(
        self,
        session_id: str,
        new_stage: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """Update the current call stage for contextual coaching"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            old_stage = session.call_stage
            session.call_stage = new_stage
            
            # Generate stage-specific coaching
            stage_suggestions = await self._generate_stage_specific_coaching(
                session, old_stage, new_stage, context or {}
            )
            
            if stage_suggestions:
                session.active_suggestions.extend(stage_suggestions)
                session.suggestions_history.extend(stage_suggestions)
            
            logger.info(f"Updated call stage from {old_stage} to {new_stage} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating call stage: {e}")
            return False
    
    async def end_coaching_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End coaching session and provide summary"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            
            # Generate session summary
            summary = await self._generate_session_summary(session)
            
            # Clean up session
            del self.active_sessions[session_id]
            
            logger.info(f"Ended coaching session {session_id}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error ending coaching session: {e}")
            return {"error": str(e)}
    
    async def _update_call_metrics(
        self,
        session: RealTimeCoachingSession,
        speech_text: str,
        speaker: str,
        audio_features: Dict[str, Any],
        timestamp: datetime
    ):
        """Update real-time call metrics"""
        try:
            metrics = session.current_metrics
            words = speech_text.split()
            word_count = len(words)
            
            # Calculate time since session start
            elapsed_time = (timestamp - session.start_time).total_seconds() / 60.0  # minutes
            
            if elapsed_time > 0:
                # Update speech pace
                if speaker == "rep":
                    current_pace = word_count / (len(speech_text) / 150)  # Estimate speaking time
                    metrics.speech_pace = (metrics.speech_pace + current_pace) / 2
                
                # Update filler word frequency
                filler_count = sum(1 for word in words if word.lower() in self.filler_words)
                if speaker == "rep":
                    metrics.filler_word_frequency = (metrics.filler_word_frequency + filler_count) / 2
                
                # Update tone energy from audio features
                if "energy" in audio_features:
                    metrics.tone_energy = (metrics.tone_energy + audio_features["energy"]) / 2
                
                # Count questions
                question_count = speech_text.count('?')
                if speaker == "rep":
                    metrics.question_frequency = (metrics.question_frequency + question_count) / 2
                
                # Detect objections (customer speech)
                if speaker == "customer":
                    for pattern in self.objection_patterns:
                        if re.search(pattern, speech_text):
                            metrics.objections_detected += 1
                            break
                
                # Count product mentions (rep speech)
                if speaker == "rep":
                    product_count = 0
                    for category, keywords in self.product_keywords.items():
                        product_count += sum(1 for keyword in keywords if keyword.lower() in speech_text.lower())
                    metrics.product_mentions += product_count
                
                # Calculate engagement score (simplified)
                metrics.engagement_score = (
                    (metrics.tone_energy * 0.3) +
                    (min(metrics.question_frequency / 2, 1) * 0.3) +
                    (min(metrics.product_mentions / 5, 1) * 0.4)
                )
                
        except Exception as e:
            logger.error(f"Error updating call metrics: {e}")
    
    async def _analyze_speech_for_coaching(
        self,
        session: RealTimeCoachingSession,
        speech_text: str,
        speaker: str,
        audio_features: Dict[str, Any],
        timestamp: datetime
    ) -> List[CoachingSuggestion]:
        """Analyze speech and generate coaching suggestions"""
        suggestions = []
        
        try:
            if speaker == "rep":
                # Analyze rep speech for coaching opportunities
                
                # Check speech pace
                words = speech_text.split()
                if len(words) > 0:
                    estimated_duration = len(speech_text) / 150  # Rough estimate
                    pace = len(words) / estimated_duration if estimated_duration > 0 else 0
                    
                    if pace > 200:  # Too fast
                        suggestions.append(CoachingSuggestion(
                            trigger=CoachingTrigger.SPEECH_PACE,
                            category=CoachingCategory.COMMUNICATION,
                            severity=CoachingSeverity.SUGGESTION,
                            title="Speech Pace",
                            message="You're speaking quite quickly. Consider slowing down for better comprehension.",
                            suggested_action="Take a breath and speak more slowly to ensure the customer follows along.",
                            confidence=0.8,
                            timestamp=timestamp,
                            context={"current_pace": pace, "optimal_pace": "150-180 wpm"},
                            examples=["Try: 'Let me slow down and explain this clearly...'"]
                        ))
                    elif pace < 120:  # Too slow
                        suggestions.append(CoachingSuggestion(
                            trigger=CoachingTrigger.SPEECH_PACE,
                            category=CoachingCategory.COMMUNICATION,
                            severity=CoachingSeverity.INFO,
                            title="Speech Pace",
                            message="Your pace seems a bit slow. Consider speaking with more energy.",
                            suggested_action="Increase your speaking pace slightly to maintain engagement.",
                            confidence=0.7,
                            timestamp=timestamp,
                            context={"current_pace": pace, "optimal_pace": "150-180 wpm"}
                        ))
                
                # Check for filler words
                filler_count = sum(1 for word in words if word.lower() in self.filler_words)
                if filler_count > 2 and len(words) > 10:  # High filler word ratio
                    suggestions.append(CoachingSuggestion(
                        trigger=CoachingTrigger.FILLER_WORDS,
                        category=CoachingCategory.COMMUNICATION,
                        severity=CoachingSeverity.WARNING,
                        title="Filler Words",
                        message=f"You used {filler_count} filler words. Try to pause instead of using fillers.",
                        suggested_action="Replace filler words with brief pauses for more professional delivery.",
                        confidence=0.9,
                        timestamp=timestamp,
                        context={"filler_count": filler_count, "total_words": len(words)},
                        examples=["Instead of 'um, like', try a brief pause"]
                    ))
                
                # Check for questions (important in sales)
                question_count = speech_text.count('?')
                if session.call_stage in ["discovery", "presentation"] and question_count == 0 and len(words) > 20:
                    suggestions.append(CoachingSuggestion(
                        trigger=CoachingTrigger.QUESTION_TECHNIQUE,
                        category=CoachingCategory.SALES_TECHNIQUE,
                        severity=CoachingSeverity.SUGGESTION,
                        title="Ask More Questions",
                        message="Consider asking questions to better understand customer needs.",
                        suggested_action="Ask open-ended questions to engage the customer and gather information.",
                        confidence=0.8,
                        timestamp=timestamp,
                        context={"call_stage": session.call_stage},
                        examples=["What challenges are you currently facing?", "How would this impact your workflow?"]
                    ))
                
                # Check for closing opportunities
                if session.call_stage in ["presentation", "objection_handling"]:
                    for pattern in self.closing_patterns:
                        if re.search(pattern, speech_text.lower()):
                            suggestions.append(CoachingSuggestion(
                                trigger=CoachingTrigger.CLOSING_OPPORTUNITY,
                                category=CoachingCategory.CLOSING,
                                severity=CoachingSeverity.CRITICAL,
                                title="Closing Opportunity",
                                message="The customer is showing buying signals. Consider moving toward a close.",
                                suggested_action="Ask for the sale or suggest next steps while momentum is positive.",
                                confidence=0.9,
                                timestamp=timestamp,
                                context={"customer_signal": "positive engagement"},
                                examples=["Would you like to move forward with this?", "What would you need to get started?"]
                            ))
                            break
                
                # Check product knowledge usage
                product_mentions = 0
                for category, keywords in self.product_keywords.items():
                    product_mentions += sum(1 for keyword in keywords if keyword.lower() in speech_text.lower())
                
                if session.call_stage == "presentation" and product_mentions == 0 and len(words) > 15:
                    suggestions.append(CoachingSuggestion(
                        trigger=CoachingTrigger.PRODUCT_KNOWLEDGE,
                        category=CoachingCategory.PRODUCT_PRESENTATION,
                        severity=CoachingSeverity.SUGGESTION,
                        title="Product Focus",
                        message="Consider mentioning specific product features or benefits.",
                        suggested_action="Highlight relevant features that address the customer's needs.",
                        confidence=0.7,
                        timestamp=timestamp,
                        context={"call_stage": session.call_stage},
                        examples=["Our automation feature can save you 10 hours per week"]
                    ))
                
                # Check tone energy
                if "energy" in audio_features and audio_features["energy"] < 0.3:
                    suggestions.append(CoachingSuggestion(
                        trigger=CoachingTrigger.TONE_QUALITY,
                        category=CoachingCategory.COMMUNICATION,
                        severity=CoachingSeverity.INFO,
                        title="Energy Level",
                        message="Your energy level seems low. Consider speaking with more enthusiasm.",
                        suggested_action="Inject more energy and enthusiasm into your voice.",
                        confidence=0.6,
                        timestamp=timestamp,
                        context={"energy_level": audio_features["energy"]}
                    ))
            
            elif speaker == "customer":
                # Analyze customer speech for objections and opportunities
                
                # Detect objections
                for pattern in self.objection_patterns:
                    if re.search(pattern, speech_text):
                        suggestions.append(CoachingSuggestion(
                            trigger=CoachingTrigger.OBJECTION_HANDLING,
                            category=CoachingCategory.OBJECTION_HANDLING,
                            severity=CoachingSeverity.WARNING,
                            title="Objection Detected",
                            message="Customer raised an objection. Address it directly and empathetically.",
                            suggested_action="Acknowledge the concern, ask clarifying questions, and provide solutions.",
                            confidence=0.85,
                            timestamp=timestamp,
                            context={"objection_text": speech_text[:100]},
                            examples=["I understand your concern about cost. Let me show you the ROI..."]
                        ))
                        break
                
                # Detect positive signals
                positive_indicators = ["great", "excellent", "perfect", "exactly", "love", "impressive"]
                if any(indicator in speech_text.lower() for indicator in positive_indicators):
                    suggestions.append(CoachingSuggestion(
                        trigger=CoachingTrigger.CLOSING_OPPORTUNITY,
                        category=CoachingCategory.CLOSING,
                        severity=CoachingSeverity.CRITICAL,
                        title="Positive Signal",
                        message="Customer is expressing positive sentiment. Great opportunity to advance the sale.",
                        suggested_action="Capitalize on this positive momentum and move toward next steps.",
                        confidence=0.8,
                        timestamp=timestamp,
                        context={"positive_signal": speech_text[:100]},
                        examples=["Since you like this approach, shall we discuss implementation?"]
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing speech for coaching: {e}")
        
        return suggestions
    
    async def _generate_stage_specific_coaching(
        self,
        session: RealTimeCoachingSession,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any]
    ) -> List[CoachingSuggestion]:
        """Generate coaching suggestions based on call stage transition"""
        suggestions = []
        
        try:
            stage_coaching = {
                "opening": {
                    "title": "Call Opening",
                    "message": "Focus on building rapport and setting agenda.",
                    "action": "Greet warmly, confirm their time, and outline what you'll cover.",
                    "examples": ["Thanks for taking the time to meet. I have 3 key points to share..."]
                },
                "discovery": {
                    "title": "Discovery Phase",
                    "message": "Ask open-ended questions to understand their needs.",
                    "action": "Focus on listening and asking follow-up questions.",
                    "examples": ["Tell me about your current process...", "What's working well and what isn't?"]
                },
                "presentation": {
                    "title": "Presentation Phase",
                    "message": "Present solutions that directly address their stated needs.",
                    "action": "Connect each feature to their specific challenges.",
                    "examples": ["Based on what you mentioned about X, our Y feature will..."]
                },
                "objection_handling": {
                    "title": "Handling Objections",
                    "message": "Listen carefully, acknowledge concerns, and provide solutions.",
                    "action": "Use the feel-felt-found technique.",
                    "examples": ["I understand how you feel. Others felt the same, but found that..."]
                },
                "closing": {
                    "title": "Closing the Sale",
                    "message": "Ask for the sale or define clear next steps.",
                    "action": "Be direct and confident in asking for commitment.",
                    "examples": ["Are you ready to move forward?", "What do we need to do to get started?"]
                }
            }
            
            if new_stage in stage_coaching:
                coaching = stage_coaching[new_stage]
                suggestions.append(CoachingSuggestion(
                    trigger=CoachingTrigger.ENGAGEMENT_LEVEL,
                    category=CoachingCategory.SALES_TECHNIQUE,
                    severity=CoachingSeverity.INFO,
                    title=coaching["title"],
                    message=coaching["message"],
                    suggested_action=coaching["action"],
                    confidence=0.9,
                    timestamp=datetime.now(),
                    context={"stage_transition": f"{old_stage} -> {new_stage}"},
                    examples=coaching["examples"]
                ))
            
        except Exception as e:
            logger.error(f"Error generating stage-specific coaching: {e}")
        
        return suggestions
    
    def _prioritize_suggestions(
        self,
        suggestions: List[CoachingSuggestion],
        session: RealTimeCoachingSession
    ) -> List[CoachingSuggestion]:
        """Prioritize coaching suggestions based on severity and context"""
        try:
            # Sort by severity and confidence
            severity_order = {
                CoachingSeverity.CRITICAL: 4,
                CoachingSeverity.WARNING: 3,
                CoachingSeverity.SUGGESTION: 2,
                CoachingSeverity.INFO: 1
            }
            
            # Filter out duplicate suggestions (same trigger within short timeframe)
            unique_suggestions = []
            recent_triggers = {s.trigger for s in session.active_suggestions}
            
            for suggestion in suggestions:
                if (suggestion.trigger not in recent_triggers or 
                    suggestion.severity == CoachingSeverity.CRITICAL):
                    unique_suggestions.append(suggestion)
            
            # Sort by priority
            prioritized = sorted(
                unique_suggestions,
                key=lambda s: (severity_order[s.severity], s.confidence),
                reverse=True
            )
            
            # Limit to top 5 suggestions to avoid overwhelming
            return prioritized[:5]
            
        except Exception as e:
            logger.error(f"Error prioritizing suggestions: {e}")
            return suggestions[:5]  # Fallback to first 5
    
    async def _generate_session_summary(
        self,
        session: RealTimeCoachingSession
    ) -> Dict[str, Any]:
        """Generate coaching session summary"""
        try:
            duration = (datetime.now() - session.start_time).total_seconds() / 60.0
            
            # Categorize suggestions
            suggestion_categories = defaultdict(int)
            for suggestion in session.suggestions_history:
                suggestion_categories[suggestion.category.value] += 1
            
            # Calculate improvement opportunities
            top_categories = sorted(
                suggestion_categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            return {
                "session_id": session.session_id,
                "sales_rep_id": session.sales_rep_id,
                "call_id": session.call_id,
                "duration_minutes": duration,
                "total_suggestions": len(session.suggestions_history),
                "final_metrics": asdict(session.current_metrics),
                "top_improvement_areas": [
                    {"category": cat, "suggestion_count": count}
                    for cat, count in top_categories
                ],
                "call_stages_covered": session.call_stage,
                "customer_sentiment": session.customer_sentiment,
                "coaching_effectiveness": {
                    "engagement_improvement": max(0, session.current_metrics.engagement_score - 0.5),
                    "communication_quality": 1.0 - (session.current_metrics.filler_word_frequency / 10),
                    "sales_technique_usage": min(1.0, session.current_metrics.question_frequency / 5)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {"error": str(e)}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load coaching configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        # Default configuration
        return {
            "coaching_sensitivity": "medium",
            "max_active_suggestions": 3,
            "suggestion_timeout_minutes": 5,
            "enable_stage_coaching": True,
            "enable_real_time_metrics": True
        }
    
    def _initialize_coaching_rules(self) -> Dict[str, Any]:
        """Initialize coaching rules and thresholds"""
        return {
            "speech_pace": {
                "optimal_range": (150, 180),  # words per minute
                "warning_threshold": 200
            },
            "filler_words": {
                "max_per_sentence": 1,
                "warning_threshold": 3
            },
            "question_frequency": {
                "discovery_min": 2,  # per 5 minutes
                "presentation_min": 1
            },
            "tone_energy": {
                "minimum_threshold": 0.3,
                "optimal_range": (0.5, 0.8)
            },
            "speaking_ratio": {
                "discovery_max": 0.3,  # rep should speak less during discovery
                "presentation_max": 0.7  # rep can speak more during presentation
            }
        }


# Global instance
real_time_coaching_engine = RealTimeCoachingEngine()