"""
Voice AI Communication System
Real-time voice interaction with customers using advanced AI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import json
import base64
import io
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Voice AI configuration"""
    provider: str = "elevenlabs"  # elevenlabs, azure, openai
    voice_id: str = "default"
    language: str = "en-US"
    speed: float = 1.0
    pitch: float = 1.0
    stability: float = 0.75
    clarity: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class VoiceAIEngine:
    """Core Voice AI Engine for customer conversations"""
    
    def __init__(self):
        self.active_calls: Dict[str, Dict] = {}
        self.conversation_templates = self._load_conversation_templates()
        self.voice_models = self._initialize_voice_models()
        
    def _load_conversation_templates(self) -> Dict[str, Dict]:
        """Load conversation templates and scripts"""
        return {
            "introduction": {
                "greeting": "Hello {name}, this is Sarah from {company}. I hope you're having a great day!",
                "purpose": "I'm calling to follow up on your interest in our {product} solution.",
                "permission": "Do you have a few minutes to chat about how we can help {company} achieve {goal}?"
            },
            "discovery": {
                "pain_points": "What challenges is {company} currently facing with {area}?",
                "current_solution": "How are you currently handling {process}?",
                "decision_process": "Who else would be involved in evaluating a solution like ours?",
                "timeline": "What's your ideal timeline for implementing a new solution?"
            },
            "presentation": {
                "value_prop": "Based on what you've shared, I think our {feature} could really help {company} {benefit}.",
                "social_proof": "We've helped similar companies like {similar_company} achieve {result}.",
                "demo_offer": "Would you like to see a quick demo of how this works?"
            },
            "objection_handling": {
                "price": "I understand budget is important. Let's look at the ROI you'd get from {savings}.",
                "timing": "I appreciate you being upfront about timing. What would need to happen for this to become a priority?",
                "authority": "Who else would need to be involved in this decision?",
                "need": "Help me understand what would need to change for this to become valuable to you."
            },
            "closing": {
                "next_steps": "Based on our conversation, I think the next step would be {action}.",
                "calendar": "I'd love to set up a {meeting_type}. What does your calendar look like next week?",
                "follow_up": "I'll send you {materials} and follow up on {date}. Sound good?"
            }
        }
    
    def _initialize_voice_models(self) -> Dict[str, Any]:
        """Initialize different voice models and personalities"""
        return {
            "professional_female": {
                "voice_id": "bella",
                "personality": "Professional, warm, confident",
                "tone": "Conversational but authoritative",
                "speaking_style": "Clear, moderate pace, friendly"
            },
            "professional_male": {
                "voice_id": "adam",
                "personality": "Professional, trustworthy, approachable",
                "tone": "Confident but not pushy",
                "speaking_style": "Clear, slightly deeper voice, reassuring"
            },
            "energetic_sales": {
                "voice_id": "antoni",
                "personality": "Enthusiastic, persuasive, dynamic",
                "tone": "Upbeat and engaging",
                "speaking_style": "Quick pace, expressive, motivating"
            },
            "consultative": {
                "voice_id": "elli",
                "personality": "Thoughtful, analytical, advisory",
                "tone": "Measured and thoughtful",
                "speaking_style": "Slower pace, deliberate, educational"
            }
        }
    
    async def start_call(self, call_id: str, lead_data: Dict, config: VoiceConfig) -> Dict[str, Any]:
        """Start a new voice AI call"""
        try:
            # Initialize call session
            call_session = {
                "call_id": call_id,
                "lead_data": lead_data,
                "config": config,
                "start_time": datetime.utcnow(),
                "conversation_state": "greeting",
                "context": self._build_conversation_context(lead_data),
                "transcript": [],
                "analytics": {
                    "sentiment_scores": [],
                    "engagement_level": 0.5,
                    "objections": [],
                    "interest_signals": []
                }
            }
            
            self.active_calls[call_id] = call_session
            
            # Generate opening message
            opening_message = await self._generate_opening_message(lead_data, config)
            
            # Convert text to speech
            audio_data = await self._text_to_speech(opening_message, config)
            
            # Add to transcript
            call_session["transcript"].append({
                "speaker": "ai",
                "message": opening_message,
                "timestamp": datetime.utcnow(),
                "audio_url": None  # Would be stored in cloud storage
            })
            
            logger.info(f"Started call {call_id} for lead {lead_data.get('id')}")
            
            return {
                "success": True,
                "call_id": call_id,
                "opening_message": opening_message,
                "audio_data": audio_data,
                "session_info": {
                    "state": call_session["conversation_state"],
                    "context": call_session["context"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start call {call_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_customer_response(
        self, 
        call_id: str, 
        audio_data: bytes = None, 
        text: str = None
    ) -> Dict[str, Any]:
        """Process customer response and generate AI reply"""
        
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call session not found"}
        
        call_session = self.active_calls[call_id]
        
        try:
            # Convert speech to text if audio provided
            if audio_data and not text:
                text = await self._speech_to_text(audio_data, call_session["config"])
            
            if not text:
                return {"success": False, "error": "No input provided"}
            
            # Add customer response to transcript
            call_session["transcript"].append({
                "speaker": "customer",
                "message": text,
                "timestamp": datetime.utcnow(),
                "audio_url": None
            })
            
            # Analyze customer response
            analysis = await self._analyze_customer_response(text, call_session)
            
            # Update conversation context
            call_session["context"].update(analysis["context_updates"])
            call_session["analytics"]["sentiment_scores"].append(analysis["sentiment"])
            
            if analysis["objections"]:
                call_session["analytics"]["objections"].extend(analysis["objections"])
            
            if analysis["interest_signals"]:
                call_session["analytics"]["interest_signals"].extend(analysis["interest_signals"])
            
            # Generate AI response
            ai_response = await self._generate_ai_response(text, call_session, analysis)
            
            # Convert to speech
            audio_data = await self._text_to_speech(ai_response, call_session["config"])
            
            # Add AI response to transcript
            call_session["transcript"].append({
                "speaker": "ai",
                "message": ai_response,
                "timestamp": datetime.utcnow(),
                "audio_url": None
            })
            
            # Update conversation state
            call_session["conversation_state"] = analysis.get("next_state", call_session["conversation_state"])
            
            return {
                "success": True,
                "ai_response": ai_response,
                "audio_data": audio_data,
                "analysis": analysis,
                "conversation_state": call_session["conversation_state"],
                "should_continue": analysis.get("should_continue", True)
            }
            
        except Exception as e:
            logger.error(f"Error processing customer response for call {call_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def end_call(self, call_id: str, reason: str = "completed") -> Dict[str, Any]:
        """End a voice AI call and generate summary"""
        
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call session not found"}
        
        call_session = self.active_calls[call_id]
        
        try:
            # Generate call summary
            summary = await self._generate_call_summary(call_session)
            
            # Calculate final analytics
            analytics = await self._calculate_call_analytics(call_session)
            
            # Clean up active call
            del self.active_calls[call_id]
            
            logger.info(f"Ended call {call_id}, reason: {reason}")
            
            return {
                "success": True,
                "call_id": call_id,
                "end_time": datetime.utcnow(),
                "reason": reason,
                "summary": summary,
                "analytics": analytics,
                "transcript": call_session["transcript"]
            }
            
        except Exception as e:
            logger.error(f"Error ending call {call_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_opening_message(self, lead_data: Dict, config: VoiceConfig) -> str:
        """Generate personalized opening message"""
        
        template = self.conversation_templates["introduction"]["greeting"]
        
        # Personalize based on lead data
        name = lead_data.get("first_name", "there")
        company = lead_data.get("company", "your organization")
        
        # Use AI to generate contextual opening
        context = {
            "lead_name": name,
            "lead_company": company,
            "lead_title": lead_data.get("title", ""),
            "lead_industry": lead_data.get("industry", ""),
            "previous_interactions": len(lead_data.get("activities", [])),
            "lead_source": lead_data.get("source", "website")
        }
        
        # This would call OpenAI GPT-4 or similar for dynamic generation
        opening = f"Hi {name}, this is Sarah calling from AI Solutions. I hope I'm catching you at a good time! I wanted to reach out because I noticed {company} has been exploring our platform. How are things going with your current lead management process?"
        
        return opening
    
    async def _analyze_customer_response(self, text: str, call_session: Dict) -> Dict[str, Any]:
        """Analyze customer response for sentiment, intent, and objections"""
        
        # Sentiment analysis (simplified - would use advanced NLP)
        sentiment_score = self._calculate_sentiment(text)
        
        # Intent detection
        intent = self._detect_intent(text)
        
        # Objection detection
        objections = self._detect_objections(text)
        
        # Interest signals
        interest_signals = self._detect_interest_signals(text)
        
        # Context updates
        context_updates = self._extract_context_updates(text)
        
        # Determine next conversation state
        next_state = self._determine_next_state(intent, call_session["conversation_state"])
        
        return {
            "sentiment": sentiment_score,
            "intent": intent,
            "objections": objections,
            "interest_signals": interest_signals,
            "context_updates": context_updates,
            "next_state": next_state,
            "should_continue": sentiment_score > -0.5 and "end_call" not in intent
        }
    
    async def _generate_ai_response(self, customer_input: str, call_session: Dict, analysis: Dict) -> str:
        """Generate contextual AI response"""
        
        conversation_state = call_session["conversation_state"]
        context = call_session["context"]
        
        # Handle objections first
        if analysis["objections"]:
            return await self._handle_objection(analysis["objections"][0], context)
        
        # Generate response based on conversation state
        if conversation_state == "greeting":
            return await self._generate_discovery_question(context)
        elif conversation_state == "discovery":
            return await self._generate_discovery_response(customer_input, context)
        elif conversation_state == "presentation":
            return await self._generate_presentation_response(context)
        elif conversation_state == "closing":
            return await self._generate_closing_response(context)
        else:
            return await self._generate_default_response(customer_input, context)
    
    async def _text_to_speech(self, text: str, config: VoiceConfig) -> bytes:
        """Convert text to speech using configured provider"""
        
        if config.provider == "elevenlabs":
            return await self._elevenlabs_tts(text, config)
        elif config.provider == "azure":
            return await self._azure_tts(text, config)
        elif config.provider == "openai":
            return await self._openai_tts(text, config)
        else:
            # Fallback to basic TTS
            return await self._basic_tts(text)
    
    async def _speech_to_text(self, audio_data: bytes, config: VoiceConfig) -> str:
        """Convert speech to text using Whisper or other STT service"""
        
        try:
            # This would integrate with OpenAI Whisper or Azure Speech Services
            # For now, return a placeholder
            return "Customer response converted from speech"
            
        except Exception as e:
            logger.error(f"Speech to text conversion failed: {e}")
            return ""
    
    async def _elevenlabs_tts(self, text: str, config: VoiceConfig) -> bytes:
        """ElevenLabs text-to-speech conversion"""
        
        # This would integrate with ElevenLabs API
        # For now, return empty bytes as placeholder
        return b""
    
    async def _azure_tts(self, text: str, config: VoiceConfig) -> bytes:
        """Azure Speech Services text-to-speech"""
        
        # This would integrate with Azure Speech Services
        return b""
    
    async def _openai_tts(self, text: str, config: VoiceConfig) -> bytes:
        """OpenAI text-to-speech conversion"""
        
        # This would integrate with OpenAI TTS API
        return b""
    
    async def _basic_tts(self, text: str) -> bytes:
        """Basic TTS fallback"""
        
        # Placeholder for basic TTS
        return b""
    
    def _build_conversation_context(self, lead_data: Dict) -> Dict[str, Any]:
        """Build conversation context from lead data"""
        
        return {
            "lead_name": lead_data.get("first_name", ""),
            "company": lead_data.get("company", ""),
            "title": lead_data.get("title", ""),
            "industry": lead_data.get("industry", ""),
            "pain_points": [],
            "budget_discussed": False,
            "timeline_discussed": False,
            "decision_makers": [],
            "competitor_mentions": [],
            "interest_level": 0.5,
            "objections_raised": [],
            "next_steps_agreed": []
        }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (-1 to 1)"""
        
        # Simplified sentiment analysis
        positive_words = ["great", "excellent", "interested", "yes", "good", "perfect", "love", "like"]
        negative_words = ["no", "not", "bad", "terrible", "expensive", "difficult", "problem", "issue"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _detect_intent(self, text: str) -> str:
        """Detect customer intent from text"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["price", "cost", "budget", "expensive"]):
            return "pricing_inquiry"
        elif any(word in text_lower for word in ["demo", "show", "see", "example"]):
            return "demo_request"
        elif any(word in text_lower for word in ["timeline", "when", "schedule", "implementation"]):
            return "timeline_inquiry"
        elif any(word in text_lower for word in ["decision", "approve", "team", "manager"]):
            return "decision_process"
        elif any(word in text_lower for word in ["not interested", "no thanks", "busy", "later"]):
            return "objection"
        else:
            return "general_inquiry"
    
    def _detect_objections(self, text: str) -> List[str]:
        """Detect objections in customer response"""
        
        objections = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["expensive", "cost", "budget", "price"]):
            objections.append("price")
        
        if any(word in text_lower for word in ["not now", "busy", "timing", "later"]):
            objections.append("timing")
        
        if any(word in text_lower for word in ["not interested", "don't need", "satisfied"]):
            objections.append("need")
        
        if any(word in text_lower for word in ["decision", "manager", "team", "authority"]):
            objections.append("authority")
        
        return objections
    
    def _detect_interest_signals(self, text: str) -> List[str]:
        """Detect positive interest signals"""
        
        signals = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["interested", "tell me more", "sounds good"]):
            signals.append("verbal_interest")
        
        if any(word in text_lower for word in ["demo", "show me", "example", "see"]):
            signals.append("demo_interest")
        
        if any(word in text_lower for word in ["when", "how long", "timeline", "start"]):
            signals.append("timeline_interest")
        
        if any(word in text_lower for word in ["team", "colleagues", "share", "discuss"]):
            signals.append("internal_discussion")
        
        return signals
    
    def _extract_context_updates(self, text: str) -> Dict[str, Any]:
        """Extract context updates from customer response"""
        
        updates = {}
        text_lower = text.lower()
        
        # Extract company size mentions
        if "employees" in text_lower or "people" in text_lower:
            # Extract number (simplified)
            words = text_lower.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    updates["company_size"] = int(word)
                    break
        
        # Extract pain points
        pain_indicators = ["problem", "issue", "challenge", "difficult", "struggle"]
        if any(indicator in text_lower for indicator in pain_indicators):
            updates["pain_points_mentioned"] = True
        
        # Extract budget information
        if any(word in text_lower for word in ["budget", "spend", "investment", "$"]):
            updates["budget_discussed"] = True
        
        return updates
    
    def _determine_next_state(self, intent: str, current_state: str) -> str:
        """Determine next conversation state based on intent"""
        
        state_transitions = {
            "greeting": {
                "general_inquiry": "discovery",
                "pricing_inquiry": "presentation",
                "demo_request": "presentation"
            },
            "discovery": {
                "pricing_inquiry": "presentation",
                "demo_request": "presentation",
                "timeline_inquiry": "closing",
                "general_inquiry": "discovery"
            },
            "presentation": {
                "pricing_inquiry": "presentation",
                "demo_request": "presentation",
                "timeline_inquiry": "closing",
                "objection": "objection_handling"
            },
            "closing": {
                "objection": "objection_handling",
                "general_inquiry": "closing"
            }
        }
        
        return state_transitions.get(current_state, {}).get(intent, current_state)
    
    async def _handle_objection(self, objection: str, context: Dict) -> str:
        """Handle specific objection types"""
        
        objection_responses = {
            "price": f"I understand budget is important for {context.get('company', 'your company')}. Let's talk about the ROI and how this investment typically pays for itself within the first quarter.",
            "timing": "I appreciate you being upfront about timing. What would need to happen for this to become a higher priority?",
            "need": "Help me understand what would need to change for a solution like this to become valuable to you.",
            "authority": "Who else would typically be involved in evaluating a solution like this?"
        }
        
        return objection_responses.get(objection, "I understand your concern. Can you tell me more about that?")
    
    async def _generate_discovery_question(self, context: Dict) -> str:
        """Generate discovery question based on context"""
        
        questions = [
            f"What's the biggest challenge {context.get('company', 'your team')} is facing with lead management right now?",
            "How are you currently tracking and following up with potential customers?",
            "What would an ideal solution look like for your team?",
            "How much time does your team spend on manual lead qualification each week?"
        ]
        
        # Simple selection based on conversation progress
        return questions[0]
    
    async def _generate_discovery_response(self, customer_input: str, context: Dict) -> str:
        """Generate response during discovery phase"""
        
        return "That's really interesting. Many companies we work with face similar challenges. Can you tell me more about how that impacts your team's productivity?"
    
    async def _generate_presentation_response(self, context: Dict) -> str:
        """Generate presentation/value proposition response"""
        
        return f"Based on what you've shared, I think our AI-powered lead scoring could really help {context.get('company', 'your team')} prioritize the highest-value prospects and increase conversion rates. Would you like to see a quick demo of how this works?"
    
    async def _generate_closing_response(self, context: Dict) -> str:
        """Generate closing response"""
        
        return "It sounds like this could be a great fit. What would be the best next step? I'd love to set up a more detailed demo with your team next week."
    
    async def _generate_default_response(self, customer_input: str, context: Dict) -> str:
        """Generate default response"""
        
        return "That's a great point. Can you tell me more about that?"
    
    async def _generate_call_summary(self, call_session: Dict) -> Dict[str, Any]:
        """Generate comprehensive call summary"""
        
        transcript = call_session["transcript"]
        context = call_session["context"]
        analytics = call_session["analytics"]
        
        return {
            "duration_minutes": (datetime.utcnow() - call_session["start_time"]).total_seconds() / 60,
            "total_exchanges": len(transcript),
            "conversation_state": call_session["conversation_state"],
            "key_points": context.get("pain_points", []),
            "objections_raised": analytics["objections"],
            "interest_signals": analytics["interest_signals"],
            "next_steps": context.get("next_steps_agreed", []),
            "overall_sentiment": sum(analytics["sentiment_scores"]) / len(analytics["sentiment_scores"]) if analytics["sentiment_scores"] else 0,
            "recommendation": self._get_follow_up_recommendation(analytics, context)
        }
    
    async def _calculate_call_analytics(self, call_session: Dict) -> Dict[str, Any]:
        """Calculate detailed call analytics"""
        
        transcript = call_session["transcript"]
        analytics = call_session["analytics"]
        
        # Calculate talk time distribution
        ai_turns = [t for t in transcript if t["speaker"] == "ai"]
        customer_turns = [t for t in transcript if t["speaker"] == "customer"]
        
        return {
            "total_turns": len(transcript),
            "ai_turns": len(ai_turns),
            "customer_turns": len(customer_turns),
            "avg_sentiment": sum(analytics["sentiment_scores"]) / len(analytics["sentiment_scores"]) if analytics["sentiment_scores"] else 0,
            "engagement_score": analytics["engagement_level"],
            "objection_count": len(analytics["objections"]),
            "interest_signal_count": len(analytics["interest_signals"]),
            "conversion_probability": self._calculate_conversion_probability(analytics, call_session["context"])
        }
    
    def _get_follow_up_recommendation(self, analytics: Dict, context: Dict) -> str:
        """Get follow-up recommendation based on call performance"""
        
        sentiment = sum(analytics["sentiment_scores"]) / len(analytics["sentiment_scores"]) if analytics["sentiment_scores"] else 0
        interest_signals = len(analytics["interest_signals"])
        objections = len(analytics["objections"])
        
        if sentiment > 0.3 and interest_signals >= 2:
            return "High priority follow-up - Schedule demo within 24 hours"
        elif sentiment > 0 and interest_signals >= 1:
            return "Medium priority - Follow up with additional information in 2-3 days"
        elif objections > 2:
            return "Address objections with targeted content before next contact"
        else:
            return "Add to nurture sequence for long-term follow-up"
    
    def _calculate_conversion_probability(self, analytics: Dict, context: Dict) -> float:
        """Calculate conversion probability based on call performance"""
        
        score = 0.5  # Base score
        
        # Adjust based on sentiment
        avg_sentiment = sum(analytics["sentiment_scores"]) / len(analytics["sentiment_scores"]) if analytics["sentiment_scores"] else 0
        score += avg_sentiment * 0.2
        
        # Adjust based on interest signals
        score += len(analytics["interest_signals"]) * 0.1
        
        # Penalize for objections
        score -= len(analytics["objections"]) * 0.05
        
        # Adjust based on context
        if context.get("budget_discussed"):
            score += 0.1
        if context.get("timeline_discussed"):
            score += 0.1
        if context.get("decision_makers"):
            score += 0.1
        
        return max(0.0, min(1.0, score))


# Global instance
voice_ai_engine = VoiceAIEngine()