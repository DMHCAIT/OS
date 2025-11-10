"""
Advanced Multi-Language Conversation Intelligence
Supports global sales teams with automatic language detection, translation, and localized conversation handling
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

# Language detection and translation
try:
    from langdetect import detect, detect_langs
    from googletrans import Translator
    import polyglot
    from polyglot.detect import Detector
    from polyglot.text import Text
except ImportError:
    pass

# Advanced NLP for multiple languages
try:
    import spacy
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages for conversation intelligence"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    TURKISH = "tr"
    POLISH = "pl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"

@dataclass
class LanguageDetectionResult:
    """Language detection analysis result"""
    detected_language: str
    confidence: float
    alternative_languages: List[Tuple[str, float]]
    is_supported: bool
    needs_translation: bool

@dataclass
class TranslationResult:
    """Translation result with metadata"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    translation_confidence: float
    translation_method: str
    cultural_notes: Optional[str] = None

@dataclass
class LocalizedResponse:
    """Culturally and linguistically appropriate response"""
    response_text: str
    language: str
    cultural_context: str
    tone_adaptation: str
    local_references: List[str]
    formality_level: str

@dataclass
class ConversationContext:
    """Enhanced conversation context with language intelligence"""
    conversation_id: str
    participant_language: str
    detected_languages: List[str]
    primary_language: str
    cultural_background: Optional[str]
    language_proficiency: Optional[str]
    communication_style: str
    regional_preferences: Dict[str, Any]

class LanguageDetectionEngine:
    """Advanced language detection with multiple methods"""
    
    def __init__(self):
        self.translator = None
        self.detection_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize language detection models"""
        try:
            self.translator = Translator()
            
            # Load language-specific models
            for lang in SupportedLanguage:
                try:
                    # Load spaCy models for supported languages
                    if lang.value in ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja']:
                        model_name = self._get_spacy_model_name(lang.value)
                        if model_name:
                            self.detection_models[lang.value] = spacy.load(model_name)
                except Exception as e:
                    logger.warning(f"Could not load spaCy model for {lang.value}: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing language models: {e}")
    
    def _get_spacy_model_name(self, lang_code: str) -> Optional[str]:
        """Get spaCy model name for language"""
        model_map = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ru': 'ru_core_news_sm',
            'zh': 'zh_core_web_sm',
            'ja': 'ja_core_news_sm'
        }
        return model_map.get(lang_code)
    
    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """Advanced language detection with multiple methods"""
        try:
            # Method 1: langdetect library
            detected_lang = detect(text)
            detected_langs = detect_langs(text)
            
            # Method 2: Polyglot (if available)
            polyglot_result = None
            try:
                detector = Detector(text)
                polyglot_result = detector.language
            except:
                pass
            
            # Combine results
            primary_detection = detected_lang
            confidence = max([lang.prob for lang in detected_langs if lang.lang == primary_detection], default=0.0)
            
            # Alternative languages
            alternatives = [(lang.lang, lang.prob) for lang in detected_langs if lang.lang != primary_detection]
            
            # Check if language is supported
            is_supported = any(lang.value == primary_detection for lang in SupportedLanguage)
            
            # Determine if translation is needed
            needs_translation = primary_detection != 'en' and is_supported
            
            return LanguageDetectionResult(
                detected_language=primary_detection,
                confidence=confidence,
                alternative_languages=alternatives,
                is_supported=is_supported,
                needs_translation=needs_translation
            )
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            # Default to English
            return LanguageDetectionResult(
                detected_language='en',
                confidence=0.5,
                alternative_languages=[],
                is_supported=True,
                needs_translation=False
            )
    
    async def translate_text(self, text: str, target_language: str = 'en', source_language: str = 'auto') -> TranslationResult:
        """Translate text with quality assessment"""
        try:
            if not self.translator:
                raise Exception("Translator not initialized")
            
            # Perform translation
            translation = self.translator.translate(text, dest=target_language, src=source_language)
            
            # Assess translation quality
            confidence = self._assess_translation_quality(text, translation.text, source_language, target_language)
            
            # Get cultural notes
            cultural_notes = await self._get_cultural_notes(text, source_language, target_language)
            
            return TranslationResult(
                original_text=text,
                translated_text=translation.text,
                source_language=translation.src,
                target_language=target_language,
                translation_confidence=confidence,
                translation_method="Google Translate",
                cultural_notes=cultural_notes
            )
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Fallback to original
                source_language=source_language,
                target_language=target_language,
                translation_confidence=0.0,
                translation_method="fallback",
                cultural_notes="Translation failed"
            )
    
    def _assess_translation_quality(self, original: str, translated: str, src_lang: str, tgt_lang: str) -> float:
        """Assess translation quality using heuristics"""
        try:
            # Length ratio check
            length_ratio = len(translated) / len(original) if len(original) > 0 else 0
            length_score = 1.0 if 0.5 <= length_ratio <= 2.0 else 0.5
            
            # Word count ratio
            orig_words = len(original.split())
            trans_words = len(translated.split())
            word_ratio = trans_words / orig_words if orig_words > 0 else 0
            word_score = 1.0 if 0.6 <= word_ratio <= 1.5 else 0.7
            
            # Character diversity (avoid repetitive translations)
            unique_chars = len(set(translated.lower()))
            total_chars = len(translated)
            diversity_score = unique_chars / total_chars if total_chars > 0 else 0
            
            # Combined score
            overall_score = (length_score * 0.4 + word_score * 0.4 + diversity_score * 0.2)
            
            return min(overall_score, 1.0)
            
        except Exception:
            return 0.7  # Default moderate confidence
    
    async def _get_cultural_notes(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """Generate cultural context notes for translation"""
        cultural_patterns = {
            'formal_address': {
                'pattern': r'\b(sir|madam|mister|miss|mrs|mr)\b',
                'note': 'Formal address detected - may require cultural adaptation'
            },
            'business_terms': {
                'pattern': r'\b(deal|contract|negotiate|proposal|agreement)\b',
                'note': 'Business terminology - ensure professional context maintained'
            },
            'time_references': {
                'pattern': r'\b(today|tomorrow|yesterday|morning|afternoon|evening)\b',
                'note': 'Time references - consider timezone differences'
            }
        }
        
        notes = []
        for category, pattern_info in cultural_patterns.items():
            if re.search(pattern_info['pattern'], text.lower()):
                notes.append(pattern_info['note'])
        
        return "; ".join(notes) if notes else None


class CulturalAdaptationEngine:
    """Adapts conversation style for different cultures"""
    
    def __init__(self):
        self.cultural_profiles = self._load_cultural_profiles()
        self.communication_styles = self._load_communication_styles()
    
    def _load_cultural_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural communication profiles"""
        return {
            'en-US': {
                'formality': 'moderate',
                'directness': 'high',
                'small_talk': 'moderate',
                'decision_speed': 'fast',
                'hierarchy_respect': 'low',
                'personal_space': 'high'
            },
            'en-GB': {
                'formality': 'high',
                'directness': 'moderate',
                'small_talk': 'high',
                'decision_speed': 'moderate',
                'hierarchy_respect': 'moderate',
                'personal_space': 'high'
            },
            'es': {
                'formality': 'high',
                'directness': 'moderate',
                'small_talk': 'high',
                'decision_speed': 'moderate',
                'hierarchy_respect': 'high',
                'personal_space': 'low'
            },
            'de': {
                'formality': 'very_high',
                'directness': 'very_high',
                'small_talk': 'low',
                'decision_speed': 'slow',
                'hierarchy_respect': 'high',
                'personal_space': 'high'
            },
            'fr': {
                'formality': 'high',
                'directness': 'moderate',
                'small_talk': 'moderate',
                'decision_speed': 'moderate',
                'hierarchy_respect': 'moderate',
                'personal_space': 'moderate'
            },
            'ja': {
                'formality': 'very_high',
                'directness': 'low',
                'small_talk': 'moderate',
                'decision_speed': 'slow',
                'hierarchy_respect': 'very_high',
                'personal_space': 'moderate'
            },
            'zh': {
                'formality': 'high',
                'directness': 'low',
                'small_talk': 'moderate',
                'decision_speed': 'moderate',
                'hierarchy_respect': 'very_high',
                'personal_space': 'low'
            },
            'ar': {
                'formality': 'high',
                'directness': 'moderate',
                'small_talk': 'high',
                'decision_speed': 'slow',
                'hierarchy_respect': 'high',
                'personal_space': 'low'
            }
        }
    
    def _load_communication_styles(self) -> Dict[str, Dict[str, Any]]:
        """Load communication style templates"""
        return {
            'formal_opening': {
                'en': "Good {time_of_day}, {title} {name}. I hope this message finds you well.",
                'es': "Buenos {time_of_day}, {title} {name}. Espero que se encuentre bien.",
                'fr': "Bon{time_of_day}, {title} {name}. J'espère que vous allez bien.",
                'de': "Guten {time_of_day}, {title} {name}. Ich hoffe, es geht Ihnen gut.",
                'ja': "{title}{name}様、いつもお世話になっております。",
                'zh': "{title}{name}，您好！希望您一切都好。"
            },
            'value_proposition': {
                'en': "I'd like to discuss how we can help improve your {business_area}.",
                'es': "Me gustaría hablar sobre cómo podemos ayudar a mejorar su {business_area}.",
                'fr': "J'aimerais discuter de la façon dont nous pouvons améliorer votre {business_area}.",
                'de': "Ich möchte gerne besprechen, wie wir Ihren {business_area} verbessern können.",
                'ja': "お客様の{business_area}の改善についてご相談させていただきたく。",
                'zh': "我想讨论一下我们如何帮助改善您的{business_area}。"
            },
            'closing': {
                'en': "Thank you for your time. I look forward to hearing from you.",
                'es': "Gracias por su tiempo. Espero tener noticias suyas pronto.",
                'fr': "Merci pour votre temps. J'ai hâte de vous entendre.",
                'de': "Vielen Dank für Ihre Zeit. Ich freue mich auf Ihre Antwort.",
                'ja': "お忙しい中、お時間をいただきありがとうございました。ご連絡をお待ちしております。",
                'zh': "感谢您的时间。期待您的回复。"
            }
        }
    
    async def adapt_response(self, response: str, target_language: str, cultural_context: Optional[str] = None) -> LocalizedResponse:
        """Adapt response for cultural and linguistic context"""
        try:
            # Get cultural profile
            cultural_profile = self.cultural_profiles.get(target_language, self.cultural_profiles.get('en-US'))
            
            # Adapt formality
            adapted_response = await self._adapt_formality(response, cultural_profile['formality'], target_language)
            
            # Adapt directness
            adapted_response = await self._adapt_directness(adapted_response, cultural_profile['directness'])
            
            # Add cultural references if appropriate
            local_references = await self._get_local_references(target_language)
            
            # Determine tone adaptation
            tone = await self._determine_tone(cultural_profile)
            
            return LocalizedResponse(
                response_text=adapted_response,
                language=target_language,
                cultural_context=cultural_context or 'general',
                tone_adaptation=tone,
                local_references=local_references,
                formality_level=cultural_profile['formality']
            )
            
        except Exception as e:
            logger.error(f"Error in cultural adaptation: {e}")
            return LocalizedResponse(
                response_text=response,
                language=target_language,
                cultural_context='fallback',
                tone_adaptation='neutral',
                local_references=[],
                formality_level='moderate'
            )
    
    async def _adapt_formality(self, text: str, formality_level: str, language: str) -> str:
        """Adapt text formality level"""
        formality_patterns = {
            'very_high': {
                'replacements': {
                    'hi': 'good day',
                    'thanks': 'thank you very much',
                    'ok': 'certainly',
                    'sure': 'absolutely'
                }
            },
            'high': {
                'replacements': {
                    'hi': 'hello',
                    'thanks': 'thank you',
                    'ok': 'very well',
                    'sure': 'of course'
                }
            },
            'moderate': {
                'replacements': {
                    'hey': 'hello',
                    'thx': 'thanks'
                }
            }
        }
        
        if formality_level in formality_patterns:
            replacements = formality_patterns[formality_level]['replacements']
            adapted_text = text
            for informal, formal in replacements.items():
                adapted_text = re.sub(r'\b' + informal + r'\b', formal, adapted_text, flags=re.IGNORECASE)
            return adapted_text
        
        return text
    
    async def _adapt_directness(self, text: str, directness_level: str) -> str:
        """Adapt directness of communication"""
        if directness_level == 'low':
            # Add softening phrases
            softening_phrases = [
                "Perhaps you might consider",
                "It might be beneficial to",
                "You may find it helpful to",
                "If I may suggest"
            ]
            
            # Look for direct statements and soften them
            direct_patterns = [
                (r'^You should', 'You might want to'),
                (r'^You need to', 'It would be helpful if you could'),
                (r'^Do this', 'Please consider doing this'),
            ]
            
            adapted_text = text
            for pattern, replacement in direct_patterns:
                adapted_text = re.sub(pattern, replacement, adapted_text)
            
            return adapted_text
            
        elif directness_level == 'very_high':
            # Make statements more direct
            indirect_patterns = [
                (r'Perhaps you might', 'You should'),
                (r'You may want to consider', 'You need to'),
                (r'It might be beneficial', 'It is beneficial'),
            ]
            
            adapted_text = text
            for pattern, replacement in indirect_patterns:
                adapted_text = re.sub(pattern, replacement, adapted_text)
            
            return adapted_text
        
        return text
    
    async def _get_local_references(self, language: str) -> List[str]:
        """Get culturally relevant references"""
        references = {
            'en-US': ['local business practices', 'market conditions', 'regional preferences'],
            'en-GB': ['British market standards', 'UK regulations', 'local customs'],
            'de': ['German efficiency standards', 'GDPR compliance', 'European markets'],
            'fr': ['French business etiquette', 'European integration', 'luxury standards'],
            'ja': ['Japanese quality standards', 'long-term relationships', 'consensus building'],
            'zh': ['Chinese market dynamics', 'relationship building', 'harmonious cooperation'],
            'es': ['Hispanic market growth', 'relationship-focused approach', 'family business values'],
            'ar': ['Middle Eastern partnerships', 'trust-based relationships', 'regional expertise']
        }
        return references.get(language, ['global best practices', 'international standards'])
    
    async def _determine_tone(self, cultural_profile: Dict[str, Any]) -> str:
        """Determine appropriate tone based on cultural profile"""
        formality = cultural_profile.get('formality', 'moderate')
        directness = cultural_profile.get('directness', 'moderate')
        
        if formality == 'very_high' and directness == 'low':
            return 'respectful_indirect'
        elif formality == 'high' and directness == 'high':
            return 'professional_direct'
        elif formality == 'low' and directness == 'high':
            return 'casual_straightforward'
        else:
            return 'balanced_professional'


class MultiLanguageConversationIntelligence:
    """Main multi-language conversation intelligence system"""
    
    def __init__(self):
        self.language_detector = LanguageDetectionEngine()
        self.cultural_adapter = CulturalAdaptationEngine()
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.translation_cache: Dict[str, TranslationResult] = {}
    
    async def initialize(self):
        """Initialize the multi-language system"""
        logger.info("Initializing Multi-Language Conversation Intelligence...")
        # Add any additional initialization here
        logger.info("Multi-Language Conversation Intelligence initialized successfully")
    
    async def process_multilingual_message(self, 
                                         conversation_id: str,
                                         message: str,
                                         participant_id: str) -> Dict[str, Any]:
        """Process a message with multi-language intelligence"""
        try:
            # Detect language
            detection_result = await self.language_detector.detect_language(message)
            
            # Get or create conversation context
            context = await self._get_conversation_context(conversation_id, detection_result.detected_language)
            
            # Translate if needed
            translation_result = None
            processed_message = message
            
            if detection_result.needs_translation:
                translation_result = await self.language_detector.translate_text(
                    message, 
                    target_language='en',
                    source_language=detection_result.detected_language
                )
                processed_message = translation_result.translated_text
            
            # Update conversation context
            await self._update_conversation_context(context, detection_result, translation_result)
            
            return {
                'conversation_id': conversation_id,
                'original_message': message,
                'processed_message': processed_message,
                'language_detection': asdict(detection_result),
                'translation': asdict(translation_result) if translation_result else None,
                'conversation_context': asdict(context),
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing multilingual message: {e}")
            return {
                'conversation_id': conversation_id,
                'original_message': message,
                'processed_message': message,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    async def generate_multilingual_response(self,
                                           conversation_id: str,
                                           response_text: str,
                                           target_language: Optional[str] = None) -> LocalizedResponse:
        """Generate culturally and linguistically appropriate response"""
        try:
            # Get conversation context
            context = self.conversation_contexts.get(conversation_id)
            if not context:
                # Default context
                context = ConversationContext(
                    conversation_id=conversation_id,
                    participant_language='en',
                    detected_languages=['en'],
                    primary_language='en',
                    cultural_background=None,
                    language_proficiency=None,
                    communication_style='professional',
                    regional_preferences={}
                )
            
            # Determine target language
            if not target_language:
                target_language = context.participant_language
            
            # Translate response if needed
            translated_response = response_text
            if target_language != 'en':
                translation = await self.language_detector.translate_text(
                    response_text,
                    target_language=target_language,
                    source_language='en'
                )
                translated_response = translation.translated_text
            
            # Apply cultural adaptation
            localized_response = await self.cultural_adapter.adapt_response(
                translated_response,
                target_language,
                context.cultural_background
            )
            
            return localized_response
            
        except Exception as e:
            logger.error(f"Error generating multilingual response: {e}")
            return LocalizedResponse(
                response_text=response_text,
                language=target_language or 'en',
                cultural_context='error_fallback',
                tone_adaptation='neutral',
                local_references=[],
                formality_level='moderate'
            )
    
    async def _get_conversation_context(self, 
                                      conversation_id: str, 
                                      detected_language: str) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id in self.conversation_contexts:
            context = self.conversation_contexts[conversation_id]
            # Update detected languages
            if detected_language not in context.detected_languages:
                context.detected_languages.append(detected_language)
            return context
        
        # Create new context
        context = ConversationContext(
            conversation_id=conversation_id,
            participant_language=detected_language,
            detected_languages=[detected_language],
            primary_language=detected_language,
            cultural_background=await self._infer_cultural_background(detected_language),
            language_proficiency=None,
            communication_style='professional',
            regional_preferences={}
        )
        
        self.conversation_contexts[conversation_id] = context
        return context
    
    async def _update_conversation_context(self,
                                         context: ConversationContext,
                                         detection_result: LanguageDetectionResult,
                                         translation_result: Optional[TranslationResult]):
        """Update conversation context with new information"""
        # Update language confidence and preferences
        if detection_result.confidence > 0.8:
            context.primary_language = detection_result.detected_language
        
        # Infer language proficiency from translation quality
        if translation_result and translation_result.translation_confidence < 0.7:
            context.language_proficiency = 'intermediate'
        elif translation_result and translation_result.translation_confidence > 0.9:
            context.language_proficiency = 'native'
        
        # Update cultural background
        if not context.cultural_background:
            context.cultural_background = await self._infer_cultural_background(detection_result.detected_language)
    
    async def _infer_cultural_background(self, language: str) -> str:
        """Infer cultural background from language"""
        cultural_mapping = {
            'en': 'english_speaking',
            'es': 'hispanic',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'zh': 'chinese',
            'ja': 'japanese',
            'ko': 'korean',
            'ar': 'arabic',
            'hi': 'indian'
        }
        return cultural_mapping.get(language, 'international')
    
    async def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation insights including language patterns"""
        context = self.conversation_contexts.get(conversation_id)
        if not context:
            return {'error': 'Conversation not found'}
        
        return {
            'conversation_id': conversation_id,
            'language_profile': {
                'primary_language': context.primary_language,
                'detected_languages': context.detected_languages,
                'language_switches': len(context.detected_languages),
                'cultural_background': context.cultural_background,
                'communication_style': context.communication_style
            },
            'adaptation_recommendations': await self._get_adaptation_recommendations(context),
            'cultural_considerations': await self._get_cultural_considerations(context)
        }
    
    async def _get_adaptation_recommendations(self, context: ConversationContext) -> List[str]:
        """Get recommendations for conversation adaptation"""
        recommendations = []
        
        if len(context.detected_languages) > 1:
            recommendations.append("Multiple languages detected - consider confirming preferred language")
        
        if context.cultural_background in ['chinese', 'japanese', 'korean']:
            recommendations.append("High-context culture - allow more time for relationship building")
        
        if context.cultural_background in ['german', 'dutch', 'scandinavian']:
            recommendations.append("Direct communication preferred - be clear and concise")
        
        if context.primary_language != 'en':
            recommendations.append("Non-native English speaker - use simpler language and confirm understanding")
        
        return recommendations
    
    async def _get_cultural_considerations(self, context: ConversationContext) -> List[str]:
        """Get cultural considerations for the conversation"""
        considerations = {
            'hispanic': [
                "Relationship-building is important before business discussion",
                "Family and personal connections may influence business decisions",
                "Formal titles and respect for hierarchy are valued"
            ],
            'chinese': [
                "Face-saving and harmony are crucial cultural values",
                "Decision-making may involve multiple stakeholders",
                "Long-term relationship building is preferred over quick sales"
            ],
            'german': [
                "Punctuality and preparation are highly valued",
                "Direct communication and technical details are appreciated",
                "Thorough analysis before decision-making is common"
            ],
            'japanese': [
                "Consensus-building and group harmony are important",
                "Indirect communication style - read between the lines",
                "Respect for hierarchy and formal processes"
            ],
            'arabic': [
                "Personal relationships and trust are fundamental",
                "Religious and cultural sensitivities should be respected",
                "Hospitality and respect for traditions are important"
            ]
        }
        
        return considerations.get(context.cultural_background, [
            "Professional courtesy and respect are universally appreciated",
            "Clear communication and cultural sensitivity are important"
        ])


# Global multi-language conversation intelligence instance
multilingual_intelligence = MultiLanguageConversationIntelligence()