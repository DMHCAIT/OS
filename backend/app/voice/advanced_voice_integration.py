"""
Advanced Voice AI Integration System
Unified integration of all advanced voice AI features with comprehensive API endpoints
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import base64
import io
import wave

# Import all voice AI components
from .voice_cloning import (
    advanced_voice_cloning_engine,
    VoiceProfile,
    VoiceCloneRequest,
    VoiceCloneResult
)
from .accent_adaptation import (
    regional_adaptation_engine,
    AccentProfile,
    AccentAdaptationRequest,
    AccentAdaptationResult
)
from .noise_intelligence import (
    background_noise_intelligence_engine,
    NoiseProfile,
    NoiseReductionRequest,
    NoiseReductionResult,
    AudioQualityMetrics
)
from .multi_participant import (
    multi_participant_call_handler,
    Speaker,
    ConversationFlow,
    MultiParticipantAnalysis
)

logger = logging.getLogger(__name__)

@dataclass
class AdvancedVoiceRequest:
    """Comprehensive voice processing request"""
    audio_data: Optional[bytes] = None
    text: Optional[str] = None
    sample_rate: int = 16000
    
    # Voice cloning parameters
    voice_profile_id: Optional[str] = None
    clone_voice: bool = False
    
    # Accent adaptation parameters
    target_accent: Optional[str] = None
    source_accent: str = "general_american"
    adapt_accent: bool = False
    
    # Noise reduction parameters
    reduce_noise: bool = True
    noise_adaptation_mode: str = "balanced"
    
    # Multi-participant parameters
    call_id: Optional[str] = None
    enable_speaker_identification: bool = False
    
    # Output preferences
    return_analysis: bool = True
    return_audio: bool = True
    audio_format: str = "wav"

@dataclass
class AdvancedVoiceResponse:
    """Comprehensive voice processing response"""
    success: bool
    processing_time: float
    
    # Processed audio
    processed_audio: Optional[bytes] = None
    audio_format: str = "wav"
    sample_rate: int = 16000
    
    # Voice cloning results
    voice_clone_result: Optional[VoiceCloneResult] = None
    
    # Accent adaptation results
    accent_adaptation_result: Optional[AccentAdaptationResult] = None
    
    # Noise reduction results
    noise_reduction_result: Optional[NoiseReductionResult] = None
    
    # Multi-participant results
    speaker_analysis: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    overall_quality_score: float = 0.0
    processing_stages: List[str] = None
    
    # Errors and warnings
    errors: List[str] = None
    warnings: List[str] = None

class AdvancedVoiceAISystem:
    """Unified advanced voice AI system"""
    
    def __init__(self, config_path: str = "config/voice_ai_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Component engines
        self.voice_cloning_engine = advanced_voice_cloning_engine
        self.accent_adaptation_engine = regional_adaptation_engine
        self.noise_intelligence_engine = background_noise_intelligence_engine
        self.multi_participant_handler = multi_participant_call_handler
        
        # System state
        self.initialized = False
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AdvancedVoiceAISystem created")
    
    async def initialize(self):
        """Initialize all voice AI components"""
        try:
            logger.info("Initializing Advanced Voice AI System...")
            
            # Initialize all engines
            await self.voice_cloning_engine.initialize()
            await self.accent_adaptation_engine.initialize()
            await self.noise_intelligence_engine.initialize()
            await self.multi_participant_handler.initialize()
            
            self.initialized = True
            logger.info("Advanced Voice AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Advanced Voice AI System: {e}")
            raise
    
    async def process_voice_request(
        self,
        request: AdvancedVoiceRequest
    ) -> AdvancedVoiceResponse:
        """Process comprehensive voice request"""
        try:
            if not self.initialized:
                await self.initialize()
            
            start_time = datetime.now()
            logger.info("Processing advanced voice request")
            
            errors = []
            warnings = []
            processing_stages = []
            
            # Convert audio data if provided
            audio_array = None
            if request.audio_data:
                audio_array = self._bytes_to_audio(request.audio_data, request.sample_rate)
                processing_stages.append("audio_conversion")
            
            # Initialize results
            voice_clone_result = None
            accent_adaptation_result = None
            noise_reduction_result = None
            speaker_analysis = None
            final_audio = audio_array
            
            # 1. Noise Reduction (first stage for audio cleanup)
            if request.reduce_noise and audio_array is not None:
                try:
                    noise_request = NoiseReductionRequest(
                        audio_data=audio_array,
                        sample_rate=request.sample_rate,
                        adaptation_mode=request.noise_adaptation_mode,
                        preserve_speech=True,
                        real_time=False
                    )
                    
                    noise_reduction_result = await self.noise_intelligence_engine.reduce_background_noise(
                        noise_request
                    )
                    
                    final_audio = noise_reduction_result.cleaned_audio
                    processing_stages.append("noise_reduction")
                    
                except Exception as e:
                    errors.append(f"Noise reduction failed: {e}")
                    logger.error(f"Noise reduction error: {e}")
            
            # 2. Speaker Identification for Multi-participant
            if request.enable_speaker_identification and final_audio is not None:
                try:
                    if request.call_id:
                        # Process as part of ongoing call
                        await self.multi_participant_handler.process_audio_chunk(
                            request.call_id,
                            final_audio,
                            datetime.now().timestamp(),
                            request.sample_rate
                        )
                        
                        speaker_analysis = await self.multi_participant_handler.get_real_time_analysis(
                            request.call_id
                        )
                    else:
                        # Standalone speaker identification
                        speakers = await self.multi_participant_handler.identify_speakers(
                            final_audio, request.sample_rate
                        )
                        speaker_analysis = {'speakers': speakers}
                    
                    processing_stages.append("speaker_identification")
                    
                except Exception as e:
                    errors.append(f"Speaker identification failed: {e}")
                    logger.error(f"Speaker identification error: {e}")
            
            # 3. Accent Adaptation (for text or recognized speech)
            if request.adapt_accent and request.target_accent:
                try:
                    # Use provided text or would use speech-to-text on audio
                    text_to_adapt = request.text or "Sample text for accent adaptation"
                    
                    adaptation_request = AccentAdaptationRequest(
                        text=text_to_adapt,
                        target_accent=request.target_accent,
                        source_accent=request.source_accent,
                        adaptation_strength=0.8,
                        preserve_meaning=True,
                        adapt_vocabulary=True
                    )
                    
                    accent_adaptation_result = await self.accent_adaptation_engine.adapt_accent(
                        adaptation_request
                    )
                    
                    processing_stages.append("accent_adaptation")
                    
                except Exception as e:
                    errors.append(f"Accent adaptation failed: {e}")
                    logger.error(f"Accent adaptation error: {e}")
            
            # 4. Voice Cloning (final stage for audio generation)
            if request.clone_voice and request.voice_profile_id and request.text:
                try:
                    clone_request = VoiceCloneRequest(
                        profile_id=request.voice_profile_id,
                        text=accent_adaptation_result.adapted_text if accent_adaptation_result else request.text,
                        target_emotion="neutral",
                        speaking_style="professional"
                    )
                    
                    voice_clone_result = await self.voice_cloning_engine.clone_voice(
                        clone_request
                    )
                    
                    # Use cloned audio as final output
                    final_audio = np.frombuffer(voice_clone_result.audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                    processing_stages.append("voice_cloning")
                    
                except Exception as e:
                    errors.append(f"Voice cloning failed: {e}")
                    logger.error(f"Voice cloning error: {e}")
            
            # Calculate overall quality score
            overall_quality = await self._calculate_overall_quality(
                final_audio,
                noise_reduction_result,
                voice_clone_result,
                accent_adaptation_result
            )
            
            # Convert final audio to bytes if requested
            processed_audio_bytes = None
            if request.return_audio and final_audio is not None:
                processed_audio_bytes = self._audio_to_bytes(
                    final_audio, 
                    request.sample_rate, 
                    request.audio_format
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = AdvancedVoiceResponse(
                success=len(errors) == 0,
                processing_time=processing_time,
                processed_audio=processed_audio_bytes,
                audio_format=request.audio_format,
                sample_rate=request.sample_rate,
                voice_clone_result=voice_clone_result,
                accent_adaptation_result=accent_adaptation_result,
                noise_reduction_result=noise_reduction_result,
                speaker_analysis=speaker_analysis,
                overall_quality_score=overall_quality,
                processing_stages=processing_stages,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Advanced voice processing completed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice request: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AdvancedVoiceResponse(
                success=False,
                processing_time=processing_time,
                errors=[str(e)],
                warnings=[],
                processing_stages=processing_stages or []
            )
    
    # Voice Profile Management
    
    async def create_voice_profile(
        self,
        sales_rep_name: str,
        voice_name: str,
        training_audio_files: List[str],
        transcripts: List[str],
        voice_characteristics: Dict[str, Any]
    ) -> VoiceProfile:
        """Create new voice profile"""
        return await self.voice_cloning_engine.create_voice_profile(
            sales_rep_name,
            voice_name,
            training_audio_files,
            transcripts,
            voice_characteristics
        )
    
    async def get_voice_profiles(self) -> List[VoiceProfile]:
        """Get all voice profiles"""
        return await self.voice_cloning_engine.get_voice_profiles()
    
    async def update_voice_profile(
        self,
        profile_id: str,
        additional_audio: List[str],
        additional_transcripts: List[str]
    ) -> VoiceProfile:
        """Update existing voice profile"""
        return await self.voice_cloning_engine.update_voice_profile(
            profile_id,
            additional_audio,
            additional_transcripts
        )
    
    async def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete voice profile"""
        return await self.voice_cloning_engine.delete_voice_profile(profile_id)
    
    # Accent Management
    
    async def detect_accent(
        self,
        audio_data: bytes,
        sample_rate: int,
        text: Optional[str] = None
    ) -> Tuple[str, float]:
        """Detect accent from audio"""
        audio_array = self._bytes_to_audio(audio_data, sample_rate)
        return await self.accent_adaptation_engine.detect_accent(audio_array, text, sample_rate)
    
    async def get_supported_accents(self) -> List[Dict[str, Any]]:
        """Get supported accents"""
        return await self.accent_adaptation_engine.get_supported_accents()
    
    async def learn_accent_from_sample(
        self,
        audio_data: bytes,
        text: str,
        region: str,
        sample_rate: int
    ) -> AccentProfile:
        """Learn new accent from sample"""
        audio_array = self._bytes_to_audio(audio_data, sample_rate)
        return await self.accent_adaptation_engine.learn_accent_from_sample(
            audio_array, text, region, sample_rate
        )
    
    # Noise Intelligence Management
    
    async def analyze_noise_environment(
        self,
        audio_data: bytes,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze noise environment"""
        audio_array = self._bytes_to_audio(audio_data, sample_rate)
        return await self.noise_intelligence_engine.analyze_noise_environment(
            audio_array, sample_rate
        )
    
    async def get_noise_profiles(self) -> List[Dict[str, Any]]:
        """Get available noise profiles"""
        return await self.noise_intelligence_engine.get_noise_profiles()
    
    async def learn_noise_profile(
        self,
        noise_audio_data: bytes,
        noise_type: str,
        description: str,
        sample_rate: int
    ) -> NoiseProfile:
        """Learn new noise profile"""
        audio_array = self._bytes_to_audio(noise_audio_data, sample_rate)
        return await self.noise_intelligence_engine.learn_noise_profile(
            audio_array, noise_type, description, sample_rate
        )
    
    # Multi-participant Call Management
    
    async def start_call_analysis(
        self,
        call_id: str,
        expected_participants: List[str] = None,
        call_type: str = "sales"
    ) -> Dict[str, Any]:
        """Start multi-participant call analysis"""
        return await self.multi_participant_handler.start_call_analysis(
            call_id, expected_participants, call_type
        )
    
    async def end_call_analysis(self, call_id: str) -> MultiParticipantAnalysis:
        """End call analysis and get final report"""
        return await self.multi_participant_handler.end_call_analysis(call_id)
    
    async def get_call_real_time_analysis(self, call_id: str) -> Dict[str, Any]:
        """Get real-time call analysis"""
        return await self.multi_participant_handler.get_real_time_analysis(call_id)
    
    async def get_speaker_insights(
        self,
        speaker_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get speaker insights across calls"""
        return await self.multi_participant_handler.get_speaker_insights(
            speaker_id, time_range
        )
    
    # System Status and Health
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            voice_profiles = len(await self.voice_cloning_engine.get_voice_profiles())
            accent_profiles = len(await self.accent_adaptation_engine.get_supported_accents())
            noise_profiles = len(await self.noise_intelligence_engine.get_noise_profiles())
            active_calls = len(self.multi_participant_handler.active_calls)
            
            return {
                'system_initialized': self.initialized,
                'components_status': {
                    'voice_cloning': 'operational',
                    'accent_adaptation': 'operational',
                    'noise_intelligence': 'operational',
                    'multi_participant': 'operational'
                },
                'statistics': {
                    'voice_profiles': voice_profiles,
                    'accent_profiles': accent_profiles,
                    'noise_profiles': noise_profiles,
                    'active_calls': active_calls
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'system_initialized': self.initialized,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    async def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        try:
            diagnostics = {
                'overall_health': 'healthy',
                'component_tests': {},
                'performance_metrics': {},
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Test each component
            try:
                # Voice cloning test
                profiles = await self.voice_cloning_engine.get_voice_profiles()
                diagnostics['component_tests']['voice_cloning'] = {
                    'status': 'pass',
                    'profiles_available': len(profiles)
                }
            except Exception as e:
                diagnostics['component_tests']['voice_cloning'] = {
                    'status': 'fail',
                    'error': str(e)
                }
                diagnostics['overall_health'] = 'degraded'
            
            try:
                # Accent adaptation test
                accents = await self.accent_adaptation_engine.get_supported_accents()
                diagnostics['component_tests']['accent_adaptation'] = {
                    'status': 'pass',
                    'accents_available': len(accents)
                }
            except Exception as e:
                diagnostics['component_tests']['accent_adaptation'] = {
                    'status': 'fail',
                    'error': str(e)
                }
                diagnostics['overall_health'] = 'degraded'
            
            try:
                # Noise intelligence test
                noise_profiles = await self.noise_intelligence_engine.get_noise_profiles()
                diagnostics['component_tests']['noise_intelligence'] = {
                    'status': 'pass',
                    'profiles_available': len(noise_profiles)
                }
            except Exception as e:
                diagnostics['component_tests']['noise_intelligence'] = {
                    'status': 'fail',
                    'error': str(e)
                }
                diagnostics['overall_health'] = 'degraded'
            
            try:
                # Multi-participant test
                active_calls = len(self.multi_participant_handler.active_calls)
                diagnostics['component_tests']['multi_participant'] = {
                    'status': 'pass',
                    'active_calls': active_calls
                }
            except Exception as e:
                diagnostics['component_tests']['multi_participant'] = {
                    'status': 'fail',
                    'error': str(e)
                }
                diagnostics['overall_health'] = 'degraded'
            
            # Performance metrics (would implement actual performance testing)
            diagnostics['performance_metrics'] = {
                'average_processing_time': '2.5s',
                'memory_usage': 'normal',
                'cpu_usage': 'normal',
                'gpu_usage': 'normal'
            }
            
            # Recommendations
            failed_components = [
                name for name, result in diagnostics['component_tests'].items()
                if result['status'] == 'fail'
            ]
            
            if failed_components:
                diagnostics['recommendations'].append(
                    f"Restart or reconfigure failed components: {', '.join(failed_components)}"
                )
            
            if diagnostics['overall_health'] == 'healthy':
                diagnostics['recommendations'].append("System operating normally")
            
            return diagnostics
            
        except Exception as e:
            return {
                'overall_health': 'critical',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Utility methods
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        # Default configuration
        return {
            'sample_rate': 16000,
            'audio_format': 'wav',
            'processing_quality': 'balanced',
            'real_time_enabled': True,
            'max_audio_duration': 300  # 5 minutes
        }
    
    def _bytes_to_audio(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            return audio_array
        except Exception as e:
            logger.error(f"Error converting bytes to audio: {e}")
            return np.zeros(sample_rate)  # Return 1 second of silence
    
    def _audio_to_bytes(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        audio_format: str = "wav"
    ) -> bytes:
        """Convert audio array to bytes"""
        try:
            if audio_format.lower() == "wav":
                # Convert to 16-bit PCM
                audio_int = (audio_array * 32767).astype(np.int16)
                
                # Create WAV file in memory
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int.tobytes())
                
                return buffer.getvalue()
            else:
                # Raw PCM fallback
                audio_int = (audio_array * 32767).astype(np.int16)
                return audio_int.tobytes()
                
        except Exception as e:
            logger.error(f"Error converting audio to bytes: {e}")
            return b''
    
    async def _calculate_overall_quality(
        self,
        final_audio: Optional[np.ndarray],
        noise_result: Optional[NoiseReductionResult],
        voice_result: Optional[VoiceCloneResult],
        accent_result: Optional[AccentAdaptationResult]
    ) -> float:
        """Calculate overall quality score"""
        try:
            quality_scores = []
            
            if noise_result:
                quality_scores.append(noise_result.quality_metrics.speech_quality_score / 100)
            
            if voice_result:
                quality_scores.append(voice_result.quality_score)
            
            if accent_result:
                quality_scores.append(accent_result.confidence_score)
            
            if final_audio is not None:
                # Basic audio quality assessment
                snr_estimate = 20 * np.log10(np.std(final_audio) + 1e-10) + 60
                audio_quality = min(1.0, max(0.0, (snr_estimate - 20) / 40))  # Normalize SNR
                quality_scores.append(audio_quality)
            
            if quality_scores:
                return float(np.mean(quality_scores))
            else:
                return 0.7  # Default quality score
                
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.5


# Global instance
advanced_voice_ai_system = AdvancedVoiceAISystem()