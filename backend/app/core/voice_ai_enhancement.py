"""
Voice AI Enhancement System
Real-time voice analysis, emotion detection, and intelligent conversation guidance
"""

import asyncio
import json
import logging
import os
import time
import numpy as np
import io
import wave
import webrtcvad
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import speech_recognition as sr
import pydub
from pydub import AudioSegment
import librosa
import scipy.signal
from collections import deque, defaultdict
import threading
import queue

# ML/AI imports
try:
    import torch
    import torch.nn as nn
    import torchaudio
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    from transformers import pipeline
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Voice emotion classifications"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CONFIDENT = "confident"
    INTERESTED = "interested"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    ANGRY = "angry"
    SAD = "sad"

class VoiceQuality(Enum):
    """Voice quality indicators"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class SpeechPattern(Enum):
    """Speech pattern types"""
    FAST_TALKING = "fast_talking"
    SLOW_TALKING = "slow_talking"
    FREQUENT_PAUSES = "frequent_pauses"
    INTERRUPTING = "interrupting"
    HESITANT = "hesitant"
    CONFIDENT = "confident"
    MONOTONE = "monotone"
    EXPRESSIVE = "expressive"

@dataclass
class VoiceMetrics:
    """Voice analysis metrics"""
    timestamp: datetime
    speaker_id: str
    
    # Audio quality metrics
    volume_level: float  # 0-1
    clarity_score: float  # 0-1
    background_noise_level: float  # 0-1
    
    # Speech characteristics
    speech_rate: float  # words per minute
    pause_frequency: float  # pauses per minute
    average_pause_duration: float  # seconds
    pitch_mean: float  # Hz
    pitch_variance: float
    
    # Emotion metrics
    primary_emotion: EmotionType
    emotion_confidence: float  # 0-1
    emotional_intensity: float  # 0-1
    emotional_stability: float  # 0-1 (consistency)
    
    # Engagement metrics
    engagement_score: float  # 0-1
    attention_level: float  # 0-1
    stress_indicators: float  # 0-1
    
    # Communication patterns
    interruption_count: int
    overlapping_speech: bool
    turn_taking_quality: float  # 0-1

@dataclass
class VoiceInsight:
    """Voice-based conversation insights"""
    insight_type: str
    description: str
    confidence: float
    recommendation: str
    urgency: str  # low, medium, high

@dataclass
class RealTimeGuidance:
    """Real-time conversation guidance"""
    guidance_type: str  # suggestion, warning, opportunity
    message: str
    urgency: str
    suggested_response: Optional[str] = None
    emotional_context: Optional[str] = None

class VoiceAIEnhancementSystem:
    """
    Advanced Voice AI Enhancement System
    Real-time voice analysis with emotion detection and conversation guidance
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_duration = 20  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)
        
        # Voice activity detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Most aggressive filtering
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        
        # Audio processing
        self.audio_buffer = deque(maxlen=1000)
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # Models
        self.models = {}
        self.processors = {}
        
        # Voice metrics tracking
        self.voice_metrics_history = defaultdict(deque)
        self.current_session_metrics = {}
        self.conversation_state = {}
        
        # Emotion analysis
        self.emotion_history = deque(maxlen=100)
        self.emotion_smoothing_window = 5
        
        # Real-time guidance
        self.guidance_callbacks = []
        self.active_guidances = []
        
        # Initialize components
        self._initialize_models()
        self._initialize_audio_processing()
        
    def _initialize_models(self):
        """Initialize AI models for voice analysis"""
        try:
            if torch:
                # Wav2Vec2 for speech recognition
                self.processors['wav2vec2'] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.models['wav2vec2'] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                
                # Emotion recognition pipeline
                self.models['emotion'] = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
                
                # Speech emotion recognition (alternative)
                self.models['speech_emotion'] = pipeline(
                    "audio-classification",
                    model="superb/wav2vec2-base-superb-er"
                )
                
            logger.info("Voice AI models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Some voice models failed to load: {e}")
            self.models = {}  # Fallback to basic processing
    
    def _initialize_audio_processing(self):
        """Initialize audio processing components"""
        try:
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("Audio processing initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processing: {e}")
    
    def register_guidance_callback(self, callback: Callable[[RealTimeGuidance], None]):
        """Register callback for real-time guidance notifications"""
        self.guidance_callbacks.append(callback)
    
    async def start_voice_analysis(self, session_id: str, speaker_id: str) -> bool:
        """Start real-time voice analysis for a conversation session"""
        try:
            self.conversation_state[session_id] = {
                'active': True,
                'speaker_id': speaker_id,
                'start_time': datetime.now(),
                'metrics': [],
                'insights': [],
                'total_speech_time': 0,
                'total_silence_time': 0,
                'emotion_transitions': []
            }
            
            self.is_processing = True
            logger.info(f"Started voice analysis for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice analysis: {e}")
            return False
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes, 
                                 speaker_id: str = None) -> Optional[VoiceMetrics]:
        """Process real-time audio chunk and extract voice metrics"""
        try:
            if session_id not in self.conversation_state:
                return None
            
            # Convert audio data to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Voice activity detection
            is_speech = self._detect_voice_activity(audio_data)
            
            if is_speech:
                # Extract voice metrics
                metrics = await self._extract_voice_metrics(
                    audio_array, session_id, speaker_id or "unknown"
                )
                
                # Store metrics
                self.conversation_state[session_id]['metrics'].append(metrics)
                
                # Analyze for insights and guidance
                await self._analyze_real_time_insights(session_id, metrics)
                
                return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return np.array([])
    
    def _detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect if audio contains speech"""
        try:
            # VAD expects specific frame sizes
            if len(audio_data) != self.frame_size * 2:  # 2 bytes per sample (16-bit)
                return False
            
            return self.vad.is_speech(audio_data, self.sample_rate)
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return False
    
    async def _extract_voice_metrics(self, audio: np.ndarray, session_id: str, 
                                   speaker_id: str) -> VoiceMetrics:
        """Extract comprehensive voice metrics from audio"""
        timestamp = datetime.now()
        
        # Audio quality metrics
        volume_level = self._calculate_volume_level(audio)
        clarity_score = self._calculate_clarity_score(audio)
        noise_level = self._estimate_background_noise(audio)
        
        # Speech characteristics
        speech_rate = await self._estimate_speech_rate(audio)
        pause_metrics = self._analyze_pause_patterns(audio)
        pitch_metrics = self._analyze_pitch_characteristics(audio)
        
        # Emotion analysis
        emotion_results = await self._analyze_emotion_from_audio(audio)
        
        # Engagement metrics
        engagement_score = self._calculate_engagement_score(audio, emotion_results)
        attention_level = self._estimate_attention_level(audio)
        stress_indicators = self._detect_stress_indicators(audio, pitch_metrics)
        
        # Communication patterns
        interruption_count = self._detect_interruptions(session_id, timestamp)
        overlapping_speech = self._detect_overlapping_speech(audio)
        turn_taking_quality = self._assess_turn_taking_quality(session_id)
        
        return VoiceMetrics(
            timestamp=timestamp,
            speaker_id=speaker_id,
            volume_level=volume_level,
            clarity_score=clarity_score,
            background_noise_level=noise_level,
            speech_rate=speech_rate,
            pause_frequency=pause_metrics['frequency'],
            average_pause_duration=pause_metrics['avg_duration'],
            pitch_mean=pitch_metrics['mean'],
            pitch_variance=pitch_metrics['variance'],
            primary_emotion=emotion_results['primary_emotion'],
            emotion_confidence=emotion_results['confidence'],
            emotional_intensity=emotion_results['intensity'],
            emotional_stability=emotion_results['stability'],
            engagement_score=engagement_score,
            attention_level=attention_level,
            stress_indicators=stress_indicators,
            interruption_count=interruption_count,
            overlapping_speech=overlapping_speech,
            turn_taking_quality=turn_taking_quality
        )
    
    def _calculate_volume_level(self, audio: np.ndarray) -> float:
        """Calculate normalized volume level"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # RMS calculation
            rms = np.sqrt(np.mean(audio**2))
            # Normalize to 0-1 range
            volume = min(rms * 10, 1.0)  # Scale factor for typical speech
            return float(volume)
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return 0.0
    
    def _calculate_clarity_score(self, audio: np.ndarray) -> float:
        """Calculate speech clarity score"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Simple clarity metric based on spectral characteristics
            fft = np.fft.fft(audio)
            power_spectrum = np.abs(fft)**2
            
            # Focus on speech frequency range (300-3400 Hz)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            speech_range_mask = (freqs >= 300) & (freqs <= 3400)
            
            speech_power = np.sum(power_spectrum[speech_range_mask])
            total_power = np.sum(power_spectrum)
            
            if total_power > 0:
                clarity = speech_power / total_power
            else:
                clarity = 0.0
            
            return min(float(clarity), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating clarity: {e}")
            return 0.5
    
    def _estimate_background_noise(self, audio: np.ndarray) -> float:
        """Estimate background noise level"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Use lower percentile of amplitude as noise floor estimate
            noise_floor = np.percentile(np.abs(audio), 10)
            noise_level = min(noise_floor * 20, 1.0)  # Scale and cap
            
            return float(noise_level)
            
        except Exception as e:
            logger.error(f"Error estimating noise: {e}")
            return 0.0
    
    async def _estimate_speech_rate(self, audio: np.ndarray) -> float:
        """Estimate words per minute from audio"""
        try:
            # Simplified estimation based on syllable detection
            # Detect syllables using energy peaks
            
            # Apply low-pass filter to smooth the signal
            nyquist = self.sample_rate / 2
            low_cutoff = 500 / nyquist
            b, a = scipy.signal.butter(4, low_cutoff, btype='low')
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            # Calculate energy envelope
            energy = np.abs(filtered_audio)
            
            # Smooth energy
            window_size = int(self.sample_rate * 0.05)  # 50ms window
            smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            
            # Find peaks (potential syllables)
            threshold = np.mean(smoothed_energy) * 1.5
            peaks, _ = scipy.signal.find_peaks(smoothed_energy, height=threshold, distance=int(self.sample_rate * 0.1))
            
            # Estimate speech rate
            audio_duration = len(audio) / self.sample_rate  # seconds
            if audio_duration > 0:
                syllables_per_second = len(peaks) / audio_duration
                # Rough conversion: ~1.5 syllables per word, 60 seconds per minute
                words_per_minute = syllables_per_second * 60 / 1.5
            else:
                words_per_minute = 0.0
            
            return min(float(words_per_minute), 300.0)  # Cap at reasonable maximum
            
        except Exception as e:
            logger.error(f"Error estimating speech rate: {e}")
            return 150.0  # Default average
    
    def _analyze_pause_patterns(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze pause patterns in speech"""
        try:
            # Detect silence periods
            energy = np.abs(audio)
            threshold = np.mean(energy) * 0.1  # 10% of mean energy
            
            silence_mask = energy < threshold
            
            # Find silence regions
            silence_starts = []
            silence_ends = []
            in_silence = False
            
            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_silence:
                    silence_starts.append(i)
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_ends.append(i)
                    in_silence = False
            
            # Calculate pause metrics
            if silence_starts and silence_ends:
                pause_durations = []
                for start, end in zip(silence_starts, silence_ends[:len(silence_starts)]):
                    duration = (end - start) / self.sample_rate
                    if duration > 0.1:  # Only count pauses longer than 100ms
                        pause_durations.append(duration)
                
                audio_duration = len(audio) / self.sample_rate
                if audio_duration > 0:
                    pause_frequency = len(pause_durations) / (audio_duration / 60)  # per minute
                    avg_duration = np.mean(pause_durations) if pause_durations else 0.0
                else:
                    pause_frequency = 0.0
                    avg_duration = 0.0
            else:
                pause_frequency = 0.0
                avg_duration = 0.0
            
            return {
                'frequency': float(pause_frequency),
                'avg_duration': float(avg_duration)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pauses: {e}")
            return {'frequency': 0.0, 'avg_duration': 0.0}
    
    def _analyze_pitch_characteristics(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze pitch characteristics of speech"""
        try:
            if len(audio) == 0:
                return {'mean': 0.0, 'variance': 0.0}
            
            # Use autocorrelation for pitch detection
            # This is a simplified implementation
            
            # Apply window to reduce edge effects
            windowed_audio = audio * np.hanning(len(audio))
            
            # Autocorrelation
            correlation = np.correlate(windowed_audio, windowed_audio, mode='full')
            correlation = correlation[correlation.size // 2:]
            
            # Find pitch period (in samples)
            min_period = int(self.sample_rate / 500)  # 500 Hz max
            max_period = int(self.sample_rate / 50)   # 50 Hz min
            
            if max_period < len(correlation):
                pitch_correlation = correlation[min_period:max_period]
                if len(pitch_correlation) > 0:
                    pitch_period = np.argmax(pitch_correlation) + min_period
                    pitch_freq = self.sample_rate / pitch_period
                else:
                    pitch_freq = 150.0  # Default
            else:
                pitch_freq = 150.0
            
            # For variance, we'd need multiple pitch estimates over time
            # This is a simplified version
            pitch_variance = 100.0  # Placeholder
            
            return {
                'mean': float(pitch_freq),
                'variance': float(pitch_variance)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pitch: {e}")
            return {'mean': 150.0, 'variance': 100.0}
    
    async def _analyze_emotion_from_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze emotion from audio using AI models"""
        try:
            if 'emotion' in self.models and len(audio) > 0:
                # Resample if needed for the model
                target_sr = 16000
                if len(audio) > target_sr * 10:  # Limit to 10 seconds max
                    audio = audio[:target_sr * 10]
                
                # Use emotion recognition model
                result = self.models['emotion'](audio, sampling_rate=target_sr)
                
                # Map model output to our emotion types
                emotion_mapping = {
                    'happy': EmotionType.HAPPY,
                    'sad': EmotionType.SAD,
                    'angry': EmotionType.ANGRY,
                    'fear': EmotionType.CONCERNED,
                    'surprise': EmotionType.EXCITED,
                    'disgust': EmotionType.FRUSTRATED,
                    'neutral': EmotionType.NEUTRAL
                }
                
                if isinstance(result, list) and len(result) > 0:
                    primary_label = result[0]['label'].lower()
                    confidence = result[0]['score']
                    
                    primary_emotion = emotion_mapping.get(primary_label, EmotionType.NEUTRAL)
                    
                    # Calculate emotional intensity and stability
                    intensity = confidence  # Use confidence as intensity
                    stability = self._calculate_emotional_stability(primary_emotion)
                    
                    return {
                        'primary_emotion': primary_emotion,
                        'confidence': float(confidence),
                        'intensity': float(intensity),
                        'stability': float(stability),
                        'raw_result': result
                    }
            
            # Fallback analysis based on audio characteristics
            return self._fallback_emotion_analysis(audio)
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return self._fallback_emotion_analysis(audio)
    
    def _fallback_emotion_analysis(self, audio: np.ndarray) -> Dict[str, Any]:
        """Fallback emotion analysis using audio features"""
        try:
            if len(audio) == 0:
                return {
                    'primary_emotion': EmotionType.NEUTRAL,
                    'confidence': 0.5,
                    'intensity': 0.5,
                    'stability': 0.5
                }
            
            # Simple heuristics based on audio characteristics
            energy = np.mean(audio**2)
            zero_crossing_rate = np.mean(np.diff(np.sign(audio)) != 0)
            
            # High energy + high ZCR might indicate excitement/anger
            if energy > 0.05 and zero_crossing_rate > 0.1:
                if zero_crossing_rate > 0.15:
                    emotion = EmotionType.ANGRY
                else:
                    emotion = EmotionType.EXCITED
                intensity = 0.7
            # Low energy might indicate sad/concerned
            elif energy < 0.01:
                emotion = EmotionType.CONCERNED
                intensity = 0.4
            # Medium energy
            else:
                emotion = EmotionType.NEUTRAL
                intensity = 0.5
            
            return {
                'primary_emotion': emotion,
                'confidence': 0.6,
                'intensity': intensity,
                'stability': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in fallback emotion analysis: {e}")
            return {
                'primary_emotion': EmotionType.NEUTRAL,
                'confidence': 0.5,
                'intensity': 0.5,
                'stability': 0.5
            }
    
    def _calculate_emotional_stability(self, current_emotion: EmotionType) -> float:
        """Calculate emotional stability based on recent emotion history"""
        try:
            self.emotion_history.append(current_emotion)
            
            if len(self.emotion_history) < 3:
                return 0.5  # Not enough history
            
            # Check for emotion changes in recent history
            recent_emotions = list(self.emotion_history)[-5:]
            unique_emotions = len(set(recent_emotions))
            
            # More unique emotions = less stability
            stability = max(0.0, 1.0 - (unique_emotions - 1) * 0.2)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating emotional stability: {e}")
            return 0.5
    
    def _calculate_engagement_score(self, audio: np.ndarray, emotion_results: Dict) -> float:
        """Calculate engagement score based on voice characteristics"""
        try:
            # Factors that contribute to engagement
            energy = np.mean(audio**2)
            
            # Energy contribution (0.4 weight)
            energy_score = min(energy * 20, 1.0)
            
            # Emotion contribution (0.3 weight)
            emotion = emotion_results['primary_emotion']
            if emotion in [EmotionType.EXCITED, EmotionType.INTERESTED, EmotionType.HAPPY]:
                emotion_score = 0.8
            elif emotion in [EmotionType.CONFIDENT, EmotionType.NEUTRAL]:
                emotion_score = 0.6
            elif emotion in [EmotionType.CONCERNED, EmotionType.CONFUSED]:
                emotion_score = 0.4
            else:
                emotion_score = 0.2
            
            # Intensity contribution (0.3 weight)
            intensity_score = emotion_results['intensity']
            
            # Weighted sum
            engagement_score = (
                energy_score * 0.4 +
                emotion_score * 0.3 +
                intensity_score * 0.3
            )
            
            return float(engagement_score)
            
        except Exception as e:
            logger.error(f"Error calculating engagement: {e}")
            return 0.5
    
    def _estimate_attention_level(self, audio: np.ndarray) -> float:
        """Estimate attention level from voice characteristics"""
        try:
            # Attention correlates with voice consistency and energy
            if len(audio) == 0:
                return 0.5
            
            # Voice consistency (less variation = more attention)
            energy_variation = np.std(audio**2)
            consistency_score = max(0, 1 - energy_variation * 10)
            
            # Energy level
            energy_level = min(np.mean(audio**2) * 15, 1.0)
            
            # Combined attention score
            attention = (consistency_score * 0.6 + energy_level * 0.4)
            
            return float(attention)
            
        except Exception as e:
            logger.error(f"Error estimating attention: {e}")
            return 0.5
    
    def _detect_stress_indicators(self, audio: np.ndarray, pitch_metrics: Dict) -> float:
        """Detect stress indicators in voice"""
        try:
            stress_score = 0.0
            
            # High pitch variance can indicate stress
            pitch_variance = pitch_metrics['variance']
            if pitch_variance > 200:
                stress_score += 0.3
            
            # High frequency content (tension)
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            high_freq_power = np.sum(np.abs(fft[(freqs > 2000) & (freqs < 4000)])**2)
            total_power = np.sum(np.abs(fft)**2)
            
            if total_power > 0 and high_freq_power / total_power > 0.1:
                stress_score += 0.4
            
            # Voice tremor (rapid fluctuations)
            if len(audio) > self.sample_rate:  # At least 1 second
                segments = np.array_split(audio, 10)
                energies = [np.mean(seg**2) for seg in segments]
                energy_variation = np.std(energies)
                if energy_variation > 0.01:
                    stress_score += 0.3
            
            return min(float(stress_score), 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting stress: {e}")
            return 0.0
    
    def _detect_interruptions(self, session_id: str, timestamp: datetime) -> int:
        """Detect interruptions in conversation"""
        try:
            if session_id not in self.conversation_state:
                return 0
            
            # Simple interruption detection based on timing
            # This would be more sophisticated with multiple audio streams
            return 0  # Placeholder
            
        except Exception as e:
            logger.error(f"Error detecting interruptions: {e}")
            return 0
    
    def _detect_overlapping_speech(self, audio: np.ndarray) -> bool:
        """Detect overlapping speech patterns"""
        try:
            # Simplified detection - would need multiple audio channels
            # Look for unusual energy patterns that might indicate overlap
            if len(audio) < self.sample_rate:  # Less than 1 second
                return False
            
            # Check for energy spikes that might indicate multiple speakers
            energy = np.abs(audio)
            energy_smooth = np.convolve(energy, np.ones(100)/100, mode='same')
            
            # Look for multiple prominent peaks
            peaks, _ = scipy.signal.find_peaks(energy_smooth, height=np.mean(energy_smooth) * 2)
            
            return len(peaks) > 3  # Heuristic threshold
            
        except Exception as e:
            logger.error(f"Error detecting overlapping speech: {e}")
            return False
    
    def _assess_turn_taking_quality(self, session_id: str) -> float:
        """Assess quality of turn-taking in conversation"""
        try:
            if session_id not in self.conversation_state:
                return 0.5
            
            # Analyze recent conversation patterns
            # This is a simplified version - would analyze speaker transitions
            return 0.7  # Placeholder
            
        except Exception as e:
            logger.error(f"Error assessing turn-taking: {e}")
            return 0.5
    
    async def _analyze_real_time_insights(self, session_id: str, metrics: VoiceMetrics):
        """Analyze voice metrics for real-time insights and guidance"""
        try:
            insights = []
            guidances = []
            
            # Emotion-based insights
            emotion = metrics.primary_emotion
            emotion_confidence = metrics.emotion_confidence
            
            if emotion == EmotionType.FRUSTRATED and emotion_confidence > 0.7:
                insights.append(VoiceInsight(
                    insight_type="emotion_alert",
                    description="Customer shows signs of frustration",
                    confidence=emotion_confidence,
                    recommendation="Acknowledge concerns and offer solutions",
                    urgency="high"
                ))
                
                guidances.append(RealTimeGuidance(
                    guidance_type="warning",
                    message="Customer frustration detected - address concerns immediately",
                    urgency="high",
                    emotional_context="frustrated",
                    suggested_response="I understand your concerns. Let me help address this right away."
                ))
            
            elif emotion == EmotionType.EXCITED and emotion_confidence > 0.6:
                insights.append(VoiceInsight(
                    insight_type="opportunity",
                    description="Customer showing high interest/excitement",
                    confidence=emotion_confidence,
                    recommendation="Capitalize on enthusiasm - move toward close",
                    urgency="medium"
                ))
                
                guidances.append(RealTimeGuidance(
                    guidance_type="opportunity",
                    message="Customer excitement detected - great time to advance the sale",
                    urgency="medium",
                    emotional_context="excited"
                ))
            
            # Engagement insights
            if metrics.engagement_score < 0.3:
                insights.append(VoiceInsight(
                    insight_type="engagement_warning",
                    description="Low customer engagement detected",
                    confidence=0.8,
                    recommendation="Ask engaging questions or change topic",
                    urgency="medium"
                ))
                
                guidances.append(RealTimeGuidance(
                    guidance_type="suggestion",
                    message="Low engagement - try asking open-ended questions",
                    urgency="medium"
                ))
            
            # Speech pattern insights
            if metrics.speech_rate > 200:  # Very fast talking
                insights.append(VoiceInsight(
                    insight_type="communication_pattern",
                    description="Customer speaking very rapidly",
                    confidence=0.7,
                    recommendation="Customer may be excited or stressed - gauge emotional state",
                    urgency="low"
                ))
            
            # Stress indicators
            if metrics.stress_indicators > 0.7:
                insights.append(VoiceInsight(
                    insight_type="stress_alert", 
                    description="High stress indicators in customer voice",
                    confidence=0.8,
                    recommendation="Slow down conversation pace, be reassuring",
                    urgency="high"
                ))
                
                guidances.append(RealTimeGuidance(
                    guidance_type="warning",
                    message="Customer stress detected - use calming tone and slower pace",
                    urgency="high",
                    emotional_context="stressed"
                ))
            
            # Store insights
            if session_id in self.conversation_state:
                self.conversation_state[session_id]['insights'].extend(insights)
            
            # Send real-time guidance
            for guidance in guidances:
                await self._send_guidance(guidance)
                
        except Exception as e:
            logger.error(f"Error analyzing real-time insights: {e}")
    
    async def _send_guidance(self, guidance: RealTimeGuidance):
        """Send real-time guidance to registered callbacks"""
        try:
            self.active_guidances.append(guidance)
            
            for callback in self.guidance_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(guidance)
                    else:
                        callback(guidance)
                except Exception as e:
                    logger.error(f"Error in guidance callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending guidance: {e}")
    
    def _audio_processing_loop(self):
        """Main audio processing loop running in background thread"""
        while True:
            try:
                if not self.processing_queue.empty():
                    audio_data = self.processing_queue.get(timeout=1)
                    # Process audio data
                    # This would handle background processing tasks
                    pass
                else:
                    # No data to process
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary with voice analytics"""
        try:
            if session_id not in self.conversation_state:
                return {}
            
            session = self.conversation_state[session_id]
            metrics_list = session.get('metrics', [])
            
            if not metrics_list:
                return {
                    'session_id': session_id,
                    'status': 'no_data',
                    'message': 'No voice metrics available for this session'
                }
            
            # Calculate aggregate metrics
            avg_engagement = np.mean([m.engagement_score for m in metrics_list])
            avg_stress = np.mean([m.stress_indicators for m in metrics_list])
            emotion_distribution = self._calculate_emotion_distribution(metrics_list)
            
            # Voice quality assessment
            avg_clarity = np.mean([m.clarity_score for m in metrics_list])
            avg_volume = np.mean([m.volume_level for m in metrics_list])
            
            # Communication patterns
            avg_speech_rate = np.mean([m.speech_rate for m in metrics_list])
            total_interruptions = sum(m.interruption_count for m in metrics_list)
            
            return {
                'session_id': session_id,
                'duration': (datetime.now() - session['start_time']).total_seconds(),
                'metrics_count': len(metrics_list),
                'voice_quality': {
                    'overall_clarity': float(avg_clarity),
                    'average_volume': float(avg_volume),
                    'quality_rating': self._get_quality_rating(avg_clarity, avg_volume)
                },
                'emotional_analysis': {
                    'average_engagement': float(avg_engagement),
                    'average_stress_level': float(avg_stress),
                    'emotion_distribution': emotion_distribution,
                    'emotional_stability': self._calculate_session_emotional_stability(metrics_list)
                },
                'communication_patterns': {
                    'average_speech_rate': float(avg_speech_rate),
                    'total_interruptions': total_interruptions,
                    'speaking_pattern': self._classify_speaking_pattern(avg_speech_rate)
                },
                'insights': session.get('insights', []),
                'recommendations': self._generate_session_recommendations(metrics_list)
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {'session_id': session_id, 'error': str(e)}
    
    def _calculate_emotion_distribution(self, metrics_list: List[VoiceMetrics]) -> Dict[str, float]:
        """Calculate distribution of emotions throughout session"""
        if not metrics_list:
            return {}
        
        emotion_counts = defaultdict(int)
        for metrics in metrics_list:
            emotion_counts[metrics.primary_emotion.value] += 1
        
        total = len(metrics_list)
        return {emotion: count/total for emotion, count in emotion_counts.items()}
    
    def _get_quality_rating(self, clarity: float, volume: float) -> str:
        """Get overall voice quality rating"""
        combined_score = (clarity + volume) / 2
        
        if combined_score > 0.8:
            return VoiceQuality.EXCELLENT.value
        elif combined_score > 0.6:
            return VoiceQuality.GOOD.value
        elif combined_score > 0.4:
            return VoiceQuality.FAIR.value
        else:
            return VoiceQuality.POOR.value
    
    def _calculate_session_emotional_stability(self, metrics_list: List[VoiceMetrics]) -> float:
        """Calculate emotional stability across entire session"""
        if len(metrics_list) < 2:
            return 0.5
        
        emotions = [m.primary_emotion for m in metrics_list]
        unique_emotions = len(set(emotions))
        
        # More stability = fewer emotion changes
        stability = max(0.0, 1.0 - (unique_emotions - 1) * 0.15)
        return float(stability)
    
    def _classify_speaking_pattern(self, speech_rate: float) -> str:
        """Classify speaking pattern based on speech rate"""
        if speech_rate > 180:
            return SpeechPattern.FAST_TALKING.value
        elif speech_rate < 120:
            return SpeechPattern.SLOW_TALKING.value
        else:
            return SpeechPattern.CONFIDENT.value
    
    def _generate_session_recommendations(self, metrics_list: List[VoiceMetrics]) -> List[str]:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        if not metrics_list:
            return ["Insufficient data for recommendations"]
        
        # Engagement recommendations
        avg_engagement = np.mean([m.engagement_score for m in metrics_list])
        if avg_engagement < 0.4:
            recommendations.append("Focus on increasing customer engagement through interactive questions")
        
        # Emotional state recommendations
        avg_stress = np.mean([m.stress_indicators for m in metrics_list])
        if avg_stress > 0.6:
            recommendations.append("Customer showed stress - follow up with reassurance and support")
        
        # Communication quality
        avg_clarity = np.mean([m.clarity_score for m in metrics_list])
        if avg_clarity < 0.5:
            recommendations.append("Consider improving call quality or checking connection")
        
        # Speech pattern insights
        avg_speech_rate = np.mean([m.speech_rate for m in metrics_list])
        if avg_speech_rate > 200:
            recommendations.append("Customer spoke rapidly - they may be excited or anxious")
        elif avg_speech_rate < 100:
            recommendations.append("Slow speech detected - customer may need more engagement")
        
        return recommendations
    
    async def stop_voice_analysis(self, session_id: str) -> Dict[str, Any]:
        """Stop voice analysis and return final summary"""
        try:
            if session_id in self.conversation_state:
                self.conversation_state[session_id]['active'] = False
                summary = await self.get_session_summary(session_id)
                
                # Clean up session data
                del self.conversation_state[session_id]
                
                logger.info(f"Stopped voice analysis for session {session_id}")
                return summary
            else:
                return {'error': 'Session not found'}
                
        except Exception as e:
            logger.error(f"Error stopping voice analysis: {e}")
            return {'error': str(e)}

# Global instance
voice_ai_enhancement = VoiceAIEnhancementSystem()