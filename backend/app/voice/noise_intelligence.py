"""
Background Noise Intelligence System
Adaptive audio filtering and enhancement for better conversation quality in noisy environments
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import librosa
import scipy.signal
import scipy.ndimage
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import queue

# Advanced audio processing imports
try:
    import noisereduce as nr
    import webrtcvad
    import pyaudio
    from scipy.signal import butter, filtfilt, hilbert, welch
    from sklearn.decomposition import FastICA, NMF
    import torch.nn.utils.spectral_norm as spectral_norm
    from pesq import pesq
    from pystoi import stoi
except ImportError:
    nr = None
    webrtcvad = None
    pyaudio = None
    pesq = None
    stoi = None

logger = logging.getLogger(__name__)

@dataclass
class NoiseProfile:
    """Profile for different types of background noise"""
    noise_id: str
    noise_type: str  # traffic, office, restaurant, wind, etc.
    description: str
    
    # Spectral characteristics
    frequency_bands: List[Tuple[float, float]]  # Dominant frequency ranges
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    
    # Temporal characteristics
    stationarity: float  # 0 = non-stationary, 1 = stationary
    periodicity: float
    onset_rate: float  # Events per second
    
    # Statistical properties
    mean_power: float
    variance: float
    snr_threshold: float  # Minimum SNR to apply this profile
    
    # Processing parameters
    filter_type: str  # spectral_subtraction, wiener, adaptive
    adaptation_rate: float
    learning_enabled: bool
    
    # Performance metrics
    success_rate: float
    average_improvement: float  # dB improvement
    last_updated: datetime
    usage_count: int

@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics"""
    snr_db: float
    speech_quality_score: float  # 0-100
    intelligibility_score: float  # 0-100
    noise_level: float
    speech_level: float
    
    # Perceptual metrics
    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None
    
    # Spectral metrics
    spectral_distortion: float = 0.0
    harmonic_distortion: float = 0.0
    
    # Temporal metrics
    speech_pause_ratio: float = 0.0
    voice_activity_detection_accuracy: float = 0.0

@dataclass
class NoiseReductionRequest:
    """Request for noise reduction processing"""
    audio_data: np.ndarray
    sample_rate: int
    noise_type: Optional[str] = None
    adaptation_mode: str = "auto"  # auto, aggressive, conservative
    preserve_speech: bool = True
    real_time: bool = False
    quality_target: str = "balanced"  # quality, speed, balanced

@dataclass
class NoiseReductionResult:
    """Result of noise reduction processing"""
    cleaned_audio: np.ndarray
    noise_removed: np.ndarray
    processing_time: float
    quality_metrics: AudioQualityMetrics
    noise_profile_used: str
    confidence_score: float
    processing_artifacts: List[str]

class SpectralSubtractionModule(nn.Module):
    """Neural spectral subtraction for noise reduction"""
    
    def __init__(self, n_fft=512, hop_length=256, n_mels=80):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Spectral analysis layers
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Noise estimation network
        self.noise_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_mels),
            nn.Sigmoid()
        )
        
        # Speech enhancement network
        self.speech_enhancer = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, n_mels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for temporal modeling
        self.temporal_attention = nn.MultiheadAttention(256, num_heads=8)
    
    def forward(self, mel_spectrogram):
        batch_size, n_mels, time_frames = mel_spectrogram.shape
        
        # Encode spectral features
        encoded = self.spectral_encoder(mel_spectrogram)
        
        # Apply temporal attention
        encoded_t = encoded.permute(2, 0, 1)  # T x B x C
        attended, _ = self.temporal_attention(encoded_t, encoded_t, encoded_t)
        attended = attended.permute(1, 2, 0)  # B x C x T
        
        # Estimate noise
        noise_mask = self.noise_estimator(attended.mean(dim=-1))
        noise_mask = noise_mask.unsqueeze(-1).expand(-1, -1, time_frames)
        
        # Enhance speech
        speech_mask = self.speech_enhancer(attended)
        
        # Apply masks
        enhanced_spectrum = mel_spectrogram * speech_mask * (1 - noise_mask)
        
        return enhanced_spectrum, speech_mask, noise_mask

class AdaptiveWienerFilter:
    """Adaptive Wiener filter for noise reduction"""
    
    def __init__(self, frame_size=512, hop_length=256, adaptation_rate=0.1):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.adaptation_rate = adaptation_rate
        
        # Filter coefficients
        self.noise_spectrum = None
        self.speech_spectrum = None
        self.wiener_gain = None
        
        # Adaptation parameters
        self.noise_floor = 1e-6
        self.max_gain = 10.0
        self.min_gain = 0.1
    
    def initialize_noise_profile(self, noise_audio: np.ndarray, sample_rate: int):
        """Initialize noise profile from noise-only audio"""
        try:
            # Compute noise spectrum
            noise_stft = librosa.stft(
                noise_audio, 
                n_fft=self.frame_size, 
                hop_length=self.hop_length
            )
            self.noise_spectrum = np.mean(np.abs(noise_stft) ** 2, axis=1)
            
            # Add noise floor
            self.noise_spectrum = np.maximum(self.noise_spectrum, self.noise_floor)
            
            logger.info("Noise profile initialized")
            
        except Exception as e:
            logger.error(f"Error initializing noise profile: {e}")
            # Default noise profile
            self.noise_spectrum = np.ones(self.frame_size // 2 + 1) * self.noise_floor
    
    def update_noise_estimate(self, current_spectrum: np.ndarray, is_speech: bool):
        """Update noise estimate adaptively"""
        try:
            if self.noise_spectrum is None:
                self.noise_spectrum = current_spectrum.copy()
                return
            
            if not is_speech:
                # Update noise spectrum during non-speech periods
                self.noise_spectrum = (
                    (1 - self.adaptation_rate) * self.noise_spectrum +
                    self.adaptation_rate * current_spectrum
                )
            
            # Ensure minimum noise floor
            self.noise_spectrum = np.maximum(self.noise_spectrum, self.noise_floor)
            
        except Exception as e:
            logger.error(f"Error updating noise estimate: {e}")
    
    def compute_wiener_gain(self, signal_spectrum: np.ndarray) -> np.ndarray:
        """Compute Wiener filter gain"""
        try:
            if self.noise_spectrum is None:
                return np.ones_like(signal_spectrum)
            
            # Estimate signal power
            signal_power = signal_spectrum ** 2
            noise_power = self.noise_spectrum
            
            # Compute SNR
            snr = signal_power / (noise_power + self.noise_floor)
            
            # Wiener gain formula
            gain = snr / (snr + 1)
            
            # Apply gain limits
            gain = np.clip(gain, self.min_gain, self.max_gain)
            
            return gain
            
        except Exception as e:
            logger.error(f"Error computing Wiener gain: {e}")
            return np.ones_like(signal_spectrum)
    
    def filter_frame(self, frame_stft: np.ndarray, is_speech: bool) -> np.ndarray:
        """Filter a single STFT frame"""
        try:
            magnitude = np.abs(frame_stft)
            phase = np.angle(frame_stft)
            
            # Update noise estimate
            self.update_noise_estimate(magnitude ** 2, is_speech)
            
            # Compute and apply Wiener gain
            gain = self.compute_wiener_gain(magnitude)
            filtered_magnitude = magnitude * gain
            
            # Reconstruct complex spectrum
            filtered_stft = filtered_magnitude * np.exp(1j * phase)
            
            return filtered_stft
            
        except Exception as e:
            logger.error(f"Error filtering frame: {e}")
            return frame_stft

class VoiceActivityDetector:
    """Advanced voice activity detection"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        
        # WebRTC VAD
        self.webrtc_vad = None
        if webrtcvad:
            self.webrtc_vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # Energy-based VAD parameters
        self.energy_threshold = None
        self.energy_history = deque(maxlen=50)
        
        # Spectral VAD parameters
        self.spectral_centroid_threshold = 2000
        self.zero_crossing_threshold = 0.3
    
    def detect_voice_activity(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[np.ndarray, float]:
        """Detect voice activity in audio"""
        try:
            frame_length = int(sample_rate * self.frame_duration / 1000)
            hop_length = frame_length // 2
            
            # Resample if necessary
            if sample_rate != self.sample_rate and self.webrtc_vad:
                audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            else:
                audio_resampled = audio
                
            vad_results = []
            
            # Frame-wise processing
            for i in range(0, len(audio_resampled) - frame_length, hop_length):
                frame = audio_resampled[i:i + frame_length]
                
                # Multiple VAD methods
                vad_scores = []
                
                # WebRTC VAD
                if self.webrtc_vad and len(frame) == frame_length:
                    try:
                        frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                        webrtc_result = self.webrtc_vad.is_speech(frame_bytes, self.sample_rate)
                        vad_scores.append(float(webrtc_result))
                    except:
                        pass
                
                # Energy-based VAD
                energy_score = self._energy_based_vad(frame)
                vad_scores.append(energy_score)
                
                # Spectral VAD
                spectral_score = self._spectral_based_vad(frame, sample_rate)
                vad_scores.append(spectral_score)
                
                # Combine scores
                if vad_scores:
                    combined_score = np.mean(vad_scores)
                    vad_results.append(combined_score)
                else:
                    vad_results.append(0.5)
            
            # Convert to numpy array
            vad_array = np.array(vad_results)
            
            # Apply smoothing
            vad_smoothed = scipy.ndimage.median_filter(vad_array, size=3)
            
            # Calculate overall speech ratio
            speech_ratio = np.mean(vad_smoothed > 0.5)
            
            return vad_smoothed, speech_ratio
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return np.ones(10) * 0.5, 0.5
    
    def _energy_based_vad(self, frame: np.ndarray) -> float:
        """Energy-based voice activity detection"""
        try:
            # Calculate frame energy
            energy = np.sum(frame ** 2)
            
            # Update energy history
            self.energy_history.append(energy)
            
            # Adaptive threshold
            if len(self.energy_history) >= 10:
                if self.energy_threshold is None:
                    self.energy_threshold = np.percentile(list(self.energy_history), 60)
                else:
                    # Slowly adapt threshold
                    current_percentile = np.percentile(list(self.energy_history), 60)
                    self.energy_threshold = 0.95 * self.energy_threshold + 0.05 * current_percentile
                
                # Score based on threshold
                if energy > self.energy_threshold * 2:
                    return 1.0
                elif energy > self.energy_threshold:
                    return (energy - self.energy_threshold) / self.energy_threshold
                else:
                    return 0.0
            else:
                return 0.5  # Unknown when not enough history
                
        except Exception as e:
            logger.error(f"Error in energy-based VAD: {e}")
            return 0.5
    
    def _spectral_based_vad(self, frame: np.ndarray, sample_rate: int) -> float:
        """Spectral-based voice activity detection"""
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sample_rate)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(frame)[0]
            avg_zcr = np.mean(zcr)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sample_rate)[0]
            avg_rolloff = np.mean(spectral_rolloff)
            
            # Score based on speech characteristics
            centroid_score = 1.0 if avg_centroid > self.spectral_centroid_threshold else avg_centroid / self.spectral_centroid_threshold
            zcr_score = 1.0 if avg_zcr < self.zero_crossing_threshold else (1 - avg_zcr)
            
            # Combine spectral features
            spectral_score = (centroid_score + zcr_score) / 2
            
            return min(1.0, spectral_score)
            
        except Exception as e:
            logger.error(f"Error in spectral-based VAD: {e}")
            return 0.5

class BackgroundNoiseIntelligenceEngine:
    """Main engine for background noise intelligence"""
    
    def __init__(
        self,
        models_dir: str = "models/noise_intelligence",
        sample_rate: int = 16000,
        device: str = "auto"
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Neural models
        self.spectral_subtraction_model = None
        
        # Classical filters
        self.wiener_filter = AdaptiveWienerFilter()
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        # Noise profiles
        self.noise_profiles: Dict[str, NoiseProfile] = {}
        self.active_noise_profile = None
        
        # Processing parameters
        self.frame_size = 512
        self.hop_length = 256
        self.overlap = 0.75
        
        # Real-time processing
        self.real_time_buffer = deque(maxlen=1000)
        self.processing_thread = None
        self.processing_active = False
        
        logger.info(f"NoiseIntelligenceEngine initialized on device: {self.device}")
    
    async def initialize(self):
        """Initialize noise intelligence models"""
        try:
            logger.info("Initializing noise intelligence models...")
            
            # Initialize neural models
            self.spectral_subtraction_model = SpectralSubtractionModule().to(self.device)
            
            # Load pre-trained models
            await self._load_pretrained_models()
            
            # Load noise profiles
            await self._load_noise_profiles()
            
            # Create default noise profiles if none exist
            if not self.noise_profiles:
                await self._create_default_noise_profiles()
            
            logger.info("Noise intelligence models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing noise intelligence engine: {e}")
            raise
    
    async def analyze_noise_environment(
        self,
        audio: np.ndarray,
        sample_rate: int = None
    ) -> Dict[str, Any]:
        """Analyze background noise environment"""
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            logger.info("Analyzing noise environment...")
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                sample_rate = self.sample_rate
            
            # Voice activity detection
            vad_results, speech_ratio = self.vad.detect_voice_activity(audio, sample_rate)
            
            # Extract noise segments (where VAD indicates no speech)
            noise_segments = []
            for i, is_speech in enumerate(vad_results):
                if is_speech < 0.3:  # Low speech probability
                    start_sample = i * self.hop_length
                    end_sample = min((i + 1) * self.hop_length, len(audio))
                    noise_segments.append(audio[start_sample:end_sample])
            
            if noise_segments:
                noise_audio = np.concatenate(noise_segments)
            else:
                # Use entire audio if no clear noise segments
                noise_audio = audio
            
            # Analyze noise characteristics
            noise_analysis = await self._analyze_noise_characteristics(noise_audio, sample_rate)
            
            # Classify noise type
            noise_type, confidence = await self._classify_noise_type(noise_analysis)
            
            # Estimate noise level
            noise_level = await self._estimate_noise_level(noise_audio)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_audio_quality(audio, sample_rate)
            
            analysis_result = {
                'noise_type': noise_type,
                'confidence': confidence,
                'noise_level_db': noise_level,
                'speech_ratio': speech_ratio,
                'quality_metrics': asdict(quality_metrics),
                'noise_characteristics': noise_analysis,
                'recommended_processing': self._recommend_processing(noise_type, noise_level),
                'vad_results': vad_results.tolist()
            }
            
            logger.info(f"Noise environment analyzed: {noise_type} (confidence: {confidence:.3f})")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing noise environment: {e}")
            raise
    
    async def reduce_background_noise(
        self,
        request: NoiseReductionRequest
    ) -> NoiseReductionResult:
        """Reduce background noise from audio"""
        try:
            logger.info(f"Reducing background noise (mode: {request.adaptation_mode})")
            
            start_time = datetime.now()
            
            # Resample if necessary
            audio = request.audio_data
            if request.sample_rate != self.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=request.sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Analyze environment if noise type not provided
            if request.noise_type is None:
                env_analysis = await self.analyze_noise_environment(audio)
                noise_type = env_analysis['noise_type']
                confidence = env_analysis['confidence']
            else:
                noise_type = request.noise_type
                confidence = 0.8
            
            # Select appropriate noise profile
            noise_profile = self.noise_profiles.get(noise_type)
            if not noise_profile:
                noise_profile = self.noise_profiles.get('general', list(self.noise_profiles.values())[0])
            
            # Apply noise reduction based on mode and profile
            if request.quality_target == "quality" and self.spectral_subtraction_model:
                cleaned_audio, artifacts = await self._neural_noise_reduction(
                    audio, noise_profile, request
                )
            else:
                cleaned_audio, artifacts = await self._classical_noise_reduction(
                    audio, noise_profile, request
                )
            
            # Extract noise component
            noise_removed = audio - cleaned_audio
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_audio_quality(cleaned_audio, self.sample_rate)
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = NoiseReductionResult(
                cleaned_audio=cleaned_audio,
                noise_removed=noise_removed,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                noise_profile_used=noise_profile.noise_id,
                confidence_score=confidence,
                processing_artifacts=artifacts
            )
            
            # Update noise profile statistics
            await self._update_noise_profile_stats(noise_profile, quality_metrics)
            
            logger.info(f"Noise reduction completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reducing background noise: {e}")
            raise
    
    async def start_real_time_processing(
        self,
        callback_func,
        noise_type: str = "auto",
        adaptation_mode: str = "balanced"
    ):
        """Start real-time noise reduction processing"""
        try:
            logger.info("Starting real-time noise processing...")
            
            self.processing_active = True
            
            # Initialize audio stream (would use pyaudio or similar)
            # This is a placeholder for real-time audio processing
            
            def processing_worker():
                while self.processing_active:
                    try:
                        # Get audio from buffer
                        if len(self.real_time_buffer) > 0:
                            audio_chunk = np.array(list(self.real_time_buffer))
                            self.real_time_buffer.clear()
                            
                            # Process chunk
                            request = NoiseReductionRequest(
                                audio_data=audio_chunk,
                                sample_rate=self.sample_rate,
                                noise_type=noise_type if noise_type != "auto" else None,
                                adaptation_mode=adaptation_mode,
                                real_time=True,
                                quality_target="speed"
                            )
                            
                            # Reduce noise
                            result = asyncio.run(self.reduce_background_noise(request))
                            
                            # Send to callback
                            callback_func(result.cleaned_audio, result.quality_metrics)
                        
                        # Small delay to prevent excessive CPU usage
                        threading.sleep(0.01)
                        
                    except Exception as e:
                        logger.error(f"Error in real-time processing: {e}")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=processing_worker)
            self.processing_thread.start()
            
            logger.info("Real-time processing started")
            
        except Exception as e:
            logger.error(f"Error starting real-time processing: {e}")
            raise
    
    async def stop_real_time_processing(self):
        """Stop real-time processing"""
        try:
            self.processing_active = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            logger.info("Real-time processing stopped")
            
        except Exception as e:
            logger.error(f"Error stopping real-time processing: {e}")
    
    async def learn_noise_profile(
        self,
        noise_audio: np.ndarray,
        noise_type: str,
        description: str = "",
        sample_rate: int = None
    ) -> NoiseProfile:
        """Learn a new noise profile from sample audio"""
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            logger.info(f"Learning noise profile for: {noise_type}")
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                noise_audio = librosa.resample(
                    noise_audio, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Analyze noise characteristics
            characteristics = await self._analyze_noise_characteristics(noise_audio, self.sample_rate)
            
            # Create noise profile
            noise_id = f"{noise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            noise_profile = NoiseProfile(
                noise_id=noise_id,
                noise_type=noise_type,
                description=description or f"Learned profile for {noise_type}",
                frequency_bands=characteristics['frequency_bands'],
                spectral_centroid=characteristics['spectral_centroid'],
                spectral_rolloff=characteristics['spectral_rolloff'],
                spectral_bandwidth=characteristics['spectral_bandwidth'],
                stationarity=characteristics['stationarity'],
                periodicity=characteristics['periodicity'],
                onset_rate=characteristics['onset_rate'],
                mean_power=characteristics['mean_power'],
                variance=characteristics['variance'],
                snr_threshold=characteristics['snr_threshold'],
                filter_type="adaptive",
                adaptation_rate=0.1,
                learning_enabled=True,
                success_rate=0.8,  # Initial estimate
                average_improvement=5.0,  # dB
                last_updated=datetime.now(),
                usage_count=0
            )
            
            # Store noise profile
            self.noise_profiles[noise_id] = noise_profile
            await self._save_noise_profile(noise_profile)
            
            logger.info(f"Noise profile learned: {noise_id}")
            
            return noise_profile
            
        except Exception as e:
            logger.error(f"Error learning noise profile: {e}")
            raise
    
    async def get_noise_profiles(self) -> List[Dict[str, Any]]:
        """Get available noise profiles"""
        profiles = []
        for profile in self.noise_profiles.values():
            profiles.append({
                'noise_id': profile.noise_id,
                'noise_type': profile.noise_type,
                'description': profile.description,
                'success_rate': profile.success_rate,
                'average_improvement': profile.average_improvement,
                'usage_count': profile.usage_count,
                'last_updated': profile.last_updated.isoformat()
            })
        
        return sorted(profiles, key=lambda x: x['success_rate'], reverse=True)
    
    # Private methods for processing
    
    async def _analyze_noise_characteristics(
        self,
        noise_audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze characteristics of noise audio"""
        try:
            characteristics = {}
            
            # Spectral analysis
            stft = librosa.stft(noise_audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            power_spectrum = magnitude ** 2
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=noise_audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=noise_audio, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=noise_audio, sr=sample_rate)[0]
            
            characteristics['spectral_centroid'] = float(np.mean(spectral_centroid))
            characteristics['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            characteristics['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            # Dominant frequency bands
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=self.frame_size)
            avg_power = np.mean(power_spectrum, axis=1)
            
            # Find peaks in spectrum
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_power, height=np.max(avg_power) * 0.1)
            
            frequency_bands = []
            for peak in peaks[:5]:  # Top 5 peaks
                freq = freq_bins[peak]
                bandwidth = sample_rate / self.frame_size  # Frequency resolution
                frequency_bands.append((float(freq - bandwidth), float(freq + bandwidth)))
            
            characteristics['frequency_bands'] = frequency_bands
            
            # Temporal analysis
            # Stationarity (variance of spectral features over time)
            centroid_var = np.var(spectral_centroid)
            stationarity = 1.0 / (1.0 + centroid_var / 1000)  # Normalize
            characteristics['stationarity'] = float(stationarity)
            
            # Periodicity (autocorrelation analysis)
            autocorr = np.correlate(noise_audio, noise_audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find periodic patterns
            if len(autocorr) > sample_rate:
                # Look for peaks in autocorrelation
                peaks, _ = find_peaks(autocorr[:sample_rate], height=np.max(autocorr) * 0.3)
                periodicity = len(peaks) / len(autocorr[:sample_rate])
            else:
                periodicity = 0.1
            
            characteristics['periodicity'] = float(periodicity)
            
            # Onset rate (events per second)
            onset_frames = librosa.onset.onset_detect(y=noise_audio, sr=sample_rate)
            duration = len(noise_audio) / sample_rate
            onset_rate = len(onset_frames) / duration if duration > 0 else 0
            characteristics['onset_rate'] = float(onset_rate)
            
            # Statistical properties
            characteristics['mean_power'] = float(np.mean(noise_audio ** 2))
            characteristics['variance'] = float(np.var(noise_audio))
            
            # SNR threshold (estimated based on noise level)
            noise_level = 20 * np.log10(np.sqrt(np.mean(noise_audio ** 2)) + 1e-10)
            characteristics['snr_threshold'] = max(0, float(20 - noise_level))  # Minimum SNR needed
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing noise characteristics: {e}")
            return {
                'frequency_bands': [(100, 500), (500, 2000), (2000, 8000)],
                'spectral_centroid': 1500.0,
                'spectral_rolloff': 3000.0,
                'spectral_bandwidth': 1000.0,
                'stationarity': 0.7,
                'periodicity': 0.1,
                'onset_rate': 2.0,
                'mean_power': 0.01,
                'variance': 0.001,
                'snr_threshold': 10.0
            }
    
    async def _classify_noise_type(
        self,
        noise_analysis: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Classify noise type based on characteristics"""
        try:
            # Simple rule-based classification
            # In practice, this would use a trained classifier
            
            spectral_centroid = noise_analysis['spectral_centroid']
            periodicity = noise_analysis['periodicity']
            onset_rate = noise_analysis['onset_rate']
            stationarity = noise_analysis['stationarity']
            
            # Traffic noise: low frequency, stationary
            if spectral_centroid < 800 and stationarity > 0.8:
                return "traffic", 0.85
            
            # Office noise: mid frequency, moderate stationarity
            elif 800 <= spectral_centroid <= 2000 and 0.4 <= stationarity <= 0.8:
                return "office", 0.75
            
            # Restaurant/cafe: higher onset rate, variable
            elif onset_rate > 3.0 and stationarity < 0.6:
                return "restaurant", 0.7
            
            # Wind noise: high frequency, low stationarity
            elif spectral_centroid > 2000 and stationarity < 0.4:
                return "wind", 0.8
            
            # Construction: high onset rate, periodic
            elif onset_rate > 5.0 and periodicity > 0.3:
                return "construction", 0.8
            
            # Air conditioning: periodic, mid frequency
            elif periodicity > 0.5 and 500 <= spectral_centroid <= 1500:
                return "air_conditioning", 0.9
            
            # Default: general noise
            else:
                return "general", 0.6
                
        except Exception as e:
            logger.error(f"Error classifying noise type: {e}")
            return "general", 0.5
    
    async def _estimate_noise_level(self, noise_audio: np.ndarray) -> float:
        """Estimate noise level in dB"""
        try:
            # RMS to dB conversion
            rms = np.sqrt(np.mean(noise_audio ** 2))
            noise_level_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
            
            # Reference to typical speech level (around -20 dB digital)
            noise_level_db += 60  # Adjust reference
            
            return float(noise_level_db)
            
        except Exception as e:
            logger.error(f"Error estimating noise level: {e}")
            return 40.0  # Default moderate noise level
    
    async def _calculate_audio_quality(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> AudioQualityMetrics:
        """Calculate comprehensive audio quality metrics"""
        try:
            # Basic SNR estimation (simplified)
            # Assume first 10% is noise, rest includes speech
            noise_segment = audio[:len(audio)//10]
            signal_segment = audio[len(audio)//10:]
            
            noise_power = np.mean(noise_segment ** 2)
            signal_power = np.mean(signal_segment ** 2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 60.0  # Very clean signal
            
            # Speech quality score (0-100)
            # Based on SNR and spectral characteristics
            if snr_db > 20:
                speech_quality = 95
            elif snr_db > 15:
                speech_quality = 85
            elif snr_db > 10:
                speech_quality = 70
            elif snr_db > 5:
                speech_quality = 55
            else:
                speech_quality = 30
            
            # Intelligibility score
            # Based on spectral clarity and SNR
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            if 1000 <= avg_centroid <= 3000 and snr_db > 10:
                intelligibility = min(100, speech_quality + 5)
            else:
                intelligibility = max(0, speech_quality - 10)
            
            # Noise and speech levels
            noise_level = 20 * np.log10(np.sqrt(noise_power) + 1e-10) + 60
            speech_level = 20 * np.log10(np.sqrt(signal_power) + 1e-10) + 60
            
            # PESQ and STOI (if libraries available)
            pesq_score = None
            stoi_score = None
            
            if pesq is not None and len(audio) >= sample_rate:
                try:
                    # Would need clean reference signal for PESQ
                    # pesq_score = pesq(sample_rate, reference_audio, audio, 'wb')
                    pass
                except:
                    pass
            
            if stoi is not None and len(audio) >= sample_rate:
                try:
                    # Would need clean reference signal for STOI
                    # stoi_score = stoi(reference_audio, audio, sample_rate, extended=False)
                    pass
                except:
                    pass
            
            # Voice activity detection accuracy (simplified)
            vad_results, speech_ratio = self.vad.detect_voice_activity(audio, sample_rate)
            vad_accuracy = speech_ratio * 100  # Simplified metric
            
            return AudioQualityMetrics(
                snr_db=float(snr_db),
                speech_quality_score=float(speech_quality),
                intelligibility_score=float(intelligibility),
                noise_level=float(noise_level),
                speech_level=float(speech_level),
                pesq_score=pesq_score,
                stoi_score=stoi_score,
                spectral_distortion=0.0,  # Would calculate if reference available
                harmonic_distortion=0.0,
                speech_pause_ratio=1.0 - speech_ratio,
                voice_activity_detection_accuracy=float(vad_accuracy)
            )
            
        except Exception as e:
            logger.error(f"Error calculating audio quality: {e}")
            return AudioQualityMetrics(
                snr_db=10.0,
                speech_quality_score=70.0,
                intelligibility_score=70.0,
                noise_level=40.0,
                speech_level=50.0
            )
    
    def _recommend_processing(self, noise_type: str, noise_level: float) -> Dict[str, Any]:
        """Recommend processing parameters based on noise analysis"""
        recommendations = {
            'filter_type': 'adaptive',
            'adaptation_mode': 'balanced',
            'aggressiveness': 'moderate'
        }
        
        # Adjust based on noise type
        if noise_type == 'traffic':
            recommendations['filter_type'] = 'low_pass'
            recommendations['aggressiveness'] = 'aggressive'
        elif noise_type == 'wind':
            recommendations['filter_type'] = 'high_pass'
            recommendations['aggressiveness'] = 'conservative'
        elif noise_type == 'restaurant':
            recommendations['filter_type'] = 'spectral_subtraction'
            recommendations['adaptation_mode'] = 'aggressive'
        
        # Adjust based on noise level
        if noise_level > 60:
            recommendations['aggressiveness'] = 'aggressive'
        elif noise_level < 40:
            recommendations['aggressiveness'] = 'conservative'
        
        return recommendations
    
    async def _neural_noise_reduction(
        self,
        audio: np.ndarray,
        noise_profile: NoiseProfile,
        request: NoiseReductionRequest
    ) -> Tuple[np.ndarray, List[str]]:
        """Neural-based noise reduction"""
        try:
            if not self.spectral_subtraction_model:
                return await self._classical_noise_reduction(audio, noise_profile, request)
            
            artifacts = []
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=80,
                hop_length=self.hop_length,
                n_fft=self.frame_size
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).to(self.device)
            
            # Apply neural denoising
            self.spectral_subtraction_model.eval()
            with torch.no_grad():
                enhanced_mel, speech_mask, noise_mask = self.spectral_subtraction_model(mel_tensor)
            
            # Convert back to audio
            enhanced_mel_np = enhanced_mel.cpu().numpy().squeeze()
            
            # Denormalize
            enhanced_mel_denorm = enhanced_mel_np * np.std(mel_spec_db) + np.mean(mel_spec_db)
            
            # Convert to linear scale
            enhanced_mel_linear = librosa.db_to_power(enhanced_mel_denorm)
            
            # Inverse mel spectrogram
            enhanced_audio = librosa.feature.inverse.mel_to_audio(
                enhanced_mel_linear,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.frame_size
            )
            
            # Ensure same length as input
            if len(enhanced_audio) > len(audio):
                enhanced_audio = enhanced_audio[:len(audio)]
            elif len(enhanced_audio) < len(audio):
                enhanced_audio = np.pad(enhanced_audio, (0, len(audio) - len(enhanced_audio)))
            
            # Check for artifacts
            if np.max(np.abs(enhanced_audio)) > 1.0:
                enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
                artifacts.append("amplitude_clipping")
            
            return enhanced_audio, artifacts
            
        except Exception as e:
            logger.error(f"Error in neural noise reduction: {e}")
            return await self._classical_noise_reduction(audio, noise_profile, request)
    
    async def _classical_noise_reduction(
        self,
        audio: np.ndarray,
        noise_profile: NoiseProfile,
        request: NoiseReductionRequest
    ) -> Tuple[np.ndarray, List[str]]:
        """Classical signal processing noise reduction"""
        try:
            artifacts = []
            
            # Voice activity detection
            vad_results, _ = self.vad.detect_voice_activity(audio, self.sample_rate)
            
            # STFT
            stft = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Initialize Wiener filter if needed
            if self.wiener_filter.noise_spectrum is None:
                # Estimate noise from non-speech segments
                noise_frames = []
                for i, is_speech in enumerate(vad_results):
                    if i < magnitude.shape[1] and is_speech < 0.3:
                        noise_frames.append(magnitude[:, i] ** 2)
                
                if noise_frames:
                    noise_spectrum = np.mean(noise_frames, axis=0)
                    self.wiener_filter.noise_spectrum = noise_spectrum
            
            # Apply filter frame by frame
            filtered_stft = np.zeros_like(stft, dtype=complex)
            
            for i in range(stft.shape[1]):
                is_speech = vad_results[i] > 0.5 if i < len(vad_results) else True
                filtered_stft[:, i] = self.wiener_filter.filter_frame(stft[:, i], is_speech)
            
            # Inverse STFT
            enhanced_audio = librosa.istft(
                filtered_stft, 
                hop_length=self.hop_length,
                length=len(audio)
            )
            
            # Apply additional processing based on adaptation mode
            if request.adaptation_mode == "aggressive":
                # Apply more aggressive noise reduction
                if nr:
                    try:
                        enhanced_audio = nr.reduce_noise(y=enhanced_audio, sr=self.sample_rate)
                    except:
                        pass
                
                # Low-pass filter for very noisy environments
                if noise_profile.noise_type in ['traffic', 'construction']:
                    sos = butter(6, 4000, btype='low', fs=self.sample_rate, output='sos')
                    enhanced_audio = filtfilt(sos, enhanced_audio)
                    artifacts.append("low_pass_filtering")
            
            elif request.adaptation_mode == "conservative":
                # Gentle enhancement
                alpha = 0.7  # Blend with original
                enhanced_audio = alpha * enhanced_audio + (1 - alpha) * audio
                artifacts.append("conservative_blending")
            
            # Normalization
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 0.95:
                enhanced_audio = enhanced_audio * (0.95 / max_val)
                artifacts.append("amplitude_normalization")
            
            return enhanced_audio, artifacts
            
        except Exception as e:
            logger.error(f"Error in classical noise reduction: {e}")
            # Return original audio if processing fails
            return audio, ["processing_failed"]
    
    async def _update_noise_profile_stats(
        self,
        noise_profile: NoiseProfile,
        quality_metrics: AudioQualityMetrics
    ):
        """Update noise profile performance statistics"""
        try:
            # Simple performance tracking
            noise_profile.usage_count += 1
            
            # Update success rate based on quality improvement
            if quality_metrics.snr_db > 10:
                current_success = 1.0
            elif quality_metrics.snr_db > 5:
                current_success = 0.7
            else:
                current_success = 0.3
            
            # Weighted average update
            weight = 1.0 / noise_profile.usage_count
            noise_profile.success_rate = (
                (1 - weight) * noise_profile.success_rate + weight * current_success
            )
            
            noise_profile.last_updated = datetime.now()
            
            # Save updated profile
            await self._save_noise_profile(noise_profile)
            
        except Exception as e:
            logger.error(f"Error updating noise profile stats: {e}")
    
    # Model and profile management
    
    async def _load_pretrained_models(self):
        """Load pre-trained models"""
        try:
            model_path = self.models_dir / "spectral_subtraction.pth"
            if model_path.exists() and self.spectral_subtraction_model:
                self.spectral_subtraction_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained spectral subtraction model")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    async def _load_noise_profiles(self):
        """Load noise profiles from storage"""
        try:
            profiles_file = self.models_dir / "noise_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for profile_data in profiles_data:
                    profile_data['last_updated'] = datetime.fromisoformat(
                        profile_data['last_updated']
                    )
                    profile = NoiseProfile(**profile_data)
                    self.noise_profiles[profile.noise_id] = profile
                
                logger.info(f"Loaded {len(self.noise_profiles)} noise profiles")
                
        except Exception as e:
            logger.warning(f"Could not load noise profiles: {e}")
    
    async def _create_default_noise_profiles(self):
        """Create default noise profiles"""
        try:
            default_profiles = [
                {
                    'noise_id': 'general',
                    'noise_type': 'general',
                    'description': 'General background noise',
                    'frequency_bands': [(100, 500), (500, 2000), (2000, 8000)],
                    'spectral_centroid': 1500.0,
                    'spectral_rolloff': 3000.0,
                    'spectral_bandwidth': 1000.0,
                    'stationarity': 0.6,
                    'periodicity': 0.2,
                    'onset_rate': 2.0,
                    'mean_power': 0.01,
                    'variance': 0.005,
                    'snr_threshold': 5.0,
                    'filter_type': 'adaptive',
                    'adaptation_rate': 0.1,
                    'learning_enabled': True,
                    'success_rate': 0.8,
                    'average_improvement': 5.0,
                    'last_updated': datetime.now(),
                    'usage_count': 0
                },
                {
                    'noise_id': 'traffic',
                    'noise_type': 'traffic',
                    'description': 'Vehicle and road traffic noise',
                    'frequency_bands': [(50, 300), (300, 800), (800, 2000)],
                    'spectral_centroid': 600.0,
                    'spectral_rolloff': 1200.0,
                    'spectral_bandwidth': 800.0,
                    'stationarity': 0.9,
                    'periodicity': 0.1,
                    'onset_rate': 0.5,
                    'mean_power': 0.02,
                    'variance': 0.003,
                    'snr_threshold': 8.0,
                    'filter_type': 'high_pass',
                    'adaptation_rate': 0.05,
                    'learning_enabled': True,
                    'success_rate': 0.85,
                    'average_improvement': 7.0,
                    'last_updated': datetime.now(),
                    'usage_count': 0
                },
                {
                    'noise_id': 'office',
                    'noise_type': 'office',
                    'description': 'Office environment noise',
                    'frequency_bands': [(200, 800), (800, 2500), (2500, 5000)],
                    'spectral_centroid': 1200.0,
                    'spectral_rolloff': 2800.0,
                    'spectral_bandwidth': 1200.0,
                    'stationarity': 0.7,
                    'periodicity': 0.3,
                    'onset_rate': 3.0,
                    'mean_power': 0.008,
                    'variance': 0.004,
                    'snr_threshold': 6.0,
                    'filter_type': 'spectral_subtraction',
                    'adaptation_rate': 0.1,
                    'learning_enabled': True,
                    'success_rate': 0.82,
                    'average_improvement': 6.0,
                    'last_updated': datetime.now(),
                    'usage_count': 0
                }
            ]
            
            for profile_data in default_profiles:
                profile = NoiseProfile(**profile_data)
                self.noise_profiles[profile.noise_id] = profile
            
            await self._save_all_noise_profiles()
            logger.info("Created default noise profiles")
            
        except Exception as e:
            logger.error(f"Error creating default noise profiles: {e}")
    
    async def _save_noise_profile(self, profile: NoiseProfile):
        """Save individual noise profile"""
        await self._save_all_noise_profiles()
    
    async def _save_all_noise_profiles(self):
        """Save all noise profiles to storage"""
        try:
            profiles_data = []
            for profile in self.noise_profiles.values():
                profile_dict = asdict(profile)
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                profiles_data.append(profile_dict)
            
            profiles_file = self.models_dir / "noise_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving noise profiles: {e}")


# Global instance
background_noise_intelligence_engine = BackgroundNoiseIntelligenceEngine()