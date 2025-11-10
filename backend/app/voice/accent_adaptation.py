"""
Accent & Regional Adaptation System
AI voice adaptation to match prospect's regional accent patterns
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import librosa
import scipy.signal
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re

# Phonetic analysis imports
try:
    import phonemizer
    from g2p_en import G2p
    import epitran
    from montreal_forced_alignment import pretrained
    import parselmouth
    import textgrid
except ImportError:
    phonemizer = None
    G2p = None
    epitran = None
    pretrained = None
    parselmouth = None
    textgrid = None

logger = logging.getLogger(__name__)

@dataclass
class AccentProfile:
    """Accent profile for regional adaptation"""
    accent_id: str
    region: str
    country: str
    dialect_name: str
    
    # Phonetic characteristics
    vowel_shifts: Dict[str, str]  # IPA vowel mappings
    consonant_variations: Dict[str, str]  # Consonant variations
    r_coloring: str  # rhotic vs non-rhotic
    th_pronunciation: str  # θ/ð variations
    
    # Prosodic features
    intonation_pattern: str  # rising, falling, flat
    stress_patterns: Dict[str, List[int]]  # Word stress patterns
    rhythm_type: str  # stress-timed vs syllable-timed
    speech_rate: float  # Average syllables per second
    
    # Regional markers
    common_phrases: List[str]
    vocabulary_variations: Dict[str, str]
    grammatical_features: List[str]
    
    # Audio features
    formant_ratios: List[float]  # F1/F2/F3 ratios
    fundamental_frequency_range: Tuple[float, float]
    voice_quality: str  # creaky, breathy, modal
    
    # Confidence and usage
    confidence_score: float
    sample_count: int
    last_updated: datetime

@dataclass
class PhoneticAnalysis:
    """Phonetic analysis result"""
    phonemes: List[str]  # IPA phonemes
    syllables: List[str]
    stress_pattern: List[int]  # 0=unstressed, 1=primary, 2=secondary
    vowel_qualities: List[Tuple[float, float]]  # F1, F2 values
    consonant_features: Dict[str, Any]
    duration_ratios: List[float]
    pitch_contour: np.ndarray

@dataclass
class AccentAdaptationRequest:
    """Request for accent adaptation"""
    text: str
    target_accent: str
    source_accent: str = "general_american"
    adaptation_strength: float = 0.8  # 0.0 to 1.0
    preserve_meaning: bool = True
    adapt_vocabulary: bool = True
    adapt_grammar: bool = False
    target_formality: str = "neutral"  # formal, neutral, casual

@dataclass
class AccentAdaptationResult:
    """Result of accent adaptation"""
    adapted_text: str
    phonetic_changes: List[Dict[str, Any]]
    prosodic_adjustments: Dict[str, Any]
    vocabulary_changes: Dict[str, str]
    confidence_score: float
    adaptation_time: float
    audio_parameters: Dict[str, Any]

class AccentClassifier(nn.Module):
    """Neural network for accent classification"""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_accents=50):
        super().__init__()
        
        # Audio feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Phonetic feature encoder
        self.phonetic_encoder = nn.Sequential(
            nn.Linear(40, 128),  # Phonetic features
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Prosodic feature encoder
        self.prosodic_encoder = nn.Sequential(
            nn.Linear(20, 64),  # Prosodic features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_accents)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
    
    def forward(self, audio_features, phonetic_features, prosodic_features):
        # Encode audio features
        audio_encoded = self.feature_encoder(audio_features.transpose(1, 2))
        audio_encoded = audio_encoded.squeeze(-1)
        
        # Encode phonetic features
        phonetic_encoded = self.phonetic_encoder(phonetic_features)
        
        # Encode prosodic features
        prosodic_encoded = self.prosodic_encoder(prosodic_features)
        
        # Combine features
        combined = torch.cat([audio_encoded, phonetic_encoded, prosodic_encoded], dim=-1)
        
        # Classify accent
        accent_logits = self.classifier(combined)
        
        return accent_logits, combined

class PhoneticTransformer(nn.Module):
    """Transformer for phonetic adaptation"""
    
    def __init__(self, vocab_size=500, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Accent conditioning
        self.accent_embedding = nn.Embedding(50, embed_dim)  # 50 accent types
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, phoneme_sequence, accent_id, sequence_length=None):
        seq_len = phoneme_sequence.size(1)
        
        # Embed phonemes and add positional encoding
        phoneme_emb = self.embedding(phoneme_sequence)
        phoneme_emb += self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Add accent conditioning
        accent_emb = self.accent_embedding(accent_id).unsqueeze(1)
        phoneme_emb = phoneme_emb + accent_emb
        
        # Transform phonemes
        transformed = self.transformer(phoneme_emb.transpose(0, 1))
        transformed = transformed.transpose(0, 1)
        
        # Project to output vocabulary
        output_logits = self.output_projection(transformed)
        
        return output_logits

class RegionalAdaptationEngine:
    """Engine for regional accent adaptation"""
    
    def __init__(self, models_dir: str = "models/accent_adaptation", device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Models
        self.accent_classifier = None
        self.phonetic_transformer = None
        self.g2p = None
        
        # Accent profiles
        self.accent_profiles: Dict[str, AccentProfile] = {}
        self.phoneme_mappings: Dict[str, Dict[str, str]] = {}
        
        # Analysis tools
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        
        # Regional databases
        self.vowel_systems = self._initialize_vowel_systems()
        self.consonant_systems = self._initialize_consonant_systems()
        self.prosodic_patterns = self._initialize_prosodic_patterns()
        
        logger.info(f"RegionalAdaptationEngine initialized on device: {self.device}")
    
    async def initialize(self):
        """Initialize accent adaptation models"""
        try:
            logger.info("Initializing accent adaptation models...")
            
            # Initialize neural models
            self.accent_classifier = AccentClassifier().to(self.device)
            self.phonetic_transformer = PhoneticTransformer().to(self.device)
            
            # Initialize G2P (grapheme-to-phoneme) converter
            if G2p:
                self.g2p = G2p()
            
            # Load pre-trained models
            await self._load_pretrained_models()
            
            # Load accent profiles
            await self._load_accent_profiles()
            
            # Initialize phoneme mappings
            await self._initialize_phoneme_mappings()
            
            logger.info("Accent adaptation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing accent adaptation engine: {e}")
            raise
    
    async def detect_accent(
        self,
        audio: np.ndarray,
        text: Optional[str] = None,
        sample_rate: int = 22050
    ) -> Tuple[str, float]:
        """Detect accent from audio sample"""
        try:
            logger.info("Detecting accent from audio...")
            
            # Extract audio features
            audio_features = await self._extract_audio_features(audio, sample_rate)
            
            # Extract phonetic features if text is provided
            if text:
                phonetic_features = await self._extract_phonetic_features(audio, text, sample_rate)
            else:
                phonetic_features = np.zeros(40)  # Default phonetic features
            
            # Extract prosodic features
            prosodic_features = await self._extract_prosodic_features(audio, sample_rate)
            
            # Convert to tensors
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
            phonetic_tensor = torch.FloatTensor(phonetic_features).unsqueeze(0).to(self.device)
            prosodic_tensor = torch.FloatTensor(prosodic_features).unsqueeze(0).to(self.device)
            
            # Classify accent
            if self.accent_classifier:
                self.accent_classifier.eval()
                with torch.no_grad():
                    accent_logits, features = self.accent_classifier(
                        audio_tensor, phonetic_tensor, prosodic_tensor
                    )
                    accent_probs = F.softmax(accent_logits, dim=-1)
                    accent_id = torch.argmax(accent_probs, dim=-1).item()
                    confidence = accent_probs[0, accent_id].item()
                
                # Map accent ID to accent name
                accent_name = self._map_accent_id_to_name(accent_id)
                
                logger.info(f"Detected accent: {accent_name} (confidence: {confidence:.3f})")
                
                return accent_name, confidence
            else:
                # Fallback to heuristic detection
                return await self._heuristic_accent_detection(audio, text, sample_rate)
                
        except Exception as e:
            logger.error(f"Error detecting accent: {e}")
            return "general_american", 0.5
    
    async def adapt_accent(
        self,
        request: AccentAdaptationRequest
    ) -> AccentAdaptationResult:
        """Adapt text and speech to target accent"""
        try:
            logger.info(f"Adapting accent from {request.source_accent} to {request.target_accent}")
            
            start_time = datetime.now()
            
            # Get accent profiles
            source_profile = self.accent_profiles.get(request.source_accent)
            target_profile = self.accent_profiles.get(request.target_accent)
            
            if not target_profile:
                raise ValueError(f"Target accent profile not found: {request.target_accent}")
            
            # Perform phonetic adaptation
            phonetic_result = await self._adapt_phonetics(request, source_profile, target_profile)
            
            # Perform prosodic adaptation
            prosodic_result = await self._adapt_prosody(request, target_profile)
            
            # Perform vocabulary adaptation
            vocabulary_result = await self._adapt_vocabulary(request, target_profile)
            
            # Combine adaptations
            adapted_text = vocabulary_result['adapted_text']
            
            # Calculate audio parameters for synthesis
            audio_params = await self._calculate_audio_parameters(target_profile, request)
            
            adaptation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = AccentAdaptationResult(
                adapted_text=adapted_text,
                phonetic_changes=phonetic_result['changes'],
                prosodic_adjustments=prosodic_result,
                vocabulary_changes=vocabulary_result['changes'],
                confidence_score=min(
                    phonetic_result['confidence'],
                    vocabulary_result['confidence']
                ),
                adaptation_time=adaptation_time,
                audio_parameters=audio_params
            )
            
            logger.info(f"Accent adaptation completed in {adaptation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adapting accent: {e}")
            raise
    
    async def learn_accent_from_sample(
        self,
        audio: np.ndarray,
        text: str,
        region: str,
        sample_rate: int = 22050
    ) -> AccentProfile:
        """Learn accent characteristics from audio sample"""
        try:
            logger.info(f"Learning accent characteristics for region: {region}")
            
            # Detect or create accent ID
            accent_id = f"learned_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Analyze phonetic characteristics
            phonetic_analysis = await self._analyze_phonetics(audio, text, sample_rate)
            
            # Analyze prosodic characteristics
            prosodic_analysis = await self._analyze_prosody(audio, sample_rate)
            
            # Extract formant characteristics
            formant_analysis = await self._analyze_formants(audio, sample_rate)
            
            # Create accent profile
            accent_profile = AccentProfile(
                accent_id=accent_id,
                region=region,
                country="Unknown",
                dialect_name=f"Learned {region}",
                vowel_shifts=phonetic_analysis['vowel_shifts'],
                consonant_variations=phonetic_analysis['consonant_variations'],
                r_coloring=phonetic_analysis['r_coloring'],
                th_pronunciation=phonetic_analysis['th_pronunciation'],
                intonation_pattern=prosodic_analysis['intonation'],
                stress_patterns=prosodic_analysis['stress_patterns'],
                rhythm_type=prosodic_analysis['rhythm'],
                speech_rate=prosodic_analysis['speech_rate'],
                common_phrases=[],  # Would be populated over time
                vocabulary_variations={},  # Would be learned over time
                grammatical_features=[],  # Would be learned over time
                formant_ratios=formant_analysis['ratios'],
                fundamental_frequency_range=formant_analysis['f0_range'],
                voice_quality=formant_analysis['quality'],
                confidence_score=0.7,  # Initial confidence
                sample_count=1,
                last_updated=datetime.now()
            )
            
            # Store accent profile
            self.accent_profiles[accent_id] = accent_profile
            await self._save_accent_profile(accent_profile)
            
            logger.info(f"Accent profile learned: {accent_id}")
            
            return accent_profile
            
        except Exception as e:
            logger.error(f"Error learning accent from sample: {e}")
            raise
    
    async def get_supported_accents(self) -> List[Dict[str, Any]]:
        """Get list of supported accents"""
        accents = []
        for profile in self.accent_profiles.values():
            accents.append({
                'accent_id': profile.accent_id,
                'region': profile.region,
                'country': profile.country,
                'dialect_name': profile.dialect_name,
                'confidence_score': profile.confidence_score,
                'sample_count': profile.sample_count
            })
        
        return sorted(accents, key=lambda x: x['confidence_score'], reverse=True)
    
    async def update_accent_profile(
        self,
        accent_id: str,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 22050
    ) -> AccentProfile:
        """Update existing accent profile with new sample"""
        try:
            if accent_id not in self.accent_profiles:
                raise ValueError(f"Accent profile not found: {accent_id}")
            
            profile = self.accent_profiles[accent_id]
            
            # Analyze new sample
            phonetic_analysis = await self._analyze_phonetics(audio, text, sample_rate)
            prosodic_analysis = await self._analyze_prosody(audio, sample_rate)
            formant_analysis = await self._analyze_formants(audio, sample_rate)
            
            # Update profile with weighted average
            weight = 1.0 / (profile.sample_count + 1)
            
            # Update phonetic characteristics
            profile.vowel_shifts.update(phonetic_analysis['vowel_shifts'])
            profile.consonant_variations.update(phonetic_analysis['consonant_variations'])
            
            # Update prosodic characteristics
            profile.speech_rate = (1 - weight) * profile.speech_rate + weight * prosodic_analysis['speech_rate']
            
            # Update formant characteristics
            profile.formant_ratios = [
                (1 - weight) * old + weight * new
                for old, new in zip(profile.formant_ratios, formant_analysis['ratios'])
            ]
            
            # Update metadata
            profile.sample_count += 1
            profile.last_updated = datetime.now()
            profile.confidence_score = min(1.0, profile.confidence_score + 0.1)
            
            # Save updated profile
            await self._save_accent_profile(profile)
            
            logger.info(f"Accent profile updated: {accent_id}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error updating accent profile: {e}")
            raise
    
    # Feature extraction methods
    
    async def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for accent classification"""
        try:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_mels=128, hop_length=self.hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure consistent shape
            target_frames = 200
            if mel_spec_db.shape[1] < target_frames:
                # Pad with zeros
                pad_width = target_frames - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            elif mel_spec_db.shape[1] > target_frames:
                # Truncate
                mel_spec_db = mel_spec_db[:, :target_frames]
            
            return mel_spec_db
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return np.zeros((128, 200))
    
    async def _extract_phonetic_features(
        self, 
        audio: np.ndarray, 
        text: str, 
        sample_rate: int
    ) -> np.ndarray:
        """Extract phonetic features"""
        try:
            features = []
            
            # Convert text to phonemes
            if self.g2p:
                phonemes = self.g2p(text)
            else:
                phonemes = list(text.lower())  # Fallback
            
            # Vowel analysis
            vowels = [p for p in phonemes if p in 'aeiou']
            vowel_diversity = len(set(vowels)) / max(len(vowels), 1)
            features.append(vowel_diversity)
            
            # Consonant analysis
            consonants = [p for p in phonemes if p not in 'aeiou']
            consonant_diversity = len(set(consonants)) / max(len(consonants), 1)
            features.append(consonant_diversity)
            
            # Formant analysis
            if len(audio) > sample_rate:  # At least 1 second
                formants = self._extract_formants(audio, sample_rate)
                features.extend(formants[:3])  # F1, F2, F3
            else:
                features.extend([800, 1200, 2400])  # Default formants
            
            # Pad or truncate to 40 features
            while len(features) < 40:
                features.append(0.0)
            
            return np.array(features[:40])
            
        except Exception as e:
            logger.error(f"Error extracting phonetic features: {e}")
            return np.zeros(40)
    
    async def _extract_prosodic_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract prosodic features"""
        try:
            features = []
            
            # Fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[voiced_flag]
            
            if len(f0_clean) > 0:
                features.extend([
                    np.mean(f0_clean),  # Mean F0
                    np.std(f0_clean),   # F0 variation
                    np.max(f0_clean) - np.min(f0_clean),  # F0 range
                ])
            else:
                features.extend([150, 20, 100])  # Default values
            
            # Energy and spectral features
            rms_energy = librosa.feature.rms(y=audio)[0]
            features.append(np.mean(rms_energy))
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            features.append(np.mean(spectral_centroid))
            
            # Zero crossing rate (related to voicing)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zcr))
            
            # Rhythm features
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
                inter_onset_intervals = np.diff(onset_times)
                features.extend([
                    np.mean(inter_onset_intervals),
                    np.std(inter_onset_intervals)
                ])
            else:
                features.extend([0.5, 0.1])  # Default rhythm
            
            # Pad or truncate to 20 features
            while len(features) < 20:
                features.append(0.0)
            
            return np.array(features[:20])
            
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return np.zeros(20)
    
    # Adaptation methods
    
    async def _adapt_phonetics(
        self,
        request: AccentAdaptationRequest,
        source_profile: Optional[AccentProfile],
        target_profile: AccentProfile
    ) -> Dict[str, Any]:
        """Adapt phonetics to target accent"""
        try:
            changes = []
            adapted_phonemes = []
            
            # Convert text to phonemes
            if self.g2p:
                phonemes = self.g2p(request.text)
            else:
                phonemes = list(request.text.lower())  # Fallback
            
            # Apply vowel shifts
            for phoneme in phonemes:
                if phoneme in target_profile.vowel_shifts:
                    new_phoneme = target_profile.vowel_shifts[phoneme]
                    changes.append({
                        'type': 'vowel_shift',
                        'original': phoneme,
                        'adapted': new_phoneme,
                        'strength': request.adaptation_strength
                    })
                    adapted_phonemes.append(new_phoneme)
                elif phoneme in target_profile.consonant_variations:
                    new_phoneme = target_profile.consonant_variations[phoneme]
                    changes.append({
                        'type': 'consonant_variation',
                        'original': phoneme,
                        'adapted': new_phoneme,
                        'strength': request.adaptation_strength
                    })
                    adapted_phonemes.append(new_phoneme)
                else:
                    adapted_phonemes.append(phoneme)
            
            confidence = 0.8 if changes else 0.5
            
            return {
                'changes': changes,
                'adapted_phonemes': adapted_phonemes,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error adapting phonetics: {e}")
            return {
                'changes': [],
                'adapted_phonemes': list(request.text.lower()),
                'confidence': 0.3
            }
    
    async def _adapt_prosody(
        self,
        request: AccentAdaptationRequest,
        target_profile: AccentProfile
    ) -> Dict[str, Any]:
        """Adapt prosodic characteristics"""
        try:
            adjustments = {}
            
            # Speaking rate adjustment
            adjustments['speaking_rate'] = target_profile.speech_rate
            
            # Intonation pattern
            adjustments['intonation_pattern'] = target_profile.intonation_pattern
            
            # Stress patterns
            words = request.text.split()
            word_stress = {}
            for word in words:
                if word.lower() in target_profile.stress_patterns:
                    word_stress[word] = target_profile.stress_patterns[word.lower()]
            
            adjustments['stress_patterns'] = word_stress
            
            # Rhythm adjustments
            adjustments['rhythm_type'] = target_profile.rhythm_type
            
            # F0 range
            adjustments['f0_range'] = target_profile.fundamental_frequency_range
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error adapting prosody: {e}")
            return {}
    
    async def _adapt_vocabulary(
        self,
        request: AccentAdaptationRequest,
        target_profile: AccentProfile
    ) -> Dict[str, Any]:
        """Adapt vocabulary to regional variations"""
        try:
            if not request.adapt_vocabulary:
                return {
                    'adapted_text': request.text,
                    'changes': {},
                    'confidence': 1.0
                }
            
            adapted_text = request.text
            changes = {}
            
            # Apply vocabulary substitutions
            for original, regional in target_profile.vocabulary_variations.items():
                if original in adapted_text.lower():
                    # Case-sensitive replacement
                    pattern = re.compile(re.escape(original), re.IGNORECASE)
                    adapted_text = pattern.sub(regional, adapted_text)
                    changes[original] = regional
            
            # Add regional phrases if appropriate
            if target_profile.common_phrases and request.target_formality == "casual":
                # Could add regional phrases contextually
                pass
            
            confidence = 0.9 if changes else 0.8
            
            return {
                'adapted_text': adapted_text,
                'changes': changes,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error adapting vocabulary: {e}")
            return {
                'adapted_text': request.text,
                'changes': {},
                'confidence': 0.5
            }
    
    async def _calculate_audio_parameters(
        self,
        target_profile: AccentProfile,
        request: AccentAdaptationRequest
    ) -> Dict[str, Any]:
        """Calculate audio synthesis parameters for target accent"""
        try:
            params = {}
            
            # Fundamental frequency
            f0_min, f0_max = target_profile.fundamental_frequency_range
            params['f0_mean'] = (f0_min + f0_max) / 2
            params['f0_std'] = (f0_max - f0_min) / 4
            
            # Speaking rate
            params['speaking_rate'] = target_profile.speech_rate
            
            # Formant frequencies
            params['formant_ratios'] = target_profile.formant_ratios
            
            # Voice quality
            params['voice_quality'] = target_profile.voice_quality
            
            # Prosodic parameters
            params['intonation'] = target_profile.intonation_pattern
            params['rhythm'] = target_profile.rhythm_type
            
            # Adaptation strength
            params['adaptation_strength'] = request.adaptation_strength
            
            return params
            
        except Exception as e:
            logger.error(f"Error calculating audio parameters: {e}")
            return {}
    
    # Analysis methods
    
    async def _analyze_phonetics(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze phonetic characteristics"""
        try:
            analysis = {
                'vowel_shifts': {},
                'consonant_variations': {},
                'r_coloring': 'rhotic',  # Default
                'th_pronunciation': 'standard'  # Default
            }
            
            # Convert text to phonemes for analysis
            if self.g2p:
                phonemes = self.g2p(text)
                
                # Analyze vowel patterns
                for i, phoneme in enumerate(phonemes):
                    if phoneme in 'aeiou':
                        # Simple vowel analysis
                        if phoneme == 'a' and i < len(phonemes) - 1:
                            if phonemes[i + 1] == 'r':
                                analysis['r_coloring'] = 'rhotic'
                        
                        # Placeholder for actual vowel shift detection
                        analysis['vowel_shifts'][phoneme] = phoneme
                
                # Analyze consonant patterns
                for phoneme in phonemes:
                    if phoneme not in 'aeiou':
                        # Placeholder for consonant variation detection
                        analysis['consonant_variations'][phoneme] = phoneme
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing phonetics: {e}")
            return {
                'vowel_shifts': {},
                'consonant_variations': {},
                'r_coloring': 'rhotic',
                'th_pronunciation': 'standard'
            }
    
    async def _analyze_prosody(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze prosodic characteristics"""
        try:
            # Extract fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[voiced_flag]
            
            # Analyze intonation
            if len(f0_clean) > 10:
                f0_trend = np.polyfit(range(len(f0_clean)), f0_clean, 1)[0]
                if f0_trend > 0.5:
                    intonation = 'rising'
                elif f0_trend < -0.5:
                    intonation = 'falling'
                else:
                    intonation = 'flat'
            else:
                intonation = 'flat'
            
            # Calculate speaking rate
            duration = len(audio) / sample_rate
            # Estimate syllables (simplified)
            syllable_count = max(1, len([c for c in str(audio) if c in 'aeiou']))
            speech_rate = syllable_count / duration
            
            # Analyze rhythm
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
            if len(onset_frames) > 2:
                onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
                inter_onset_intervals = np.diff(onset_times)
                rhythm_regularity = 1.0 / (1.0 + np.std(inter_onset_intervals))
                
                if rhythm_regularity > 0.7:
                    rhythm = 'syllable-timed'
                else:
                    rhythm = 'stress-timed'
            else:
                rhythm = 'stress-timed'
            
            return {
                'intonation': intonation,
                'speech_rate': speech_rate,
                'rhythm': rhythm,
                'stress_patterns': {}  # Would be populated with actual analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prosody: {e}")
            return {
                'intonation': 'flat',
                'speech_rate': 3.0,
                'rhythm': 'stress-timed',
                'stress_patterns': {}
            }
    
    async def _analyze_formants(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze formant characteristics"""
        try:
            # Extract formants
            formants = self._extract_formants(audio, sample_rate)
            
            # Calculate formant ratios
            if len(formants) >= 3:
                f1, f2, f3 = formants[:3]
                ratios = [f2/f1, f3/f1, f3/f2] if f1 > 0 else [2.0, 3.0, 1.5]
            else:
                ratios = [2.0, 3.0, 1.5]  # Default ratios
            
            # Analyze F0 range
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[voiced_flag]
            
            if len(f0_clean) > 0:
                f0_range = (float(np.min(f0_clean)), float(np.max(f0_clean)))
            else:
                f0_range = (100.0, 200.0)  # Default range
            
            # Analyze voice quality (simplified)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            if avg_centroid > 3000:
                quality = 'breathy'
            elif avg_centroid < 1500:
                quality = 'creaky'
            else:
                quality = 'modal'
            
            return {
                'ratios': ratios,
                'f0_range': f0_range,
                'quality': quality,
                'formants': formants
            }
            
        except Exception as e:
            logger.error(f"Error analyzing formants: {e}")
            return {
                'ratios': [2.0, 3.0, 1.5],
                'f0_range': (100.0, 200.0),
                'quality': 'modal',
                'formants': [800, 1200, 2400]
            }
    
    # Helper methods
    
    def _extract_formants(self, audio: np.ndarray, sample_rate: int) -> List[float]:
        """Extract formant frequencies (simplified)"""
        try:
            # Using FFT-based spectral peaks as rough formant estimates
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find peaks in positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Find spectral peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.1)
            
            if len(peaks) >= 3:
                # Take first few peaks as formants
                formant_freqs = positive_freqs[peaks[:3]].tolist()
                return [abs(f) for f in formant_freqs]
            else:
                return [800, 1200, 2400]  # Default formants
                
        except Exception:
            return [800, 1200, 2400]
    
    def _map_accent_id_to_name(self, accent_id: int) -> str:
        """Map accent classifier ID to accent name"""
        accent_names = [
            'general_american', 'british_received_pronunciation', 'australian',
            'canadian', 'irish', 'scottish', 'welsh', 'south_african',
            'new_zealand', 'indian', 'singapore', 'jamaican',
            'southern_american', 'northeastern_american', 'midwestern_american',
            'california', 'new_york', 'boston', 'texas', 'chicago'
        ]
        
        if 0 <= accent_id < len(accent_names):
            return accent_names[accent_id]
        else:
            return 'general_american'
    
    async def _heuristic_accent_detection(
        self,
        audio: np.ndarray,
        text: Optional[str],
        sample_rate: int
    ) -> Tuple[str, float]:
        """Fallback heuristic accent detection"""
        try:
            # Simple heuristics based on audio characteristics
            
            # Extract fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[voiced_flag]
            
            if len(f0_clean) > 0:
                avg_f0 = np.mean(f0_clean)
                
                # Very simple classification based on F0
                if avg_f0 > 180:
                    return 'british_received_pronunciation', 0.6
                elif avg_f0 < 140:
                    return 'australian', 0.6
                else:
                    return 'general_american', 0.7
            else:
                return 'general_american', 0.5
                
        except Exception as e:
            logger.error(f"Error in heuristic accent detection: {e}")
            return 'general_american', 0.3
    
    def _initialize_vowel_systems(self) -> Dict[str, Dict[str, str]]:
        """Initialize vowel system mappings for different accents"""
        return {
            'general_american': {
                'æ': 'æ', 'ɑ': 'ɑ', 'ɔ': 'ɔ', 'i': 'i', 'u': 'u'
            },
            'british_received_pronunciation': {
                'æ': 'a', 'ɑ': 'ɑː', 'ɔ': 'ɒ', 'i': 'iː', 'u': 'uː'
            },
            'australian': {
                'æ': 'ɛ', 'ɑ': 'aː', 'ɔ': 'oː', 'i': 'ɪi', 'u': 'ʉː'
            }
        }
    
    def _initialize_consonant_systems(self) -> Dict[str, Dict[str, str]]:
        """Initialize consonant system mappings"""
        return {
            'general_american': {
                'r': 'ɹ', 't': 't', 'θ': 'θ', 'ð': 'ð'
            },
            'british_received_pronunciation': {
                'r': '', 't': 't', 'θ': 'θ', 'ð': 'ð'  # Non-rhotic
            },
            'australian': {
                'r': '', 't': 't', 'θ': 'f', 'ð': 'v'  # TH-fronting
            }
        }
    
    def _initialize_prosodic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize prosodic pattern templates"""
        return {
            'general_american': {
                'intonation': 'falling',
                'rhythm': 'stress-timed',
                'speech_rate': 3.5,
                'f0_range': (80, 250)
            },
            'british_received_pronunciation': {
                'intonation': 'rising',
                'rhythm': 'stress-timed',
                'speech_rate': 3.0,
                'f0_range': (85, 280)
            },
            'australian': {
                'intonation': 'rising',
                'rhythm': 'syllable-timed',
                'speech_rate': 3.2,
                'f0_range': (90, 270)
            }
        }
    
    async def _initialize_phoneme_mappings(self):
        """Initialize phoneme mapping tables"""
        try:
            # Load default phoneme mappings
            default_mappings = {
                'general_american_to_british': {
                    'ɑ': 'ɑː',
                    'æ': 'a',
                    'ɹ': '',  # R-dropping
                },
                'british_to_general_american': {
                    'ɑː': 'ɑ',
                    'a': 'æ',
                    '': 'ɹ',  # R-insertion
                }
            }
            
            self.phoneme_mappings.update(default_mappings)
            
        except Exception as e:
            logger.warning(f"Could not initialize phoneme mappings: {e}")
    
    async def _load_pretrained_models(self):
        """Load pre-trained models"""
        try:
            # Check for existing models
            classifier_path = self.models_dir / "accent_classifier.pth"
            transformer_path = self.models_dir / "phonetic_transformer.pth"
            
            if classifier_path.exists() and self.accent_classifier:
                self.accent_classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained accent classifier")
            
            if transformer_path.exists() and self.phonetic_transformer:
                self.phonetic_transformer.load_state_dict(
                    torch.load(transformer_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained phonetic transformer")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    async def _load_accent_profiles(self):
        """Load accent profiles from storage"""
        try:
            profiles_file = self.models_dir / "accent_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for profile_data in profiles_data:
                    # Handle datetime fields
                    profile_data['last_updated'] = datetime.fromisoformat(
                        profile_data['last_updated']
                    )
                    
                    profile = AccentProfile(**profile_data)
                    self.accent_profiles[profile.accent_id] = profile
                
                logger.info(f"Loaded {len(self.accent_profiles)} accent profiles")
            else:
                # Create default profiles
                await self._create_default_profiles()
                
        except Exception as e:
            logger.warning(f"Could not load accent profiles: {e}")
            await self._create_default_profiles()
    
    async def _create_default_profiles(self):
        """Create default accent profiles"""
        try:
            default_profiles = [
                {
                    'accent_id': 'general_american',
                    'region': 'United States',
                    'country': 'USA',
                    'dialect_name': 'General American',
                    'vowel_shifts': self.vowel_systems['general_american'],
                    'consonant_variations': self.consonant_systems['general_american'],
                    'r_coloring': 'rhotic',
                    'th_pronunciation': 'standard',
                    'intonation_pattern': 'falling',
                    'stress_patterns': {},
                    'rhythm_type': 'stress-timed',
                    'speech_rate': 3.5,
                    'common_phrases': ['How are you?', 'Have a good day!'],
                    'vocabulary_variations': {},
                    'grammatical_features': [],
                    'formant_ratios': [2.0, 3.0, 1.5],
                    'fundamental_frequency_range': (80.0, 250.0),
                    'voice_quality': 'modal',
                    'confidence_score': 1.0,
                    'sample_count': 100,
                    'last_updated': datetime.now()
                },
                {
                    'accent_id': 'british_received_pronunciation',
                    'region': 'England',
                    'country': 'UK',
                    'dialect_name': 'Received Pronunciation',
                    'vowel_shifts': self.vowel_systems['british_received_pronunciation'],
                    'consonant_variations': self.consonant_systems['british_received_pronunciation'],
                    'r_coloring': 'non-rhotic',
                    'th_pronunciation': 'standard',
                    'intonation_pattern': 'rising',
                    'stress_patterns': {},
                    'rhythm_type': 'stress-timed',
                    'speech_rate': 3.0,
                    'common_phrases': ['How do you do?', 'Cheerio!'],
                    'vocabulary_variations': {'elevator': 'lift', 'apartment': 'flat'},
                    'grammatical_features': [],
                    'formant_ratios': [1.8, 2.8, 1.6],
                    'fundamental_frequency_range': (85.0, 280.0),
                    'voice_quality': 'modal',
                    'confidence_score': 1.0,
                    'sample_count': 100,
                    'last_updated': datetime.now()
                }
            ]
            
            for profile_data in default_profiles:
                profile = AccentProfile(**profile_data)
                self.accent_profiles[profile.accent_id] = profile
            
            # Save default profiles
            await self._save_all_accent_profiles()
            
            logger.info("Created default accent profiles")
            
        except Exception as e:
            logger.error(f"Error creating default profiles: {e}")
    
    async def _save_accent_profile(self, profile: AccentProfile):
        """Save individual accent profile"""
        try:
            # Load all profiles and update
            await self._save_all_accent_profiles()
            
        except Exception as e:
            logger.error(f"Error saving accent profile: {e}")
    
    async def _save_all_accent_profiles(self):
        """Save all accent profiles to storage"""
        try:
            profiles_data = []
            
            for profile in self.accent_profiles.values():
                profile_dict = asdict(profile)
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                profiles_data.append(profile_dict)
            
            profiles_file = self.models_dir / "accent_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving accent profiles: {e}")


# Global instance
regional_adaptation_engine = RegionalAdaptationEngine()