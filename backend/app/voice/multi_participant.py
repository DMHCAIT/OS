"""
Multi-participant Call Handling System
Conference call management with speaker identification, conversation flow, and multi-speaker analysis
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import librosa
import scipy.signal
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import queue
import uuid

# Speaker recognition and diarization imports
try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines import SpeakerDiarization
    from resemblyzer import VoiceEncoder, preprocess_wav
    import webrtcvad
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
except ImportError:
    pyannote = None
    resemblyzer = None
    webrtcvad = None
    DBSCAN = None
    KMeans = None
    cosine_similarity = None
    nx = None

logger = logging.getLogger(__name__)

@dataclass
class Speaker:
    """Individual speaker in the conference"""
    speaker_id: str
    name: Optional[str] = None
    role: str = "participant"  # host, participant, guest, sales_rep, prospect
    
    # Voice characteristics
    voice_embedding: Optional[np.ndarray] = None
    fundamental_frequency: float = 150.0
    speaking_rate: float = 3.5  # words per second
    vocal_intensity: float = 0.7
    
    # Speaking patterns
    interruption_count: int = 0
    total_speaking_time: float = 0.0
    turn_count: int = 0
    average_turn_duration: float = 5.0
    
    # Engagement metrics
    engagement_score: float = 0.5
    participation_level: str = "moderate"
    attention_score: float = 0.7
    
    # Communication style
    communication_style: str = "balanced"  # dominant, submissive, balanced
    energy_level: str = "moderate"  # high, moderate, low
    formality_level: str = "professional"  # casual, professional, formal
    
    # Technical info
    audio_quality_score: float = 0.8
    connection_stability: float = 0.9
    last_active: datetime = None
    join_time: datetime = None

@dataclass
class ConversationTurn:
    """Individual speaking turn in conversation"""
    turn_id: str
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    
    # Content analysis
    transcript: Optional[str] = None
    word_count: int = 0
    sentiment_score: float = 0.0
    confidence_level: float = 0.0
    
    # Turn characteristics
    interruption: bool = False
    overlapping_speakers: List[str] = None
    silence_before: float = 0.0
    silence_after: float = 0.0
    
    # Voice analysis
    average_pitch: float = 150.0
    pitch_variation: float = 20.0
    speaking_rate: float = 3.5
    energy_level: float = 0.7
    
    # Turn type
    turn_type: str = "statement"  # question, statement, response, interruption

@dataclass
class ConversationFlow:
    """Overall conversation flow analysis"""
    total_duration: float
    speaker_count: int
    turn_count: int
    interruption_count: int
    
    # Flow metrics
    turn_taking_balance: float  # 0 = dominated by one speaker, 1 = balanced
    conversation_pace: float  # turns per minute
    silence_ratio: float  # percentage of silence
    overlap_ratio: float  # percentage of overlapping speech
    
    # Engagement metrics
    overall_engagement: float
    energy_distribution: Dict[str, float]
    participation_distribution: Dict[str, float]
    
    # Conversation quality
    flow_quality_score: float
    coherence_score: float
    collaboration_score: float

@dataclass
class MultiParticipantAnalysis:
    """Complete multi-participant call analysis"""
    call_id: str
    start_time: datetime
    end_time: datetime
    participants: List[Speaker]
    conversation_turns: List[ConversationTurn]
    conversation_flow: ConversationFlow
    
    # Call outcomes
    dominant_speaker: Optional[str] = None
    quietest_speaker: Optional[str] = None
    most_engaged_speaker: Optional[str] = None
    
    # Recommendations
    coaching_recommendations: List[str] = None
    engagement_suggestions: List[str] = None
    technical_issues: List[str] = None

class SpeakerEmbeddingNetwork(nn.Module):
    """Neural network for speaker embedding extraction"""
    
    def __init__(self, input_dim=40, embedding_dim=256, hidden_dim=512):
        super().__init__()
        
        # MFCC feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Speaker embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.L2Norm(dim=1)  # L2 normalize embeddings
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, 1000)  # Adjust based on speaker count
    
    def forward(self, mfcc_features, return_embedding=True):
        # Encode features
        encoded = self.feature_encoder(mfcc_features)
        
        # Temporal pooling
        pooled = self.temporal_pool(encoded).squeeze(-1)
        
        # Generate embedding
        embedding = self.embedding_layers(pooled)
        
        if return_embedding:
            return embedding
        else:
            # For training with classification
            logits = self.classifier(embedding)
            return embedding, logits

class ConversationFlowAnalyzer(nn.Module):
    """Neural network for conversation flow analysis"""
    
    def __init__(self, speaker_embedding_dim=256, sequence_length=100):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Turn feature encoder
        self.turn_encoder = nn.Sequential(
            nn.Linear(speaker_embedding_dim + 10, 128),  # +10 for turn features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Sequence modeling
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(256, num_heads=8)
        
        # Flow prediction heads
        self.engagement_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.dominance_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.quality_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, turn_embeddings, turn_features):
        batch_size, seq_len, _ = turn_embeddings.shape
        
        # Combine embeddings with turn features
        combined_input = torch.cat([turn_embeddings, turn_features], dim=-1)
        
        # Encode turns
        encoded_turns = self.turn_encoder(combined_input)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded_turns)
        
        # Self-attention
        lstm_out_t = lstm_out.transpose(0, 1)
        attended, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attended = attended.transpose(0, 1)
        
        # Global pooling
        global_features = torch.mean(attended, dim=1)
        
        # Predictions
        engagement = self.engagement_predictor(global_features)
        dominance = self.dominance_predictor(global_features)
        quality = self.quality_predictor(global_features)
        
        return engagement, dominance, quality

class MultiParticipantCallHandler:
    """Main handler for multi-participant conference calls"""
    
    def __init__(
        self,
        models_dir: str = "models/multi_participant",
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
        self.speaker_embedding_model = None
        self.conversation_flow_model = None
        self.diarization_pipeline = None
        self.voice_encoder = None
        
        # Active call management
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self.speaker_database: Dict[str, Speaker] = {}
        self.conversation_history: List[MultiParticipantAnalysis] = []
        
        # Processing parameters
        self.frame_duration = 0.5  # seconds
        self.min_speaker_duration = 1.0  # minimum duration to consider a speaker
        self.similarity_threshold = 0.85  # speaker similarity threshold
        
        # Real-time processing
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.audio_buffers: Dict[str, deque] = {}
        
        logger.info(f"MultiParticipantCallHandler initialized on device: {self.device}")
    
    async def initialize(self):
        """Initialize multi-participant call handling models"""
        try:
            logger.info("Initializing multi-participant call models...")
            
            # Initialize neural models
            self.speaker_embedding_model = SpeakerEmbeddingNetwork().to(self.device)
            self.conversation_flow_model = ConversationFlowAnalyzer().to(self.device)
            
            # Initialize speaker diarization
            if pyannote:
                try:
                    # Would require pyannote.audio model setup
                    # self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                    pass
                except:
                    logger.warning("Pyannote.audio not properly configured")
            
            # Initialize voice encoder
            if resemblyzer:
                try:
                    self.voice_encoder = VoiceEncoder()
                except:
                    logger.warning("Resemblyzer not available")
            
            # Load pre-trained models
            await self._load_pretrained_models()
            
            # Load speaker database
            await self._load_speaker_database()
            
            logger.info("Multi-participant call models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing multi-participant call handler: {e}")
            raise
    
    async def start_call_analysis(
        self,
        call_id: str,
        expected_participants: List[str] = None,
        call_type: str = "sales",
        real_time_callbacks: Dict[str, callable] = None
    ) -> Dict[str, Any]:
        """Start analyzing a multi-participant call"""
        try:
            logger.info(f"Starting call analysis for call: {call_id}")
            
            # Initialize call data structure
            call_data = {
                'call_id': call_id,
                'start_time': datetime.now(),
                'call_type': call_type,
                'expected_participants': expected_participants or [],
                'detected_speakers': {},
                'conversation_turns': [],
                'audio_buffer': deque(maxlen=10000),
                'real_time_callbacks': real_time_callbacks or {},
                'processing_active': True,
                'current_speakers': set(),
                'speaker_embeddings': {},
                'turn_counter': 0
            }
            
            self.active_calls[call_id] = call_data
            self.audio_buffers[call_id] = deque(maxlen=100000)
            
            # Start real-time processing thread
            processing_thread = threading.Thread(
                target=self._real_time_processing_worker,
                args=(call_id,)
            )
            processing_thread.start()
            self.processing_threads[call_id] = processing_thread
            
            logger.info(f"Call analysis started for: {call_id}")
            
            return {
                'call_id': call_id,
                'status': 'started',
                'start_time': call_data['start_time'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting call analysis: {e}")
            raise
    
    async def process_audio_chunk(
        self,
        call_id: str,
        audio_chunk: np.ndarray,
        timestamp: float,
        sample_rate: int = None
    ) -> Dict[str, Any]:
        """Process incoming audio chunk for active call"""
        try:
            if call_id not in self.active_calls:
                raise ValueError(f"No active call found: {call_id}")
            
            if sample_rate and sample_rate != self.sample_rate:
                audio_chunk = librosa.resample(
                    audio_chunk, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Add to audio buffer
            self.audio_buffers[call_id].extend([(audio_chunk, timestamp)])
            
            # Trigger real-time analysis if buffer is sufficiently large
            if len(self.audio_buffers[call_id]) >= 10:
                # Process in background thread
                pass
            
            return {
                'call_id': call_id,
                'chunk_processed': True,
                'buffer_size': len(self.audio_buffers[call_id])
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                'call_id': call_id,
                'chunk_processed': False,
                'error': str(e)
            }
    
    async def get_real_time_analysis(
        self,
        call_id: str
    ) -> Dict[str, Any]:
        """Get current real-time analysis for active call"""
        try:
            if call_id not in self.active_calls:
                raise ValueError(f"No active call found: {call_id}")
            
            call_data = self.active_calls[call_id]
            
            # Current speakers
            current_speakers = []
            for speaker_id, speaker_info in call_data['detected_speakers'].items():
                if speaker_info.get('currently_speaking', False):
                    current_speakers.append({
                        'speaker_id': speaker_id,
                        'name': speaker_info.get('name'),
                        'speaking_duration': speaker_info.get('current_turn_duration', 0),
                        'confidence': speaker_info.get('confidence', 0)
                    })
            
            # Recent conversation turns
            recent_turns = call_data['conversation_turns'][-5:] if call_data['conversation_turns'] else []
            
            # Call statistics
            duration = (datetime.now() - call_data['start_time']).total_seconds()
            speaker_count = len(call_data['detected_speakers'])
            turn_count = len(call_data['conversation_turns'])
            
            return {
                'call_id': call_id,
                'duration': duration,
                'speaker_count': speaker_count,
                'turn_count': turn_count,
                'current_speakers': current_speakers,
                'recent_turns': recent_turns,
                'conversation_pace': turn_count / max(duration / 60, 1),  # turns per minute
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time analysis: {e}")
            return {
                'call_id': call_id,
                'error': str(e)
            }
    
    async def identify_speakers(
        self,
        audio: np.ndarray,
        sample_rate: int = None
    ) -> List[Dict[str, Any]]:
        """Identify speakers in audio segment"""
        try:
            if sample_rate and sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Speaker diarization
            if self.diarization_pipeline:
                # Use pyannote.audio for diarization
                diarization_result = await self._pyannote_diarization(audio)
            else:
                # Fallback to simple segmentation
                diarization_result = await self._simple_speaker_segmentation(audio)
            
            # Speaker identification
            speakers = []
            for segment in diarization_result:
                segment_audio = audio[
                    int(segment['start'] * self.sample_rate):
                    int(segment['end'] * self.sample_rate)
                ]
                
                if len(segment_audio) < self.sample_rate * self.min_speaker_duration:
                    continue  # Skip very short segments
                
                # Extract speaker embedding
                speaker_embedding = await self._extract_speaker_embedding(segment_audio)
                
                # Match against known speakers
                speaker_id, confidence = await self._match_speaker(speaker_embedding)
                
                speakers.append({
                    'speaker_id': speaker_id,
                    'confidence': confidence,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'duration': segment['end'] - segment['start'],
                    'embedding': speaker_embedding.tolist() if speaker_embedding is not None else None
                })
            
            return speakers
            
        except Exception as e:
            logger.error(f"Error identifying speakers: {e}")
            return []
    
    async def analyze_conversation_flow(
        self,
        call_id: str
    ) -> ConversationFlow:
        """Analyze conversation flow for completed or ongoing call"""
        try:
            if call_id in self.active_calls:
                call_data = self.active_calls[call_id]
                conversation_turns = call_data['conversation_turns']
                duration = (datetime.now() - call_data['start_time']).total_seconds()
            else:
                # Look in conversation history
                call_analysis = None
                for analysis in self.conversation_history:
                    if analysis.call_id == call_id:
                        call_analysis = analysis
                        break
                
                if not call_analysis:
                    raise ValueError(f"Call not found: {call_id}")
                
                conversation_turns = call_analysis.conversation_turns
                duration = (call_analysis.end_time - call_analysis.start_time).total_seconds()
            
            # Calculate flow metrics
            speaker_count = len(set(turn.speaker_id for turn in conversation_turns))
            turn_count = len(conversation_turns)
            interruption_count = sum(1 for turn in conversation_turns if turn.interruption)
            
            # Turn-taking balance
            if speaker_count > 1:
                speaker_turn_counts = defaultdict(int)
                for turn in conversation_turns:
                    speaker_turn_counts[turn.speaker_id] += 1
                
                # Calculate entropy for balance
                total_turns = sum(speaker_turn_counts.values())
                if total_turns > 0:
                    probs = [count / total_turns for count in speaker_turn_counts.values()]
                    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                    max_entropy = np.log2(speaker_count)
                    turn_taking_balance = entropy / max_entropy if max_entropy > 0 else 0
                else:
                    turn_taking_balance = 0
            else:
                turn_taking_balance = 0
            
            # Conversation pace
            conversation_pace = turn_count / max(duration / 60, 1)  # turns per minute
            
            # Silence and overlap ratios
            total_speaking_time = sum(turn.duration for turn in conversation_turns)
            silence_ratio = max(0, (duration - total_speaking_time) / duration) if duration > 0 else 0
            
            overlap_count = sum(1 for turn in conversation_turns if turn.overlapping_speakers)
            overlap_ratio = overlap_count / max(turn_count, 1)
            
            # Engagement metrics
            speaker_participation = defaultdict(float)
            speaker_energy = defaultdict(list)
            
            for turn in conversation_turns:
                speaker_participation[turn.speaker_id] += turn.duration
                speaker_energy[turn.speaker_id].append(turn.energy_level)
            
            total_participation = sum(speaker_participation.values())
            participation_distribution = {}
            energy_distribution = {}
            
            for speaker_id in speaker_participation:
                participation_distribution[speaker_id] = (
                    speaker_participation[speaker_id] / total_participation
                    if total_participation > 0 else 0
                )
                energy_distribution[speaker_id] = (
                    np.mean(speaker_energy[speaker_id])
                    if speaker_energy[speaker_id] else 0.5
                )
            
            overall_engagement = np.mean(list(energy_distribution.values())) if energy_distribution else 0.5
            
            # Quality scores
            flow_quality_score = (
                turn_taking_balance * 0.3 +
                min(1.0, conversation_pace / 5.0) * 0.2 +  # Optimal ~5 turns/min
                (1.0 - silence_ratio) * 0.2 +
                (1.0 - overlap_ratio) * 0.3
            )
            
            coherence_score = max(0, 1.0 - interruption_count / max(turn_count, 1))
            collaboration_score = turn_taking_balance * overall_engagement
            
            conversation_flow = ConversationFlow(
                total_duration=duration,
                speaker_count=speaker_count,
                turn_count=turn_count,
                interruption_count=interruption_count,
                turn_taking_balance=turn_taking_balance,
                conversation_pace=conversation_pace,
                silence_ratio=silence_ratio,
                overlap_ratio=overlap_ratio,
                overall_engagement=overall_engagement,
                energy_distribution=energy_distribution,
                participation_distribution=participation_distribution,
                flow_quality_score=flow_quality_score,
                coherence_score=coherence_score,
                collaboration_score=collaboration_score
            )
            
            return conversation_flow
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            raise
    
    async def end_call_analysis(
        self,
        call_id: str
    ) -> MultiParticipantAnalysis:
        """End call analysis and generate final report"""
        try:
            if call_id not in self.active_calls:
                raise ValueError(f"No active call found: {call_id}")
            
            logger.info(f"Ending call analysis for: {call_id}")
            
            call_data = self.active_calls[call_id]
            call_data['processing_active'] = False
            
            # Stop processing thread
            if call_id in self.processing_threads:
                processing_thread = self.processing_threads[call_id]
                processing_thread.join(timeout=5.0)
                del self.processing_threads[call_id]
            
            # Final analysis
            end_time = datetime.now()
            
            # Convert detected speakers to Speaker objects
            participants = []
            for speaker_id, speaker_info in call_data['detected_speakers'].items():
                speaker = Speaker(
                    speaker_id=speaker_id,
                    name=speaker_info.get('name'),
                    role=speaker_info.get('role', 'participant'),
                    voice_embedding=speaker_info.get('embedding'),
                    total_speaking_time=speaker_info.get('total_speaking_time', 0),
                    turn_count=speaker_info.get('turn_count', 0),
                    interruption_count=speaker_info.get('interruption_count', 0),
                    engagement_score=speaker_info.get('engagement_score', 0.5),
                    join_time=speaker_info.get('join_time'),
                    last_active=speaker_info.get('last_active')
                )
                participants.append(speaker)
            
            # Analyze conversation flow
            conversation_flow = await self.analyze_conversation_flow(call_id)
            
            # Generate insights and recommendations
            recommendations = await self._generate_recommendations(call_data, conversation_flow)
            
            # Create final analysis
            analysis = MultiParticipantAnalysis(
                call_id=call_id,
                start_time=call_data['start_time'],
                end_time=end_time,
                participants=participants,
                conversation_turns=call_data['conversation_turns'],
                conversation_flow=conversation_flow,
                dominant_speaker=self._find_dominant_speaker(participants),
                quietest_speaker=self._find_quietest_speaker(participants),
                most_engaged_speaker=self._find_most_engaged_speaker(participants),
                coaching_recommendations=recommendations['coaching'],
                engagement_suggestions=recommendations['engagement'],
                technical_issues=recommendations['technical']
            )
            
            # Store in history
            self.conversation_history.append(analysis)
            
            # Clean up active call data
            del self.active_calls[call_id]
            if call_id in self.audio_buffers:
                del self.audio_buffers[call_id]
            
            # Save analysis
            await self._save_call_analysis(analysis)
            
            logger.info(f"Call analysis completed for: {call_id}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error ending call analysis: {e}")
            raise
    
    async def get_speaker_insights(
        self,
        speaker_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get insights for specific speaker across calls"""
        try:
            speaker_calls = []
            
            # Collect data from conversation history
            for analysis in self.conversation_history:
                if time_range:
                    if not (time_range[0] <= analysis.start_time <= time_range[1]):
                        continue
                
                for participant in analysis.participants:
                    if participant.speaker_id == speaker_id:
                        speaker_calls.append({
                            'call_id': analysis.call_id,
                            'date': analysis.start_time,
                            'duration': analysis.conversation_flow.total_duration,
                            'speaking_time': participant.total_speaking_time,
                            'turn_count': participant.turn_count,
                            'engagement_score': participant.engagement_score,
                            'interruption_count': participant.interruption_count
                        })
                        break
            
            if not speaker_calls:
                return {
                    'speaker_id': speaker_id,
                    'total_calls': 0,
                    'insights': 'No call data available for this speaker'
                }
            
            # Calculate aggregate insights
            total_calls = len(speaker_calls)
            total_speaking_time = sum(call['speaking_time'] for call in speaker_calls)
            total_turns = sum(call['turn_count'] for call in speaker_calls)
            avg_engagement = np.mean([call['engagement_score'] for call in speaker_calls])
            total_interruptions = sum(call['interruption_count'] for call in speaker_calls)
            
            # Speaking patterns
            speaking_times = [call['speaking_time'] for call in speaker_calls]
            avg_speaking_time = np.mean(speaking_times)
            speaking_consistency = 1.0 - (np.std(speaking_times) / max(avg_speaking_time, 1))
            
            # Engagement trends
            engagement_scores = [call['engagement_score'] for call in speaker_calls]
            engagement_trend = "stable"
            if len(engagement_scores) >= 3:
                recent_avg = np.mean(engagement_scores[-3:])
                early_avg = np.mean(engagement_scores[:3])
                if recent_avg > early_avg + 0.1:
                    engagement_trend = "improving"
                elif recent_avg < early_avg - 0.1:
                    engagement_trend = "declining"
            
            # Communication style analysis
            interruption_rate = total_interruptions / max(total_turns, 1)
            if interruption_rate > 0.3:
                communication_style = "dominant"
            elif interruption_rate < 0.1 and avg_engagement < 0.5:
                communication_style = "passive"
            else:
                communication_style = "balanced"
            
            insights = {
                'speaker_id': speaker_id,
                'total_calls': total_calls,
                'total_speaking_time': total_speaking_time,
                'average_speaking_time_per_call': avg_speaking_time,
                'speaking_consistency': speaking_consistency,
                'average_engagement': avg_engagement,
                'engagement_trend': engagement_trend,
                'communication_style': communication_style,
                'interruption_rate': interruption_rate,
                'total_turns': total_turns,
                'calls_analyzed': len(speaker_calls),
                'date_range': {
                    'start': min(call['date'] for call in speaker_calls).isoformat(),
                    'end': max(call['date'] for call in speaker_calls).isoformat()
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting speaker insights: {e}")
            return {
                'speaker_id': speaker_id,
                'error': str(e)
            }
    
    # Private methods
    
    def _real_time_processing_worker(self, call_id: str):
        """Background worker for real-time call processing"""
        try:
            logger.info(f"Starting real-time processing for call: {call_id}")
            
            last_analysis_time = datetime.now()
            analysis_interval = 5.0  # seconds
            
            while (call_id in self.active_calls and 
                   self.active_calls[call_id]['processing_active']):
                
                try:
                    # Check if enough time has passed for analysis
                    if (datetime.now() - last_analysis_time).total_seconds() >= analysis_interval:
                        # Collect recent audio
                        if call_id in self.audio_buffers and len(self.audio_buffers[call_id]) > 0:
                            
                            # Get recent audio chunks
                            recent_chunks = list(self.audio_buffers[call_id])[-50:]  # Last 50 chunks
                            
                            if recent_chunks:
                                # Combine audio chunks
                                audio_data = np.concatenate([chunk[0] for chunk in recent_chunks])
                                
                                # Process for speaker identification
                                asyncio.run(self._process_real_time_chunk(call_id, audio_data))
                                
                                last_analysis_time = datetime.now()
                    
                    # Sleep to prevent excessive CPU usage
                    threading.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in real-time processing worker: {e}")
                    threading.sleep(1.0)
            
            logger.info(f"Real-time processing ended for call: {call_id}")
            
        except Exception as e:
            logger.error(f"Error in real-time processing worker: {e}")
    
    async def _process_real_time_chunk(self, call_id: str, audio_data: np.ndarray):
        """Process audio chunk for real-time analysis"""
        try:
            call_data = self.active_calls[call_id]
            
            # Identify speakers in current chunk
            speakers = await self.identify_speakers(audio_data)
            
            # Update call data with current speakers
            current_time = datetime.now()
            
            for speaker_info in speakers:
                speaker_id = speaker_info['speaker_id']
                
                # Update or create speaker data
                if speaker_id not in call_data['detected_speakers']:
                    call_data['detected_speakers'][speaker_id] = {
                        'speaker_id': speaker_id,
                        'name': self._get_speaker_name(speaker_id),
                        'join_time': current_time,
                        'total_speaking_time': 0,
                        'turn_count': 0,
                        'interruption_count': 0,
                        'engagement_score': 0.5,
                        'currently_speaking': False,
                        'last_active': current_time
                    }
                
                speaker_data = call_data['detected_speakers'][speaker_id]
                speaker_data['currently_speaking'] = True
                speaker_data['last_active'] = current_time
                speaker_data['total_speaking_time'] += speaker_info['duration']
                
                # Create conversation turn
                turn = ConversationTurn(
                    turn_id=f"{call_id}_{call_data['turn_counter']}",
                    speaker_id=speaker_id,
                    start_time=speaker_info['start_time'],
                    end_time=speaker_info['end_time'],
                    duration=speaker_info['duration'],
                    average_pitch=150.0,  # Would extract from audio
                    speaking_rate=3.5,   # Would extract from audio
                    energy_level=0.7     # Would extract from audio
                )
                
                call_data['conversation_turns'].append(turn)
                call_data['turn_counter'] += 1
                speaker_data['turn_count'] += 1
            
            # Update currently speaking status
            current_speaker_ids = {speaker['speaker_id'] for speaker in speakers}
            for speaker_id, speaker_data in call_data['detected_speakers'].items():
                speaker_data['currently_speaking'] = speaker_id in current_speaker_ids
            
            # Trigger callbacks if registered
            if 'speaker_update' in call_data['real_time_callbacks']:
                callback = call_data['real_time_callbacks']['speaker_update']
                callback(call_id, current_speaker_ids)
            
        except Exception as e:
            logger.error(f"Error processing real-time chunk: {e}")
    
    async def _extract_speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio segment"""
        try:
            # Use ResemblyZer if available
            if self.voice_encoder:
                # Preprocess audio
                processed_audio = preprocess_wav(audio, self.sample_rate)
                embedding = self.voice_encoder.embed_utterance(processed_audio)
                return embedding
            
            # Fallback to neural model
            if self.speaker_embedding_model:
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(
                    y=audio, sr=self.sample_rate, n_mfcc=40, hop_length=512
                )
                
                # Ensure minimum length
                if mfccs.shape[1] < 50:
                    mfccs = np.pad(mfccs, ((0, 0), (0, 50 - mfccs.shape[1])), mode='constant')
                elif mfccs.shape[1] > 200:
                    mfccs = mfccs[:, :200]
                
                # Convert to tensor
                mfcc_tensor = torch.FloatTensor(mfccs).unsqueeze(0).to(self.device)
                
                # Extract embedding
                self.speaker_embedding_model.eval()
                with torch.no_grad():
                    embedding = self.speaker_embedding_model(mfcc_tensor)
                
                return embedding.cpu().numpy().squeeze()
            
            # Simple fallback embedding
            return np.random.randn(256).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return np.random.randn(256).astype(np.float32)
    
    async def _match_speaker(
        self, 
        speaker_embedding: np.ndarray
    ) -> Tuple[str, float]:
        """Match speaker embedding against known speakers"""
        try:
            if len(self.speaker_database) == 0:
                # No known speakers, create new ID
                new_speaker_id = f"speaker_{len(self.speaker_database) + 1}"
                return new_speaker_id, 0.5
            
            # Calculate similarities
            best_match_id = None
            best_similarity = 0.0
            
            for speaker_id, speaker in self.speaker_database.items():
                if speaker.voice_embedding is not None:
                    # Cosine similarity
                    if cosine_similarity:
                        similarity = cosine_similarity([speaker_embedding], [speaker.voice_embedding])[0][0]
                    else:
                        # Fallback dot product similarity
                        norm1 = np.linalg.norm(speaker_embedding)
                        norm2 = np.linalg.norm(speaker.voice_embedding)
                        if norm1 > 0 and norm2 > 0:
                            similarity = np.dot(speaker_embedding, speaker.voice_embedding) / (norm1 * norm2)
                        else:
                            similarity = 0.0
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = speaker_id
            
            # Check if similarity is above threshold
            if best_similarity >= self.similarity_threshold:
                return best_match_id, best_similarity
            else:
                # Create new speaker
                new_speaker_id = f"speaker_{len(self.speaker_database) + 1}"
                return new_speaker_id, 0.7  # Confidence for new speaker
                
        except Exception as e:
            logger.error(f"Error matching speaker: {e}")
            new_speaker_id = f"speaker_{len(self.speaker_database) + 1}"
            return new_speaker_id, 0.5
    
    async def _pyannote_diarization(self, audio: np.ndarray) -> List[Dict[str, float]]:
        """Speaker diarization using pyannote.audio"""
        try:
            if not self.diarization_pipeline:
                return await self._simple_speaker_segmentation(audio)
            
            # Convert to pyannote format and process
            # This would require proper pyannote.audio setup
            # diarization = self.diarization_pipeline({'audio': audio})
            
            # For now, fallback to simple segmentation
            return await self._simple_speaker_segmentation(audio)
            
        except Exception as e:
            logger.error(f"Error in pyannote diarization: {e}")
            return await self._simple_speaker_segmentation(audio)
    
    async def _simple_speaker_segmentation(self, audio: np.ndarray) -> List[Dict[str, float]]:
        """Simple speaker segmentation fallback"""
        try:
            # Voice activity detection
            if webrtcvad:
                vad = webrtcvad.Vad(2)
                frame_duration = 30  # ms
                frame_length = int(self.sample_rate * frame_duration / 1000)
                
                segments = []
                current_start = None
                
                for i in range(0, len(audio) - frame_length, frame_length):
                    frame = audio[i:i + frame_length]
                    frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                    
                    try:
                        is_speech = vad.is_speech(frame_bytes, self.sample_rate)
                        
                        if is_speech and current_start is None:
                            current_start = i / self.sample_rate
                        elif not is_speech and current_start is not None:
                            end_time = i / self.sample_rate
                            if end_time - current_start >= self.min_speaker_duration:
                                segments.append({
                                    'start': current_start,
                                    'end': end_time
                                })
                            current_start = None
                    except:
                        pass
                
                # Close final segment if needed
                if current_start is not None:
                    end_time = len(audio) / self.sample_rate
                    if end_time - current_start >= self.min_speaker_duration:
                        segments.append({
                            'start': current_start,
                            'end': end_time
                        })
                
                return segments
            else:
                # Very simple fallback - divide into equal segments
                duration = len(audio) / self.sample_rate
                num_segments = max(1, int(duration / 5))  # 5-second segments
                
                segments = []
                segment_duration = duration / num_segments
                
                for i in range(num_segments):
                    start = i * segment_duration
                    end = min((i + 1) * segment_duration, duration)
                    segments.append({
                        'start': start,
                        'end': end
                    })
                
                return segments
                
        except Exception as e:
            logger.error(f"Error in simple speaker segmentation: {e}")
            # Return single segment as fallback
            return [{'start': 0.0, 'end': len(audio) / self.sample_rate}]
    
    def _get_speaker_name(self, speaker_id: str) -> Optional[str]:
        """Get speaker name from database"""
        if speaker_id in self.speaker_database:
            return self.speaker_database[speaker_id].name
        return None
    
    def _find_dominant_speaker(self, participants: List[Speaker]) -> Optional[str]:
        """Find the speaker who talked the most"""
        if not participants:
            return None
        
        dominant = max(participants, key=lambda s: s.total_speaking_time)
        return dominant.speaker_id
    
    def _find_quietest_speaker(self, participants: List[Speaker]) -> Optional[str]:
        """Find the speaker who talked the least"""
        if not participants:
            return None
        
        quietest = min(participants, key=lambda s: s.total_speaking_time)
        return quietest.speaker_id
    
    def _find_most_engaged_speaker(self, participants: List[Speaker]) -> Optional[str]:
        """Find the most engaged speaker"""
        if not participants:
            return None
        
        most_engaged = max(participants, key=lambda s: s.engagement_score)
        return most_engaged.speaker_id
    
    async def _generate_recommendations(
        self, 
        call_data: Dict[str, Any], 
        conversation_flow: ConversationFlow
    ) -> Dict[str, List[str]]:
        """Generate coaching and improvement recommendations"""
        try:
            recommendations = {
                'coaching': [],
                'engagement': [],
                'technical': []
            }
            
            # Coaching recommendations
            if conversation_flow.turn_taking_balance < 0.3:
                recommendations['coaching'].append(
                    "Consider encouraging more balanced participation from all speakers"
                )
            
            if conversation_flow.interruption_count > conversation_flow.turn_count * 0.2:
                recommendations['coaching'].append(
                    "High interruption rate detected - consider establishing clearer speaking protocols"
                )
            
            if conversation_flow.conversation_pace < 2:
                recommendations['coaching'].append(
                    "Conversation pace is slow - consider more engaging discussion techniques"
                )
            
            if conversation_flow.silence_ratio > 0.3:
                recommendations['coaching'].append(
                    "High silence ratio - encourage more active participation"
                )
            
            # Engagement recommendations
            if conversation_flow.overall_engagement < 0.5:
                recommendations['engagement'].append(
                    "Overall engagement is low - consider more interactive discussion formats"
                )
            
            for speaker_id, participation in conversation_flow.participation_distribution.items():
                if participation < 0.1:
                    speaker_name = self._get_speaker_name(speaker_id) or speaker_id
                    recommendations['engagement'].append(
                        f"Speaker {speaker_name} had very low participation - consider direct engagement"
                    )
            
            # Technical recommendations
            for speaker_id, speaker_data in call_data['detected_speakers'].items():
                audio_quality = speaker_data.get('audio_quality_score', 0.8)
                if audio_quality < 0.6:
                    speaker_name = self._get_speaker_name(speaker_id) or speaker_id
                    recommendations['technical'].append(
                        f"Poor audio quality detected for speaker {speaker_name} - check microphone setup"
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                'coaching': ["Error generating coaching recommendations"],
                'engagement': ["Error generating engagement recommendations"],
                'technical': ["Error generating technical recommendations"]
            }
    
    async def _load_pretrained_models(self):
        """Load pre-trained models"""
        try:
            embedding_path = self.models_dir / "speaker_embedding.pth"
            flow_path = self.models_dir / "conversation_flow.pth"
            
            if embedding_path.exists() and self.speaker_embedding_model:
                self.speaker_embedding_model.load_state_dict(
                    torch.load(embedding_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained speaker embedding model")
            
            if flow_path.exists() and self.conversation_flow_model:
                self.conversation_flow_model.load_state_dict(
                    torch.load(flow_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained conversation flow model")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    async def _load_speaker_database(self):
        """Load speaker database"""
        try:
            db_file = self.models_dir / "speaker_database.json"
            if db_file.exists():
                with open(db_file, 'r') as f:
                    speakers_data = json.load(f)
                
                for speaker_data in speakers_data:
                    if 'voice_embedding' in speaker_data:
                        speaker_data['voice_embedding'] = np.array(speaker_data['voice_embedding'])
                    
                    # Handle datetime fields
                    if 'join_time' in speaker_data and speaker_data['join_time']:
                        speaker_data['join_time'] = datetime.fromisoformat(speaker_data['join_time'])
                    if 'last_active' in speaker_data and speaker_data['last_active']:
                        speaker_data['last_active'] = datetime.fromisoformat(speaker_data['last_active'])
                    
                    speaker = Speaker(**speaker_data)
                    self.speaker_database[speaker.speaker_id] = speaker
                
                logger.info(f"Loaded {len(self.speaker_database)} speakers from database")
                
        except Exception as e:
            logger.warning(f"Could not load speaker database: {e}")
    
    async def _save_call_analysis(self, analysis: MultiParticipantAnalysis):
        """Save call analysis to storage"""
        try:
            # Create directory for call analyses
            analyses_dir = self.models_dir / "call_analyses"
            analyses_dir.mkdir(exist_ok=True)
            
            # Save individual analysis
            analysis_file = analyses_dir / f"{analysis.call_id}.json"
            analysis_dict = asdict(analysis)
            
            # Handle datetime and numpy array serialization
            analysis_dict['start_time'] = analysis.start_time.isoformat()
            analysis_dict['end_time'] = analysis.end_time.isoformat()
            
            for participant in analysis_dict['participants']:
                if participant['voice_embedding'] is not None:
                    participant['voice_embedding'] = participant['voice_embedding'].tolist()
                if participant['join_time']:
                    participant['join_time'] = participant['join_time'].isoformat()
                if participant['last_active']:
                    participant['last_active'] = participant['last_active'].isoformat()
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving call analysis: {e}")


# Global instance
multi_participant_call_handler = MultiParticipantCallHandler()