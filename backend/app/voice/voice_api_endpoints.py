"""
FastAPI Endpoints for Advanced Voice AI Features
Comprehensive REST API for all voice AI capabilities
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import base64
import io
import asyncio
import logging
import uuid
from pathlib import Path

from .advanced_voice_integration import (
    advanced_voice_ai_system,
    AdvancedVoiceRequest,
    AdvancedVoiceResponse
)
from .voice_cloning import VoiceProfile, VoiceCloneRequest, VoiceCloneResult
from .accent_adaptation import AccentProfile, AccentAdaptationRequest, AccentAdaptationResult
from .noise_intelligence import NoiseProfile, NoiseReductionRequest, NoiseReductionResult
from .multi_participant import Speaker, ConversationFlow, MultiParticipantAnalysis

logger = logging.getLogger(__name__)

# Pydantic models for API endpoints

class VoiceProcessingRequest(BaseModel):
    """Comprehensive voice processing request"""
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

class VoiceProcessingResponse(BaseModel):
    """Voice processing response"""
    success: bool
    processing_time: float
    
    # Audio data (base64 encoded)
    audio_data: Optional[str] = None
    audio_format: str = "wav"
    sample_rate: int = 16000
    
    # Processing results
    voice_clone_result: Optional[Dict[str, Any]] = None
    accent_adaptation_result: Optional[Dict[str, Any]] = None
    noise_reduction_result: Optional[Dict[str, Any]] = None
    speaker_analysis: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    overall_quality_score: float = 0.0
    processing_stages: List[str] = []
    
    # Status
    errors: List[str] = []
    warnings: List[str] = []

class VoiceProfileCreateRequest(BaseModel):
    """Request to create voice profile"""
    sales_rep_name: str = Field(..., min_length=2, max_length=100)
    voice_name: str = Field(..., min_length=2, max_length=100)
    voice_characteristics: Dict[str, Any] = {}

class AccentDetectionRequest(BaseModel):
    """Request for accent detection"""
    sample_rate: int = 16000
    text: Optional[str] = None

class NoiseAnalysisRequest(BaseModel):
    """Request for noise analysis"""
    sample_rate: int = 16000
    analysis_type: str = "comprehensive"

class CallAnalysisRequest(BaseModel):
    """Request to start call analysis"""
    call_id: str = Field(..., min_length=1)
    expected_participants: Optional[List[str]] = []
    call_type: str = "sales"

class BatchProcessingRequest(BaseModel):
    """Request for batch processing"""
    requests: List[VoiceProcessingRequest] = Field(..., max_items=10)
    process_parallel: bool = True

class BatchProcessingResponse(BaseModel):
    """Batch processing response"""
    success: bool
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_processing_time: float
    results: List[VoiceProcessingResponse]
    errors: List[str] = []

# Create FastAPI app
app = FastAPI(
    title="Advanced Voice AI System",
    description="Comprehensive Voice AI with cloning, accent adaptation, noise intelligence, and multi-participant support",
    version="2.0.0",
    docs_url="/voice-ai/docs",
    openapi_url="/voice-ai/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize the voice AI system"""
    try:
        logger.info("Starting Advanced Voice AI System...")
        await advanced_voice_ai_system.initialize()
        logger.info("Advanced Voice AI System started successfully")
    except Exception as e:
        logger.error(f"Failed to start Advanced Voice AI System: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Advanced Voice AI System...")

# Main Voice Processing Endpoints

@app.post(
    "/voice-ai/process",
    response_model=VoiceProcessingResponse,
    summary="Process Voice with Advanced AI",
    description="Comprehensive voice processing with cloning, accent adaptation, noise reduction, and speaker analysis"
)
async def process_voice(
    audio_file: Optional[UploadFile] = File(None),
    request_data: str = Form(...)
):
    """Process voice with all advanced AI features"""
    try:
        # Parse request data
        request_dict = json.loads(request_data)
        request = VoiceProcessingRequest(**request_dict)
        
        # Read audio file if provided
        audio_data = None
        if audio_file:
            audio_data = await audio_file.read()
        
        # Create advanced voice request
        advanced_request = AdvancedVoiceRequest(
            audio_data=audio_data,
            text=request.text,
            sample_rate=request.sample_rate,
            voice_profile_id=request.voice_profile_id,
            clone_voice=request.clone_voice,
            target_accent=request.target_accent,
            source_accent=request.source_accent,
            adapt_accent=request.adapt_accent,
            reduce_noise=request.reduce_noise,
            noise_adaptation_mode=request.noise_adaptation_mode,
            call_id=request.call_id,
            enable_speaker_identification=request.enable_speaker_identification,
            return_analysis=request.return_analysis,
            return_audio=request.return_audio,
            audio_format=request.audio_format
        )
        
        # Process the request
        response = await advanced_voice_ai_system.process_voice_request(advanced_request)
        
        # Convert to API response format
        api_response = VoiceProcessingResponse(
            success=response.success,
            processing_time=response.processing_time,
            audio_data=base64.b64encode(response.processed_audio).decode() if response.processed_audio else None,
            audio_format=response.audio_format,
            sample_rate=response.sample_rate,
            voice_clone_result=response.voice_clone_result.__dict__ if response.voice_clone_result else None,
            accent_adaptation_result=response.accent_adaptation_result.__dict__ if response.accent_adaptation_result else None,
            noise_reduction_result=response.noise_reduction_result.__dict__ if response.noise_reduction_result else None,
            speaker_analysis=response.speaker_analysis,
            overall_quality_score=response.overall_quality_score,
            processing_stages=response.processing_stages,
            errors=response.errors or [],
            warnings=response.warnings or []
        )
        
        return api_response
        
    except Exception as e:
        logger.error(f"Error processing voice request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/voice-ai/batch-process",
    response_model=BatchProcessingResponse,
    summary="Batch Process Multiple Voice Requests",
    description="Process multiple voice requests in parallel or sequentially"
)
async def batch_process_voice(
    batch_request: BatchProcessingRequest
):
    """Process multiple voice requests"""
    try:
        start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0
        errors = []
        
        if batch_request.process_parallel:
            # Process requests in parallel
            tasks = []
            for request in batch_request.requests:
                advanced_request = AdvancedVoiceRequest(
                    text=request.text,
                    sample_rate=request.sample_rate,
                    voice_profile_id=request.voice_profile_id,
                    clone_voice=request.clone_voice,
                    target_accent=request.target_accent,
                    source_accent=request.source_accent,
                    adapt_accent=request.adapt_accent,
                    reduce_noise=request.reduce_noise,
                    noise_adaptation_mode=request.noise_adaptation_mode,
                    call_id=request.call_id,
                    enable_speaker_identification=request.enable_speaker_identification,
                    return_analysis=request.return_analysis,
                    return_audio=request.return_audio,
                    audio_format=request.audio_format
                )
                tasks.append(advanced_voice_ai_system.process_voice_request(advanced_request))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    failed_count += 1
                    errors.append(f"Request {i}: {str(response)}")
                    results.append(VoiceProcessingResponse(
                        success=False,
                        processing_time=0.0,
                        errors=[str(response)]
                    ))
                else:
                    successful_count += 1
                    results.append(VoiceProcessingResponse(
                        success=response.success,
                        processing_time=response.processing_time,
                        audio_data=base64.b64encode(response.processed_audio).decode() if response.processed_audio else None,
                        audio_format=response.audio_format,
                        sample_rate=response.sample_rate,
                        voice_clone_result=response.voice_clone_result.__dict__ if response.voice_clone_result else None,
                        accent_adaptation_result=response.accent_adaptation_result.__dict__ if response.accent_adaptation_result else None,
                        noise_reduction_result=response.noise_reduction_result.__dict__ if response.noise_reduction_result else None,
                        speaker_analysis=response.speaker_analysis,
                        overall_quality_score=response.overall_quality_score,
                        processing_stages=response.processing_stages,
                        errors=response.errors or [],
                        warnings=response.warnings or []
                    ))
        else:
            # Process requests sequentially
            for i, request in enumerate(batch_request.requests):
                try:
                    advanced_request = AdvancedVoiceRequest(
                        text=request.text,
                        sample_rate=request.sample_rate,
                        voice_profile_id=request.voice_profile_id,
                        clone_voice=request.clone_voice,
                        target_accent=request.target_accent,
                        source_accent=request.source_accent,
                        adapt_accent=request.adapt_accent,
                        reduce_noise=request.reduce_noise,
                        noise_adaptation_mode=request.noise_adaptation_mode,
                        call_id=request.call_id,
                        enable_speaker_identification=request.enable_speaker_identification,
                        return_analysis=request.return_analysis,
                        return_audio=request.return_audio,
                        audio_format=request.audio_format
                    )
                    
                    response = await advanced_voice_ai_system.process_voice_request(advanced_request)
                    successful_count += 1
                    
                    results.append(VoiceProcessingResponse(
                        success=response.success,
                        processing_time=response.processing_time,
                        audio_data=base64.b64encode(response.processed_audio).decode() if response.processed_audio else None,
                        audio_format=response.audio_format,
                        sample_rate=response.sample_rate,
                        voice_clone_result=response.voice_clone_result.__dict__ if response.voice_clone_result else None,
                        accent_adaptation_result=response.accent_adaptation_result.__dict__ if response.accent_adaptation_result else None,
                        noise_reduction_result=response.noise_reduction_result.__dict__ if response.noise_reduction_result else None,
                        speaker_analysis=response.speaker_analysis,
                        overall_quality_score=response.overall_quality_score,
                        processing_stages=response.processing_stages,
                        errors=response.errors or [],
                        warnings=response.warnings or []
                    ))
                    
                except Exception as e:
                    failed_count += 1
                    errors.append(f"Request {i}: {str(e)}")
                    results.append(VoiceProcessingResponse(
                        success=False,
                        processing_time=0.0,
                        errors=[str(e)]
                    ))
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchProcessingResponse(
            success=failed_count == 0,
            total_requests=len(batch_request.requests),
            successful_requests=successful_count,
            failed_requests=failed_count,
            total_processing_time=total_processing_time,
            results=results,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice Cloning Endpoints

@app.post(
    "/voice-ai/voice-profiles",
    response_model=Dict[str, Any],
    summary="Create Voice Profile",
    description="Create a new voice profile for a sales representative"
)
async def create_voice_profile(
    request: VoiceProfileCreateRequest,
    training_audio: List[UploadFile] = File(...),
    transcripts: List[str] = Form(...)
):
    """Create a new voice profile"""
    try:
        # Save uploaded training audio files
        training_files = []
        for i, audio_file in enumerate(training_audio):
            file_path = f"temp/training_audio_{uuid.uuid4()}_{i}.wav"
            Path(file_path).parent.mkdir(exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(await audio_file.read())
            
            training_files.append(file_path)
        
        # Create voice profile
        profile = await advanced_voice_ai_system.create_voice_profile(
            request.sales_rep_name,
            request.voice_name,
            training_files,
            transcripts,
            request.voice_characteristics
        )
        
        # Cleanup temporary files
        for file_path in training_files:
            try:
                Path(file_path).unlink()
            except:
                pass
        
        return {
            "success": True,
            "profile_id": profile.profile_id,
            "message": f"Voice profile '{request.voice_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating voice profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/voice-profiles",
    response_model=List[Dict[str, Any]],
    summary="Get Voice Profiles",
    description="Get all available voice profiles"
)
async def get_voice_profiles():
    """Get all voice profiles"""
    try:
        profiles = await advanced_voice_ai_system.get_voice_profiles()
        return [
            {
                "profile_id": profile.profile_id,
                "sales_rep_name": profile.sales_rep_name,
                "voice_name": profile.voice_name,
                "created_at": profile.created_at.isoformat(),
                "voice_characteristics": profile.voice_characteristics,
                "training_quality": profile.quality_score
            }
            for profile in profiles
        ]
        
    except Exception as e:
        logger.error(f"Error getting voice profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(
    "/voice-ai/voice-profiles/{profile_id}",
    response_model=Dict[str, Any],
    summary="Delete Voice Profile",
    description="Delete a voice profile"
)
async def delete_voice_profile(profile_id: str):
    """Delete a voice profile"""
    try:
        success = await advanced_voice_ai_system.delete_voice_profile(profile_id)
        
        if success:
            return {
                "success": True,
                "message": f"Voice profile {profile_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Voice profile not found")
            
    except Exception as e:
        logger.error(f"Error deleting voice profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Accent Adaptation Endpoints

@app.post(
    "/voice-ai/detect-accent",
    response_model=Dict[str, Any],
    summary="Detect Accent",
    description="Detect accent from audio sample"
)
async def detect_accent(
    audio_file: UploadFile = File(...),
    request_data: str = Form(...)
):
    """Detect accent from audio"""
    try:
        request_dict = json.loads(request_data)
        request = AccentDetectionRequest(**request_dict)
        
        audio_data = await audio_file.read()
        
        accent, confidence = await advanced_voice_ai_system.detect_accent(
            audio_data, request.sample_rate, request.text
        )
        
        return {
            "detected_accent": accent,
            "confidence": confidence,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error detecting accent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/supported-accents",
    response_model=List[Dict[str, Any]],
    summary="Get Supported Accents",
    description="Get list of supported accents for adaptation"
)
async def get_supported_accents():
    """Get supported accents"""
    try:
        accents = await advanced_voice_ai_system.get_supported_accents()
        return accents
        
    except Exception as e:
        logger.error(f"Error getting supported accents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/voice-ai/learn-accent",
    response_model=Dict[str, Any],
    summary="Learn New Accent",
    description="Learn a new accent from sample audio"
)
async def learn_accent_from_sample(
    audio_file: UploadFile = File(...),
    text: str = Form(...),
    region: str = Form(...),
    sample_rate: int = Form(16000)
):
    """Learn new accent from sample"""
    try:
        audio_data = await audio_file.read()
        
        accent_profile = await advanced_voice_ai_system.learn_accent_from_sample(
            audio_data, text, region, sample_rate
        )
        
        return {
            "success": True,
            "accent_id": accent_profile.accent_id,
            "region": accent_profile.region,
            "message": f"New accent for {region} learned successfully"
        }
        
    except Exception as e:
        logger.error(f"Error learning accent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Noise Intelligence Endpoints

@app.post(
    "/voice-ai/analyze-noise",
    response_model=Dict[str, Any],
    summary="Analyze Noise Environment",
    description="Analyze noise characteristics in audio"
)
async def analyze_noise_environment(
    audio_file: UploadFile = File(...),
    request_data: str = Form(...)
):
    """Analyze noise environment"""
    try:
        request_dict = json.loads(request_data)
        request = NoiseAnalysisRequest(**request_dict)
        
        audio_data = await audio_file.read()
        
        noise_analysis = await advanced_voice_ai_system.analyze_noise_environment(
            audio_data, request.sample_rate
        )
        
        return {
            "success": True,
            "noise_analysis": noise_analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing noise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/noise-profiles",
    response_model=List[Dict[str, Any]],
    summary="Get Noise Profiles",
    description="Get available noise profiles for intelligent filtering"
)
async def get_noise_profiles():
    """Get noise profiles"""
    try:
        profiles = await advanced_voice_ai_system.get_noise_profiles()
        return profiles
        
    except Exception as e:
        logger.error(f"Error getting noise profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/voice-ai/learn-noise-profile",
    response_model=Dict[str, Any],
    summary="Learn Noise Profile",
    description="Learn a new noise profile for better filtering"
)
async def learn_noise_profile(
    audio_file: UploadFile = File(...),
    noise_type: str = Form(...),
    description: str = Form(...),
    sample_rate: int = Form(16000)
):
    """Learn new noise profile"""
    try:
        audio_data = await audio_file.read()
        
        noise_profile = await advanced_voice_ai_system.learn_noise_profile(
            audio_data, noise_type, description, sample_rate
        )
        
        return {
            "success": True,
            "profile_id": noise_profile.profile_id,
            "noise_type": noise_profile.noise_type,
            "message": f"Noise profile for {noise_type} learned successfully"
        }
        
    except Exception as e:
        logger.error(f"Error learning noise profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-participant Call Endpoints

@app.post(
    "/voice-ai/start-call-analysis",
    response_model=Dict[str, Any],
    summary="Start Call Analysis",
    description="Start multi-participant call analysis"
)
async def start_call_analysis(request: CallAnalysisRequest):
    """Start call analysis"""
    try:
        result = await advanced_voice_ai_system.start_call_analysis(
            request.call_id,
            request.expected_participants,
            request.call_type
        )
        
        return {
            "success": True,
            "call_id": request.call_id,
            "analysis_started": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error starting call analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/voice-ai/end-call-analysis/{call_id}",
    response_model=Dict[str, Any],
    summary="End Call Analysis",
    description="End call analysis and get final report"
)
async def end_call_analysis(call_id: str):
    """End call analysis"""
    try:
        analysis = await advanced_voice_ai_system.end_call_analysis(call_id)
        
        return {
            "success": True,
            "call_id": call_id,
            "final_analysis": {
                "call_id": analysis.call_id,
                "duration": analysis.duration,
                "participants": [
                    {
                        "speaker_id": speaker.speaker_id,
                        "name": speaker.name,
                        "speaking_time": speaker.total_speaking_time,
                        "emotion_summary": speaker.emotion_summary
                    }
                    for speaker in analysis.participants
                ],
                "conversation_flow": {
                    "total_turns": len(analysis.conversation_flow.turn_sequence),
                    "dominant_speaker": analysis.conversation_flow.dominant_speaker,
                    "interaction_quality": analysis.conversation_flow.interaction_quality
                },
                "insights": analysis.insights,
                "summary": analysis.summary
            }
        }
        
    except Exception as e:
        logger.error(f"Error ending call analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/call-analysis/{call_id}",
    response_model=Dict[str, Any],
    summary="Get Real-time Call Analysis",
    description="Get real-time analysis of ongoing call"
)
async def get_call_real_time_analysis(call_id: str):
    """Get real-time call analysis"""
    try:
        analysis = await advanced_voice_ai_system.get_call_real_time_analysis(call_id)
        
        return {
            "success": True,
            "call_id": call_id,
            "real_time_analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting call analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Status and Management Endpoints

@app.get(
    "/voice-ai/status",
    response_model=Dict[str, Any],
    summary="Get System Status",
    description="Get overall system status and health information"
)
async def get_system_status():
    """Get system status"""
    try:
        status = await advanced_voice_ai_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/diagnostics",
    response_model=Dict[str, Any],
    summary="Run System Diagnostics",
    description="Run comprehensive system diagnostics and health checks"
)
async def run_system_diagnostics():
    """Run system diagnostics"""
    try:
        diagnostics = await advanced_voice_ai_system.run_system_diagnostics()
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voice-ai/health",
    response_model=Dict[str, Any],
    summary="Health Check",
    description="Quick health check endpoint"
)
async def health_check():
    """Health check"""
    try:
        return {
            "status": "healthy" if advanced_voice_ai_system.initialized else "initializing",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Audio Download Endpoint

@app.get(
    "/voice-ai/download-audio/{request_id}",
    response_class=StreamingResponse,
    summary="Download Processed Audio",
    description="Download processed audio file"
)
async def download_audio(request_id: str, audio_format: str = "wav"):
    """Download processed audio"""
    try:
        # This would typically retrieve stored processed audio
        # For now, return a placeholder response
        
        def generate_audio():
            # Generate placeholder audio data
            import numpy as np
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_int = (audio * 32767).astype(np.int16)
            
            return audio_int.tobytes()
        
        audio_data = generate_audio()
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=f"audio/{audio_format}",
            headers={"Content-Disposition": f"attachment; filename=processed_audio_{request_id}.{audio_format}"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time processing (would be implemented separately)
# This is a placeholder for real-time voice processing capabilities

@app.get("/voice-ai/")
async def voice_ai_root():
    """Root endpoint for voice AI system"""
    return {
        "message": "Advanced Voice AI System",
        "version": "2.0.0",
        "features": [
            "Voice Cloning",
            "Accent Adaptation", 
            "Background Noise Intelligence",
            "Multi-participant Call Handling"
        ],
        "endpoints": {
            "docs": "/voice-ai/docs",
            "status": "/voice-ai/status",
            "process": "/voice-ai/process",
            "batch_process": "/voice-ai/batch-process"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "voice_api_endpoints:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )