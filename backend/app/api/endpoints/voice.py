"""
Voice AI API Endpoints
Real-time voice communication and call management
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import base64

from app.models.call import (
    Call, CallCreate, CallUpdate, CallResponse,
    VoiceSettings, VoiceMessage, CallSchedule
)
from app.voice.voice_ai import voice_ai_engine, VoiceConfig
from app.core.security import get_current_user
from app.core.database import get_database

router = APIRouter()


@router.post("/calls", response_model=CallResponse)
async def create_call(
    call_data: CallCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Create a new voice call"""
    try:
        # Create call record
        new_call = Call(**call_data.dict())
        new_call.created_by = current_user["id"]
        
        # Insert into database
        result = await db.calls.insert_one(new_call.dict())
        
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to create call")
        
        # Return created call
        created_call = await db.calls.find_one({"_id": result.inserted_id})
        return CallResponse(**created_call)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating call: {str(e)}")


@router.post("/calls/{call_id}/start")
async def start_voice_call(
    call_id: str,
    voice_config: Optional[VoiceConfig] = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Start a voice AI call"""
    try:
        # Get call and lead data
        call = await db.calls.find_one({"id": call_id})
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        lead = await db.leads.find_one({"id": call["lead_id"]})
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Use default voice config if not provided
        if not voice_config:
            voice_config = VoiceConfig()
        
        # Start the AI call
        result = await voice_ai_engine.start_call(call_id, lead, voice_config)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update call status
        await db.calls.update_one(
            {"id": call_id},
            {
                "$set": {
                    "status": "in_progress",
                    "started_at": datetime.utcnow()
                }
            }
        )
        
        return {
            "success": True,
            "call_id": call_id,
            "opening_message": result["opening_message"],
            "session_info": result["session_info"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting call: {str(e)}")


@router.post("/calls/{call_id}/respond")
async def process_customer_response(
    call_id: str,
    response_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Process customer response and generate AI reply"""
    try:
        # Extract response data
        text = response_data.get("text")
        audio_data = response_data.get("audio_data")
        
        if audio_data:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
        else:
            audio_bytes = None
        
        # Process with voice AI engine
        result = await voice_ai_engine.process_customer_response(
            call_id, audio_bytes, text
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "ai_response": result["ai_response"],
            "audio_data": base64.b64encode(result["audio_data"]).decode() if result["audio_data"] else None,
            "analysis": result["analysis"],
            "conversation_state": result["conversation_state"],
            "should_continue": result["should_continue"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")


@router.post("/calls/{call_id}/end")
async def end_voice_call(
    call_id: str,
    end_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """End a voice AI call"""
    try:
        reason = end_data.get("reason", "completed")
        
        # End the AI call
        result = await voice_ai_engine.end_call(call_id, reason)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update call record in database
        await db.calls.update_one(
            {"id": call_id},
            {
                "$set": {
                    "status": "completed",
                    "ended_at": result["end_time"],
                    "conversation": result["transcript"],
                    "analytics": result["analytics"],
                    "outcome": result["summary"]["recommendation"]
                }
            }
        )
        
        return {
            "success": True,
            "call_id": call_id,
            "summary": result["summary"],
            "analytics": result["analytics"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending call: {str(e)}")


@router.websocket("/calls/{call_id}/realtime")
async def realtime_call_communication(
    websocket: WebSocket,
    call_id: str
):
    """WebSocket endpoint for real-time voice communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio_chunk":
                # Process audio chunk in real-time
                audio_data = base64.b64decode(message["data"])
                
                # Process with voice AI (streaming mode)
                response = await voice_ai_engine.process_realtime_audio(
                    call_id, audio_data
                )
                
                if response:
                    await websocket.send_text(json.dumps({
                        "type": "ai_response",
                        "data": response
                    }))
            
            elif message["type"] == "text_input":
                # Process text input
                result = await voice_ai_engine.process_customer_response(
                    call_id, None, message["text"]
                )
                
                await websocket.send_text(json.dumps({
                    "type": "ai_response",
                    "text": result["ai_response"],
                    "audio": base64.b64encode(result["audio_data"]).decode() if result["audio_data"] else None
                }))
            
            elif message["type"] == "end_call":
                break
                
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
    
    finally:
        await websocket.close()


@router.get("/calls/{call_id}/transcript")
async def get_call_transcript(
    call_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get call transcript"""
    try:
        call = await db.calls.find_one({"id": call_id})
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        return {
            "call_id": call_id,
            "transcript": call.get("conversation", []),
            "analytics": call.get("analytics", {}),
            "summary": call.get("outcome", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")


@router.get("/calls/{call_id}/recording")
async def get_call_recording(
    call_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get call recording URL"""
    try:
        call = await db.calls.find_one({"id": call_id})
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        recording_url = call.get("recording_url")
        
        if not recording_url:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        return {"recording_url": recording_url}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recording: {str(e)}")


@router.post("/calls/schedule")
async def schedule_call(
    schedule_data: CallSchedule,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Schedule a voice AI call"""
    try:
        # Create scheduled call
        call_data = {
            "lead_id": schedule_data.lead_id,
            "status": "scheduled",
            "scheduled_at": schedule_data.scheduled_at,
            "purpose": f"Scheduled call - {schedule_data.notes}" if schedule_data.notes else "Scheduled call",
            "created_by": current_user["id"]
        }
        
        new_call = Call(**call_data)
        
        # Insert into database
        result = await db.calls.insert_one(new_call.dict())
        
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to schedule call")
        
        return {
            "success": True,
            "call_id": new_call.id,
            "scheduled_at": schedule_data.scheduled_at,
            "message": "Call scheduled successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling call: {str(e)}")


@router.get("/voices")
async def get_available_voices():
    """Get list of available AI voices"""
    return {
        "voices": [
            {
                "id": "bella",
                "name": "Bella",
                "gender": "female",
                "accent": "American",
                "description": "Professional, warm, confident"
            },
            {
                "id": "adam",
                "name": "Adam",
                "gender": "male", 
                "accent": "American",
                "description": "Professional, trustworthy, approachable"
            },
            {
                "id": "antoni",
                "name": "Antoni",
                "gender": "male",
                "accent": "British",
                "description": "Energetic, persuasive, dynamic"
            },
            {
                "id": "elli",
                "name": "Elli",
                "gender": "female",
                "accent": "American",
                "description": "Thoughtful, analytical, advisory"
            }
        ]
    }


@router.post("/test-voice")
async def test_voice(
    text: str,
    voice_config: VoiceConfig,
    current_user: dict = Depends(get_current_user)
):
    """Test voice synthesis with given text and settings"""
    try:
        # Generate speech from text
        audio_data = await voice_ai_engine._text_to_speech(text, voice_config)
        
        return {
            "success": True,
            "audio_data": base64.b64encode(audio_data).decode() if audio_data else None,
            "text": text,
            "voice_settings": voice_config.__dict__
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing voice: {str(e)}")


@router.post("/upload-audio")
async def upload_audio_for_transcription(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload audio file for transcription"""
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await file.read()
        
        # Convert to text using speech recognition
        config = VoiceConfig()
        text = await voice_ai_engine._speech_to_text(audio_data, config)
        
        return {
            "success": True,
            "transcript": text,
            "filename": file.filename,
            "size": len(audio_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")