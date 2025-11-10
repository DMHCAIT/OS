# API Documentation

## Overview

The AI Lead Management & Voice Communication System provides a comprehensive REST API for managing leads, conducting voice AI conversations, and analyzing customer interactions.

**Base URL**: `http://localhost:8000/api/v1`

**Authentication**: JWT Bearer tokens

## Authentication

### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

### Using Authentication
Include the token in all subsequent requests:
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Lead Management

### Create Lead
```http
POST /api/v1/leads
Content-Type: application/json
Authorization: Bearer <token>

{
  "first_name": "John",
  "last_name": "Doe",
  "email": "john.doe@company.com",
  "phone": "+1-555-123-4567",
  "company": "Acme Corp",
  "title": "VP of Sales",
  "industry": "technology",
  "source": "website",
  "notes": "Interested in our AI solution"
}
```

**Response:**
```json
{
  "id": "lead_abc123",
  "first_name": "John",
  "last_name": "Doe",
  "email": "john.doe@company.com",
  "phone": "+1-555-123-4567",
  "company": "Acme Corp",
  "title": "VP of Sales",
  "industry": "technology",
  "source": "website",
  "status": "new",
  "score": {
    "overall_score": 0.75,
    "demographic_score": 0.8,
    "behavioral_score": 0.6,
    "engagement_score": 0.5,
    "conversion_probability": 0.72
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Get Leads
```http
GET /api/v1/leads?page=1&per_page=20&status=new&min_score=0.7
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 50, max: 100)
- `status` (string): Filter by lead status
- `source` (string): Filter by lead source
- `assigned_to` (string): Filter by assigned user
- `search` (string): Search in name, email, company
- `min_score` (float): Minimum lead score (0-1)

**Response:**
```json
{
  "leads": [
    {
      "id": "lead_abc123",
      "first_name": "John",
      "last_name": "Doe",
      // ... full lead object
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 20,
  "total_pages": 8
}
```

### Update Lead
```http
PUT /api/v1/leads/{lead_id}
Content-Type: application/json
Authorization: Bearer <token>

{
  "status": "qualified",
  "notes": "Had productive call, very interested"
}
```

### Get Lead Score
```http
POST /api/v1/leads/{lead_id}/score
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overall_score": 0.85,
  "demographic_score": 0.8,
  "behavioral_score": 0.9,
  "engagement_score": 0.8,
  "intent_score": 0.9,
  "conversion_probability": 0.82,
  "factors": {
    "positive_factors": [
      "Has company information",
      "Senior title/decision maker",
      "Requested product demo"
    ],
    "negative_factors": [],
    "recommendations": [
      "High priority - Schedule immediate call"
    ]
  }
}
```

### Add Lead Activity
```http
POST /api/v1/leads/{lead_id}/activities
Content-Type: application/json
Authorization: Bearer <token>

{
  "activity_type": "email_sent",
  "description": "Sent product demo invitation",
  "metadata": {
    "email_subject": "Demo of our AI solution",
    "campaign_id": "campaign_123"
  }
}
```

## Voice AI Calls

### Create Call
```http
POST /api/v1/voice/calls
Content-Type: application/json
Authorization: Bearer <token>

{
  "lead_id": "lead_abc123",
  "phone_number": "+1-555-123-4567",
  "call_type": "outbound",
  "purpose": "Discovery call to understand needs",
  "voice_settings": {
    "voice_id": "bella",
    "speed": 1.0,
    "language": "en-US"
  }
}
```

### Start Voice Call
```http
POST /api/v1/voice/calls/{call_id}/start
Content-Type: application/json
Authorization: Bearer <token>

{
  "voice_config": {
    "provider": "elevenlabs",
    "voice_id": "bella",
    "language": "en-US",
    "speed": 1.0,
    "stability": 0.75
  }
}
```

**Response:**
```json
{
  "success": true,
  "call_id": "call_xyz789",
  "opening_message": "Hi John, this is Sarah calling from AI Solutions. I hope I'm catching you at a good time!",
  "session_info": {
    "state": "greeting",
    "context": {
      "lead_name": "John",
      "company": "Acme Corp",
      "industry": "technology"
    }
  }
}
```

### Process Customer Response
```http
POST /api/v1/voice/calls/{call_id}/respond
Content-Type: application/json
Authorization: Bearer <token>

{
  "text": "Hi Sarah, yes I have a few minutes to chat",
  "audio_data": "base64_encoded_audio_data" // optional
}
```

**Response:**
```json
{
  "success": true,
  "ai_response": "Great! I wanted to reach out because I noticed Acme Corp has been exploring lead management solutions. What challenges are you currently facing with your sales process?",
  "audio_data": "base64_encoded_response_audio",
  "analysis": {
    "sentiment": 0.7,
    "intent": "general_inquiry",
    "objections": [],
    "interest_signals": ["verbal_interest"]
  },
  "conversation_state": "discovery",
  "should_continue": true
}
```

### Real-time WebSocket Communication
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/voice/calls/{call_id}/realtime');

// Send audio chunk
ws.send(JSON.stringify({
  type: 'audio_chunk',
  data: base64AudioData
}));

// Send text input
ws.send(JSON.stringify({
  type: 'text_input',
  text: 'Customer response text'
}));

// Receive AI response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'ai_response') {
    // Handle AI response
    console.log(response.text);
    playAudio(response.audio);
  }
};
```

### End Call
```http
POST /api/v1/voice/calls/{call_id}/end
Content-Type: application/json
Authorization: Bearer <token>

{
  "reason": "completed"
}
```

**Response:**
```json
{
  "success": true,
  "call_id": "call_xyz789",
  "summary": {
    "duration_minutes": 12.5,
    "total_exchanges": 18,
    "conversation_state": "closing",
    "objections_raised": ["price"],
    "interest_signals": ["demo_interest", "timeline_interest"],
    "overall_sentiment": 0.6,
    "recommendation": "Medium priority - Follow up with pricing information"
  },
  "analytics": {
    "conversion_probability": 0.65,
    "engagement_score": 0.7,
    "total_turns": 18,
    "avg_sentiment": 0.6
  }
}
```

## Analytics

### Dashboard Metrics
```http
GET /api/v1/analytics/dashboard?days=30
Authorization: Bearer <token>
```

**Response:**
```json
{
  "totalLeads": 1250,
  "newLeadsToday": 23,
  "totalCalls": 350,
  "callsToday": 12,
  "conversionRate": 15.2,
  "avgLeadScore": 0.68,
  "revenue": 125000,
  "voiceAICallsActive": 3
}
```

### Lead Analytics
```http
GET /api/v1/leads/analytics/overview?days=30
Authorization: Bearer <token>
```

**Response:**
```json
{
  "total_leads": 1250,
  "new_leads": 180,
  "qualified_leads": 320,
  "converted_leads": 95,
  "average_score": 0.68,
  "conversion_rate": 0.152,
  "by_source": {
    "website": 450,
    "referral": 280,
    "social_media": 220
  },
  "by_status": {
    "new": 180,
    "contacted": 320,
    "qualified": 280,
    "closed_won": 95
  },
  "by_industry": {
    "technology": 350,
    "healthcare": 280,
    "finance": 220
  }
}
```

## File Upload

### Import Leads
```http
POST /api/v1/leads/import
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: leads.csv
```

### Export Leads
```http
GET /api/v1/leads/export?format=csv
Authorization: Bearer <token>
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": "Validation error",
  "status_code": 422,
  "details": {
    "field": "email",
    "message": "Invalid email format"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Common Status Codes:**
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Rate Limiting

API calls are rate limited:
- **100 requests per minute** per user
- **1000 requests per hour** per user

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## Webhooks

Configure webhooks to receive real-time notifications:

### Webhook Events
- `lead.created`
- `lead.updated`
- `lead.score_changed`
- `call.started`
- `call.completed`
- `call.failed`

### Webhook Payload
```json
{
  "event": "lead.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "lead": {
      "id": "lead_abc123",
      // ... full lead object
    }
  }
}
```

## SDKs and Libraries

### Python SDK
```python
from ai_lead_client import LeadManagementClient

client = LeadManagementClient(
    api_key="your_api_key",
    base_url="http://localhost:8000/api/v1"
)

# Create a lead
lead = client.leads.create({
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com"
})

# Start a voice call
call = client.voice.start_call(
    lead_id=lead.id,
    phone_number="+1-555-123-4567"
)
```

### JavaScript SDK
```javascript
import { LeadManagementClient } from '@ai-lead/client';

const client = new LeadManagementClient({
  apiKey: 'your_api_key',
  baseUrl: 'http://localhost:8000/api/v1'
});

// Create a lead
const lead = await client.leads.create({
  firstName: 'John',
  lastName: 'Doe',
  email: 'john@example.com'
});

// Start a voice call
const call = await client.voice.startCall({
  leadId: lead.id,
  phoneNumber: '+1-555-123-4567'
});
```

## Testing

Use the interactive API documentation at `http://localhost:8000/docs` to test endpoints directly in your browser.

For automated testing, example requests are provided in the `/examples` directory of the repository.