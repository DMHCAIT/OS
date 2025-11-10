# Required API Keys for Production Deployment
# Complete list of API keys needed for full functionality

## 🔑 Core AI/ML Services (REQUIRED)

### 1. OpenAI API (Essential for AI features)
**Get from:** https://platform.openai.com/api-keys
**Cost:** $0.002/1K tokens for GPT-3.5, $0.02/1K tokens for GPT-4
**Usage:** Conversation AI, text analysis, lead scoring
```env
OPENAI_API_KEY="sk-..."
```

### 2. Azure Speech Services (For Voice AI)
**Get from:** https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/
**Cost:** $1 per hour of audio processed
**Usage:** Speech-to-text, text-to-speech
```env
AZURE_SPEECH_KEY="your-azure-speech-key"
AZURE_SPEECH_REGION="eastus"
```

### 3. ElevenLabs (For Premium Voice)
**Get from:** https://elevenlabs.io/
**Cost:** $5-99/month based on usage
**Usage:** High-quality voice generation
```env
ELEVENLABS_API_KEY="your-elevenlabs-key"
```

## 📈 Economic Data APIs (For Market Intelligence)

### 4. FRED API (Federal Reserve Economic Data)
**Get from:** https://fred.stlouisfed.org/docs/api/api_key.html
**Cost:** FREE
**Usage:** Economic indicators (GDP, unemployment, inflation)
```env
FRED_API_KEY="your-fred-api-key"
```

### 5. Alpha Vantage (Financial Data)
**Get from:** https://www.alphavantage.co/support/#api-key
**Cost:** FREE for 5 calls/min, $49.99/month for premium
**Usage:** Stock market data, financial indicators
```env
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
```

### 6. News API (Market Sentiment)
**Get from:** https://newsapi.org/register
**Cost:** FREE for 1000 requests/day, $449/month for business
**Usage:** News analysis for market sentiment
```env
NEWS_API_KEY="your-news-api-key"
```

## 📍 Geographic Services (For Territory Optimization)

### 7. Google Maps API
**Get from:** https://developers.google.com/maps/documentation/javascript/get-api-key
**Cost:** $200 free credit monthly, then pay-per-use
**Usage:** Geocoding, distance calculation
```env
GOOGLE_MAPS_API_KEY="your-google-maps-key"
```

## 💬 Communication Services (Optional)

### 8. Twilio (For SMS/Voice calls)
**Get from:** https://www.twilio.com/
**Cost:** $0.0075 per SMS, $0.013/minute for voice
**Usage:** Automated SMS and phone calls
```env
TWILIO_ACCOUNT_SID="your-twilio-sid"
TWILIO_AUTH_TOKEN="your-twilio-token"
TWILIO_PHONE_NUMBER="+1234567890"
```

### 9. SendGrid (For Email)
**Get from:** https://sendgrid.com/
**Cost:** FREE for 100 emails/day, $14.95/month for more
**Usage:** Automated email campaigns
```env
SENDGRID_API_KEY="your-sendgrid-key"
```

## 📊 Monitoring Services (Recommended)

### 10. Sentry (Error Tracking)
**Get from:** https://sentry.io/
**Cost:** FREE for 5K errors/month, $26/month for team
**Usage:** Error tracking and performance monitoring
```env
SENTRY_DSN="your-sentry-dsn"
```

## 💳 API Key Cost Summary

### Minimum Setup (Essential only):
- OpenAI: $50-200/month (depending on usage)
- Azure Speech: $30-100/month
- Total: ~$80-300/month

### Full Production Setup:
- All services: $150-500/month depending on scale
- Enterprise features: $500-2000/month for high volume

## 🔧 How to Set Up API Keys

### 1. Create .env file from template:
```bash
cp backend/.env.template .env
```

### 2. Edit the .env file:
```bash
nano .env
# Or use any text editor to add your API keys
```

### 3. Secure your API keys:
```bash
# Set proper permissions
chmod 600 .env

# Never commit .env to git
echo ".env" >> .gitignore
```

### 4. Test API connections:
```bash
# After deployment, test health endpoint
curl http://your-domain.com/health

# Check API documentation
curl http://your-domain.com/docs
```

## ⚠️ Security Best Practices

### API Key Security:
1. **Rotate keys regularly** (monthly for production)
2. **Set usage limits** on all API services
3. **Monitor usage** to detect unauthorized access
4. **Use environment variables** never hardcode keys
5. **Implement rate limiting** to prevent abuse
6. **Set up alerts** for unusual usage patterns

### Key Management Commands:
```bash
# Check current API usage
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/usage

# Monitor costs in real-time
# Set up billing alerts in each service dashboard
```

## 🚀 Quick Setup Script

```bash
#!/bin/bash
# quick-setup.sh - Automated API key setup

echo "🔑 Setting up API keys for AI Sales Automation..."

# Copy template
cp backend/.env.template .env

echo "✅ Template copied. Please edit .env file with your API keys:"
echo "1. OpenAI API Key (Required)"
echo "2. Azure Speech Key (Required)"
echo "3. FRED API Key (Free - Required for market analysis)"
echo "4. Google Maps API Key (Required for territory optimization)"
echo "5. Other keys as needed"

echo ""
echo "📝 Open .env file in editor:"
echo "nano .env"
echo ""
echo "🚀 After adding keys, deploy with:"
echo "./deploy.sh"
```