#!/bin/bash

# 🔑 API Keys Setup Guide for AI Sales Automation
# ================================================

echo "🔑 API Keys Setup Guide for AI Sales Automation"
echo "=============================================="
echo ""

echo "📋 REQUIRED API KEYS FOR FULL FUNCTIONALITY:"
echo ""

echo "1. 🤖 OPENAI API KEY (Required for AI conversations)"
echo "   - Go to: https://platform.openai.com/api-keys"
echo "   - Create account or sign in"
echo "   - Generate new API key"
echo "   - Copy to .env file: OPENAI_API_KEY=\"sk-your-key-here\""
echo ""

echo "2. 🎤 ELEVENLABS API KEY (Required for voice AI)"
echo "   - Go to: https://elevenlabs.io/"
echo "   - Create account"
echo "   - Navigate to Profile > API Keys"
echo "   - Generate new API key"
echo "   - Copy to .env file: ELEVENLABS_API_KEY=\"your-key-here\""
echo ""

echo "3. 📞 TWILIO CREDENTIALS (Required for automatic calling)"
echo "   - Go to: https://www.twilio.com/"
echo "   - Create account"
echo "   - Get Account SID and Auth Token from Console"
echo "   - Buy a phone number in Phone Numbers section"
echo "   - Add to .env file:"
echo "     TWILIO_ACCOUNT_SID=\"your-account-sid\""
echo "     TWILIO_AUTH_TOKEN=\"your-auth-token\""
echo "     TWILIO_PHONE_NUMBER=\"+1234567890\""
echo ""

echo "🔧 OPTIONAL INTEGRATIONS:"
echo ""

echo "4. 📧 EMAIL SETUP (For automated emails)"
echo "   - Use Gmail App Password or SMTP service"
echo "   - Add to .env file:"
echo "     SMTP_USERNAME=\"your-email@gmail.com\""
echo "     SMTP_PASSWORD=\"your-app-password\""
echo ""

echo "5. 🔗 CRM INTEGRATION (Optional)"
echo "   - Salesforce, HubSpot, or Pipedrive API keys"
echo "   - Add to .env file as needed"
echo ""

echo "💰 ESTIMATED COSTS:"
echo "   - OpenAI GPT-4: ~$0.03 per 1K tokens (~$0.50 per conversation)"
echo "   - ElevenLabs Voice: ~$0.18 per 1K characters (~$0.25 per call)"
echo "   - Twilio Calls: ~$0.013 per minute (~$0.10 per 5-min call)"
echo "   - Total per call: ~$0.85 (vs $50-200 for human sales calls)"
echo ""

echo "✅ QUICK SETUP CHECKLIST:"
echo "   □ Get OpenAI API key"
echo "   □ Get ElevenLabs API key"
echo "   □ Get Twilio account and phone number"
echo "   □ Update .env file with real keys"
echo "   □ Test with a few sample calls"
echo "   □ Upload your historical sales data"
echo "   □ Train models with your data"
echo "   □ Start automatic calling!"
echo ""

echo "🚀 After adding your API keys, restart the system:"
echo "   cd /Users/rubeenakhan/Desktop/OS/backend"
echo "   ./quick_setup_updated.sh"
echo ""

echo "📞 Your AI Sales Automation will be ready for automatic calling!"