# Railway Quick Deploy Script

echo "🚀 Fixing Railway deployment configuration..."

# Create a simpler structure for Railway deployment
echo "
# Railway will use this Procfile
web: cd backend && python -m uvicorn main:app --host 0.0.0.0 --port \$PORT
" > Procfile

echo "
# Simplified requirements for faster build
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
motor==3.3.2
redis==5.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
requests==2.31.0
" > requirements-minimal.txt

echo "✅ Railway configuration fixed!"
echo "📝 Deploy steps:"
echo "1. Go to Railway dashboard"
echo "2. Redeploy the service" 
echo "3. Or push these changes to trigger auto-deploy"

echo "🔧 Alternative: Use this Railway template link:"
echo "https://railway.app/template/fastapi"