# Railway Environment Variables Configuration

## 🔑 Required Environment Variables for Railway Deployment

Copy and paste these into your Railway dashboard under each service:

### 🖥️ Backend Service Environment Variables

```bash
# AI & Speech Services
OPENAI_API_KEY=your_openai_api_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_REGION=eastus

# Authentication & Security
JWT_SECRET=randomly_generated_32_character_string
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
ENVIRONMENT=production
PYTHONPATH=/app
DEBUG=false

# Database URLs (Auto-configured by Railway)
DATABASE_URL=${MongoDB.DATABASE_URL}
REDIS_URL=${Redis.REDIS_URL}

# Email & Communication
SENDGRID_API_KEY=your_sendgrid_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

# External APIs
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Monitoring
SENTRY_DSN=your_sentry_dsn_url

# File Storage
UPLOAD_MAX_SIZE=10485760
ALLOWED_EXTENSIONS=.pdf,.doc,.docx,.txt,.csv,.mp3,.wav
```

### 🎨 Frontend Service Environment Variables

```bash
# API Configuration  
REACT_APP_API_URL=${backend.RAILWAY_PUBLIC_DOMAIN}
REACT_APP_ENVIRONMENT=production

# Features
REACT_APP_ENABLE_VOICE=true
REACT_APP_ENABLE_ML=true
REACT_APP_ENABLE_DASHBOARD=true

# External Services
REACT_APP_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

## 🗄️ Database Services

Railway will automatically configure:

- **MongoDB**: For main data storage
- **Redis**: For caching and sessions

## 🚀 Quick Setup Commands

### 1. Install Railway CLI
```bash
npm install -g @railway/cli
```

### 2. Login and Initialize
```bash
railway login
cd /Users/rubeenakhan/Desktop/OS
railway init
```

### 3. Add Databases
```bash
# Add MongoDB
railway add --database mongodb

# Add Redis  
railway add --database redis
```

### 4. Deploy Services
```bash
# Deploy backend
railway up --service backend

# Deploy frontend  
railway up --service frontend
```

### 5. Set Environment Variables
```bash
# Set variables via CLI (optional)
railway env set OPENAI_API_KEY=your_key_here
railway env set AZURE_SPEECH_KEY=your_key_here

# Or use the Railway dashboard (recommended)
railway open --dashboard
```

## 🔗 Service Connections

Railway automatically handles:
- ✅ **Internal networking** between services
- ✅ **Database connections** with connection strings
- ✅ **SSL certificates** for HTTPS
- ✅ **Load balancing** and scaling
- ✅ **Health checks** and monitoring

## 📊 Expected Costs

| Service | Resource | Monthly Cost |
|---------|----------|--------------|
| Backend | 1GB RAM, 1 CPU | $5-10 |
| Frontend | Static hosting | $0-5 |
| MongoDB | 512MB storage | $5 |
| Redis | 25MB cache | $5 |
| **Total** |  | **$15-25** |

## 🔧 Advanced Configuration

### Custom Domains
```bash
# Add custom domain
railway domain add yourdomain.com

# Add subdomain for API
railway domain add api.yourdomain.com --service backend
```

### Scaling Configuration
```bash
# Auto-scaling settings
RAILWAY_DEPLOYMENT_REPLICA_COUNT=1-5
RAILWAY_DEPLOYMENT_AUTOSCALE_TARGET_CPU=80
RAILWAY_DEPLOYMENT_AUTOSCALE_TARGET_MEMORY=80
```

### Health Checks
```bash
# Backend health check
RAILWAY_HEALTHCHECK_PATH=/health
RAILWAY_HEALTHCHECK_TIMEOUT=30s
RAILWAY_HEALTHCHECK_INTERVAL=30s

# Frontend health check  
RAILWAY_HEALTHCHECK_PATH=/
RAILWAY_HEALTHCHECK_TIMEOUT=10s
```

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   railway logs --deployment latest
   ```

2. **Environment Variables**
   ```bash
   # List current env vars
   railway env list
   
   # Check specific variable
   railway env get OPENAI_API_KEY
   ```

3. **Database Connection**
   ```bash
   # Check database status
   railway status
   
   # Connect to database
   railway connect mongodb
   ```

### Useful Commands

```bash
# View all services
railway status

# View logs
railway logs --service backend
railway logs --service frontend

# Open in browser
railway open

# SSH into service
railway shell

# Restart service
railway restart --service backend
```

## ✅ Deployment Checklist

- [ ] Railway CLI installed
- [ ] Project initialized (`railway init`)
- [ ] MongoDB database added
- [ ] Redis cache added  
- [ ] Backend environment variables set
- [ ] Frontend environment variables set
- [ ] API keys obtained and configured
- [ ] Services deployed successfully
- [ ] Health checks passing
- [ ] Custom domain configured (optional)

## 🎯 Final Steps

1. **Test your deployment**:
   ```bash
   # Check if services are running
   railway open
   ```

2. **Monitor performance**:
   ```bash
   # View metrics
   railway logs --tail
   ```

3. **Set up monitoring alerts** in Railway dashboard

4. **Configure custom domain** if needed

Your AI Lead Management System will be live and ready for production use! 🚀