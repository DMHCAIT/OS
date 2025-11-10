# Cloud Deployment Guide
# Complete guide for deploying to major cloud providers

## 1. AWS Deployment (Recommended)

### AWS Services Needed:
- EC2 instances (t3.large or larger)
- RDS for MongoDB (or DocumentDB)
- ElastiCache for Redis
- Application Load Balancer
- S3 for file storage
- CloudWatch for monitoring

### AWS Setup Commands:
```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.large \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx

# 2. Install Docker on EC2
ssh -i your-key.pem ubuntu@your-ec2-ip
sudo apt update && sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu

# 3. Clone and deploy
git clone your-repo
cd advanced-ai-sales-automation
sudo ./deploy.sh
```

### AWS Cost Estimate (Monthly):
- EC2 t3.large: $67
- DocumentDB: $200
- ElastiCache: $45
- Load Balancer: $23
- Total: ~$335/month

## 2. Google Cloud Platform

### GCP Services:
- Compute Engine
- Cloud SQL
- Memorystore (Redis)
- Cloud Load Balancing
- Cloud Storage

### GCP Setup:
```bash
# Create VM instance
gcloud compute instances create ai-sales-automation \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

# SSH and deploy
gcloud compute ssh ai-sales-automation
# Follow same deployment steps
```

## 3. Microsoft Azure

### Azure Services:
- Virtual Machines
- Cosmos DB
- Cache for Redis
- Application Gateway
- Blob Storage

## 4. DigitalOcean (Budget Option)

### Recommended Droplet:
- CPU: 4 vCPUs
- RAM: 8GB
- Storage: 160GB SSD
- Cost: ~$48/month

```bash
# After creating droplet
ssh root@your-droplet-ip
apt update && apt install docker.io docker-compose git -y

# Clone and deploy
git clone your-repo
cd advanced-ai-sales-automation
chmod +x deploy.sh
./deploy.sh
```

## 5. Domain and SSL Setup

### Domain Configuration:
```bash
# Point your domain to your server IP
# A record: yourdomain.com -> your-server-ip
# A record: api.yourdomain.com -> your-server-ip

# SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com
```