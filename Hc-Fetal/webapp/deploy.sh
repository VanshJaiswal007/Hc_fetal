#!/bin/bash

# Deployment script for Fetal HC Web Application
# Usage: ./deploy.sh [dev|prod]

set -e

MODE=${1:-dev}

echo "=========================================="
echo "Fetal HC Web Application Deployment"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if model file exists
if [ ! -f "models/checkpoint_epoch_91.pth" ]; then
    echo "ERROR: Model file not found"
    echo "Please place your model at: models/checkpoint_epoch_91.pth"
    exit 1
fi

echo "✓ Docker installed"
echo "✓ Docker Compose installed"
echo "✓ Model file found"
echo ""

# Create necessary directories
mkdir -p data/uploads
mkdir -p data/processed

if [ "$MODE" = "prod" ]; then
    echo "Deploying in PRODUCTION mode..."
    echo ""
    
    # Build production image
    echo "Building production image..."
    docker build -f Dockerfile.prod -t fetal-hc-webapp:prod .
    
    # Stop existing containers
    echo "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    
    # Start production containers
    echo "Starting production containers..."
    docker-compose -f docker-compose.prod.yml up -d
    
    echo ""
    echo "=========================================="
    echo "Production deployment complete!"
    echo "=========================================="
    echo ""
    echo "Application: http://localhost"
    echo "API: http://localhost/api"
    echo ""
    echo "View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "Stop: docker-compose -f docker-compose.prod.yml down"
    
else
    echo "Deploying in DEVELOPMENT mode..."
    echo ""
    
    # Build development image
    echo "Building development image..."
    docker-compose build
    
    # Stop existing containers
    echo "Stopping existing containers..."
    docker-compose down 2>/dev/null || true
    
    # Start development containers
    echo "Starting development containers..."
    docker-compose up -d
    
    echo ""
    echo "=========================================="
    echo "Development deployment complete!"
    echo "=========================================="
    echo ""
    echo "Application: http://localhost:5000"
    echo "API: http://localhost:5000/api"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Stop: docker-compose down"
fi

echo ""
echo "Waiting for application to start..."
sleep 5

# Health check
if curl -f http://localhost:5000/ > /dev/null 2>&1 || curl -f http://localhost/ > /dev/null 2>&1; then
    echo "✓ Application is running!"
else
    echo "⚠ Application may still be starting..."
    echo "Check logs for details"
fi

echo ""
echo "Deployment script completed!"
