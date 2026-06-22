# deploy.ps1 - Windows PowerShell deployment script for Fetal HC Web Application
# Usage: .\deploy.ps1 [dev|prod]

param(
    [string]$Mode = "dev"
)

Write-Host "=========================================="
Write-Host "Fetal HC Web Application Deployment"
Write-Host "=========================================="
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed" -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows from:"
    Write-Host "https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# Check if Docker Compose is installed
try {
    $composeVersion = docker-compose --version
    Write-Host "[OK] Docker Compose installed: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker Compose is not installed" -ForegroundColor Red
    exit 1
}

# Check if model file exists
if (-not (Test-Path "models\checkpoint_epoch_91.pth")) {
    Write-Host "ERROR: Model file not found" -ForegroundColor Red
    Write-Host "Please place your model at: models\checkpoint_epoch_91.pth" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Model file found" -ForegroundColor Green
Write-Host ""

# Create necessary directories
Write-Host "Creating data directories..."
New-Item -ItemType Directory -Force -Path "data\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
Write-Host "[OK] Data directories ready" -ForegroundColor Green
Write-Host ""

if ($Mode -eq "prod") {
    Write-Host "Deploying in PRODUCTION mode..." -ForegroundColor Yellow
    Write-Host ""
    
    # Stop existing containers
    Write-Host "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down 2>$null
    
    # Build production image
    Write-Host "Building production image..."
    docker-compose -f docker-compose.prod.yml build
    
    # Start production containers
    Write-Host "Starting production containers..."
    docker-compose -f docker-compose.prod.yml up -d
    
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "Production deployment complete!" -ForegroundColor Green
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Application: http://localhost" -ForegroundColor Cyan
    Write-Host "API: http://localhost/api" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Useful commands:"
    Write-Host "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
    Write-Host "  Stop: docker-compose -f docker-compose.prod.yml down"
    
} else {
    Write-Host "Deploying in DEVELOPMENT mode..." -ForegroundColor Yellow
    Write-Host ""
    
    # Stop existing containers
    Write-Host "Stopping existing containers..."
    docker-compose down 2>$null
    
    # Build development image
    Write-Host "Building development image..."
    docker-compose build
    
    # Start development containers
    Write-Host "Starting development containers..."
    docker-compose up -d
    
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "Development deployment complete!" -ForegroundColor Green
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Application: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "API: http://localhost:5000/api" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Useful commands:"
    Write-Host "  View logs: docker-compose logs -f"
    Write-Host "  Stop: docker-compose down"
}

Write-Host ""
Write-Host "Waiting for application to start..."
Start-Sleep -Seconds 5

# Health check
Write-Host "Checking application health..."
try {
    if ($Mode -eq "prod") {
        $response = Invoke-WebRequest -Uri "http://localhost/" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    } else {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    }
    Write-Host "[OK] Application is running!" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Application may still be starting..." -ForegroundColor Yellow
    Write-Host "Check logs for details: docker-compose logs -f"
}

Write-Host ""
Write-Host "Deployment script completed!" -ForegroundColor Green
