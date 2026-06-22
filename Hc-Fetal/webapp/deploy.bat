@echo off
REM deploy.bat - Windows batch deployment script for Fetal HC Web Application
REM Usage: deploy.bat [dev|prod]

echo ==========================================
echo Fetal HC Web Application Deployment
echo ==========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed
    echo Please install Docker Desktop for Windows from:
    echo https://www.docker.com/products/docker-desktop/
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose is not installed
    exit /b 1
)

REM Check model file
if not exist "models\checkpoint_epoch_91.pth" (
    echo ERROR: Model file not found
    echo Please place your model at: models\checkpoint_epoch_91.pth
    exit /b 1
)

echo [OK] Docker installed
echo [OK] Docker Compose installed
echo [OK] Model file found
echo.

REM Create directories
echo Creating data directories...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\processed" mkdir data\processed
echo [OK] Data directories ready
echo.

REM Get mode parameter
set MODE=%1
if "%MODE%"=="" set MODE=dev

if "%MODE%"=="prod" (
    echo Deploying in PRODUCTION mode...
    echo.
    
    echo Stopping existing containers...
    docker-compose -f docker-compose.prod.yml down 2>nul
    
    echo Building production image...
    docker-compose -f docker-compose.prod.yml build
    
    echo Starting production containers...
    docker-compose -f docker-compose.prod.yml up -d
    
    echo.
    echo ==========================================
    echo Production deployment complete!
    echo ==========================================
    echo.
    echo Application: http://localhost
    echo API: http://localhost/api
    echo.
    echo Useful commands:
    echo   View logs: docker-compose -f docker-compose.prod.yml logs -f
    echo   Stop: docker-compose -f docker-compose.prod.yml down
    
) else (
    echo Deploying in DEVELOPMENT mode...
    echo.
    
    echo Stopping existing containers...
    docker-compose down 2>nul
    
    echo Building development image...
    docker-compose build
    
    echo Starting development containers...
    docker-compose up -d
    
    echo.
    echo ==========================================
    echo Development deployment complete!
    echo ==========================================
    echo.
    echo Application: http://localhost:5000
    echo API: http://localhost:5000/api
    echo.
    echo Useful commands:
    echo   View logs: docker-compose logs -f
    echo   Stop: docker-compose down
)

echo.
echo Waiting for application to start...
timeout /t 5 /nobreak >nul

echo.
echo Deployment script completed!
echo.
pause
