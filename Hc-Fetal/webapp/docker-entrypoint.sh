#!/bin/bash
set -e

# Wait for model file to be available
echo "Checking for model file..."
if [ ! -f "/app/models/checkpoint_epoch_91.pth" ]; then
    echo "Model file not found at /app/models/checkpoint_epoch_91.pth"
    if [ -n "$MODEL_URL" ]; then
        echo "Attempting to download model from MODEL_URL..."
        mkdir -p /app/models
        # Try downloading with curl or wget
        if command -v curl > /dev/null 2>&1; then
            curl -fsSL "$MODEL_URL" -o /app/models/checkpoint_epoch_91.pth || true
        fi
        if [ ! -f "/app/models/checkpoint_epoch_91.pth" ] && command -v wget > /dev/null 2>&1; then
            wget -q "$MODEL_URL" -O /app/models/checkpoint_epoch_91.pth || true
        fi

        if [ -f "/app/models/checkpoint_epoch_91.pth" ]; then
            echo "✓ Model downloaded to /app/models/checkpoint_epoch_91.pth"
        else
            echo "ERROR: Could not download model from MODEL_URL"
            echo "Please provide the model by mounting it as a volume:"
            echo "  -v /path/to/model:/app/models/checkpoint_epoch_91.pth"
            exit 1
        fi
    else
        echo "ERROR: Model file not found and MODEL_URL not provided."
        echo "Please mount the model file as a volume:"
        echo "  -v /path/to/model:/app/models/checkpoint_epoch_91.pth"
        exit 1
    fi
else
    echo "✓ Model file found"
fi

# Create data directories if they don't exist
mkdir -p /app/data/uploads
mkdir -p /app/data/processed

echo "✓ Data directories ready"

# Check if running in production mode
if [ "$FLASK_ENV" = "production" ]; then
    echo "Starting in PRODUCTION mode with Gunicorn..."
    exec gunicorn -w ${WORKERS:-4} -b 0.0.0.0:5000 --timeout 300 app:app
else
    echo "Starting in DEVELOPMENT mode..."
    exec python app.py
fi
