#!/bin/bash
set -e

# Wait for model file to be available
echo "Checking for model file..."
if [ ! -f "/app/models/checkpoint_epoch_91.pth" ]; then
    echo "ERROR: Model file not found at /app/models/checkpoint_epoch_91.pth"
    echo "Please mount the model file as a volume:"
    echo "  -v /path/to/model:/app/models/checkpoint_epoch_91.pth"
    exit 1
fi

echo "✓ Model file found"

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
