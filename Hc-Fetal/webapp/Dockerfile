# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY webapp/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall PyTorch with CPU-only version to save space
RUN pip uninstall -y torch torchvision && \
    pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
RUN pip install --no-cache-dir segmentation-models-pytorch --no-deps && \
    pip install --no-cache-dir timm --no-deps

# Copy application code
COPY webapp/ /app/

# Copy src directory from project root
COPY src/ /app/src/

# Create necessary directories
RUN mkdir -p data/uploads data/processed models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
