# Root-level Dockerfile for Render builds
# Uses the same base as the project's webapp Dockerfile but copies from the Hc-Fetal/webapp directory
FROM python:3.10-slim

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

# Copy webapp requirements and install
COPY Hc-Fetal/webapp/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall PyTorch CPU-only
RUN pip uninstall -y torch torchvision || true && \
    pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
RUN pip install --no-cache-dir segmentation-models-pytorch --no-deps && \
    pip install --no-cache-dir timm --no-deps

# Use gunicorn as production WSGI server so the host platform can detect the open HTTP port
RUN pip install --no-cache-dir gunicorn

# Copy app and src
COPY Hc-Fetal/webapp/ /app/
COPY Hc-Fetal/src/ /app/src/

# Create directories
RUN mkdir -p data/uploads data/processed models

EXPOSE 5000
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Use gunicorn and bind to the PORT env var set by Render (fallback to 5000)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app --workers 1"]
