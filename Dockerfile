# =============================================================================
# Grovio AI Image Generation API - Dockerfile
# Marketing-focused image generation with 50+ optimized models
# =============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash grovio && \
    chown -R grovio:grovio /app
USER grovio

# Default port (can be overridden by environment variable)
ENV PORT=5005

# MongoDB collection for this service
ENV MONGODB_COLLECTION=Image_Mini_App

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application using the PORT environment variable
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT}
