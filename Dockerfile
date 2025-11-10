FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run application
CMD ["python", "-u", "groq_colab_agent_complete.py"]

# ═══════════════════════════════════════════════════════════════════
# FILE 5: docker-compose.yml
# ═══════════════════════════════════════════════════════════════════

version: '3.8'

services:
  groq-agent:
    build: .
    container_name: groq-colab-agent
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GROQ_MODEL=mixtral-8x7b-32768
      - LOG_LEVEL=INFO
    ports:
      - "8765:8765"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  redis:
    image: redis:7-alpine
    container_name: groq-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
