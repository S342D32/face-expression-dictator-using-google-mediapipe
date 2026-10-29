FROM python:3.11-slim

# System deps for mediapipe + av + psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (prod only — no torch/piper)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create logs dir
RUN mkdir -p logs

EXPOSE 8765 8766

CMD ["python", "main.py"]
