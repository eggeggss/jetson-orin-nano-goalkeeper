FROM python:3.10-slim

# System dependencies for OpenCV, PortAudio, and building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gcc \
    g++ \
    curl \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/data/events /app/data/persons /app/models

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/api/status || exit 1

EXPOSE 8000

CMD ["python", "main.py"]
