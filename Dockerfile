FROM python:3.10-slim

# Install system dependencies needed by pillow / torch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Cloud Run will send traffic to this port
ENV PORT=8080
EXPOSE 8080

# Start FastAPI app (main.py with variable "app")
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
