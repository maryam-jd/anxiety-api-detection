FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by opencv and mediapipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]