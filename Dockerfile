FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV
# opencv-python-headless usually doesn't need libgl1, but libglib2.0-0 is often required
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

# Explicitly set workers to 1 to save RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]