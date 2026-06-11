# 1. Choose a stable, clean Python environment
FROM python:3.10-slim

# 2. Install system-level audio dependencies (FFmpeg and libsndfile)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Establish the working folder inside the cloud container
WORKDIR /app

# 4. Copy and install your dependencies first (improves build caching speeds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your local repository files into the machine
COPY . .

# 6. Expose the mandatory Hugging Face port
EXPOSE 7860

# 7. Step into the interface directory and launch the server
WORKDIR /app/Web_Interface
CMD ["python", "app.py"]