# 1. Base Python environment
FROM python:3.10-slim

# 2. Install heavy-duty Linux build tools and speech dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Establish the working folder
WORKDIR /app

# 4. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project
COPY . .

# 6. Expose the Hugging Face port
EXPOSE 7860

# 7. Start the server
WORKDIR /app/Web_Interface
CMD ["python", "app.py"]