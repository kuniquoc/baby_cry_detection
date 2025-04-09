FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for static files and templates
RUN mkdir -p /app/static /app/templates

# Copy the static files and templates
COPY templates /app/templates
COPY static /app/static

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
