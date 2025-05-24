FROM ubuntu:22.04

# Set build-time variables
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.13.0
ARG TZ=UTC

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PORT=8000

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION}*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create nonroot user and group
RUN addgroup --system --gid 65532 nonroot && \
    adduser --system --uid 65532 --ingroup nonroot nonroot

# Create necessary directories and set permissions
RUN mkdir -p /app/static /app/templates /app/data && \
    chown -R nonroot:nonroot /app

# Copy application code and static files
COPY core/ ./core/
COPY models/ ./models/
COPY services/ ./services/
COPY src/ ./src/
COPY runs/ ./runs/
COPY utils/ ./utils/
COPY websocket/ ./websocket/
COPY static/ ./static/
COPY templates/ ./templates/
COPY app.py .
COPY fcm_service.py .
COPY firebase_service.py .

# Change ownership of all copied files
RUN chown -R nonroot:nonroot /app

# Switch to nonroot user
USER nonroot

# Expose the port the app runs on
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
